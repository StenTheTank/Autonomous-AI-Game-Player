import type { IndustryType, PlayerId } from './types';
import { tileSpec } from './data/industries';
import { levelForSpace } from './income';
import { HUMAN, activePlayerId, buildAutomaDeck, dealEra, log, playerLabel, type Card, type GameState } from './state';
import {
  applyBuild, applyDevelop, applyLoan, applyNetworkSingle, applyScout, cardAllowsBuild,
  sellBuilding, type BeerSource, type BuildOption, type NetworkOption, type SellableBuilding,
} from './actions';
import { runAutomaTurn } from './mautoma/bot';
import { canalEraCleanup, scoreFlippedIndustries, scoreLinks } from './scoring';

export type HumanAction =
  | { type: 'build'; cardIdx: number; option: BuildOption }
  | { type: 'network'; cardIdx: number; option: NetworkOption }
  | { type: 'sell'; cardIdx: number; sales: { sale: SellableBuilding; beer: BeerSource[] }[] }
  | { type: 'develop'; cardIdx: number; industries: IndustryType[] }
  | { type: 'loan'; cardIdx: number }
  | { type: 'scout'; cardIdx: number; extraDiscards: [number, number] }
  | { type: 'pass'; cardIdx: number };

/** Validate the card is usable for this action (build checks specifically). */
function checkCard(state: GameState, player: PlayerId, action: HumanAction): Card {
  const card = state.players[player].hand[action.cardIdx];
  if (!card) throw new Error('No such card in hand');
  if (action.type === 'build' && !cardAllowsBuild(state, player, card, action.option.city, action.option.industry)) {
    throw new Error('Card does not allow this build');
  }
  return card;
}

function discardCard(state: GameState, player: PlayerId, idx: number): void {
  const removed = state.players[player].hand.splice(idx, 1)[0];
  if (removed && removed.kind !== 'wildLocation' && removed.kind !== 'wildIndustry') {
    state.players[player].discard.push(removed);
  }
}

/** Apply one player action (for `activePlayerId(state)`). */
export function applyHumanAction(state: GameState, action: HumanAction): void {
  const actor = activePlayerId(state);
  checkCard(state, actor, action);
  switch (action.type) {
    case 'build':
      applyBuild(state, actor, action.option);
      break;
    case 'network':
      applyNetworkSingle(state, actor, action.option);
      break;
    case 'sell':
      if (action.sales.length === 0) throw new Error('Sell needs at least one sale');
      for (const { sale, beer } of action.sales) sellBuilding(state, actor, sale, beer);
      break;
    case 'develop':
      applyDevelop(state, actor, action.industries);
      break;
    case 'loan':
      applyLoan(state, actor);
      break;
    case 'scout': {
      applyScout(state, action.extraDiscards, actor);
      break;
    }
    case 'pass':
      log(state, `${playerLabel(state, actor)} passed.`);
      break;
  }
  discardCardForAction(state, actor, action);
  state.actionsLeft -= 1;
  if (state.actionsLeft === 0) {
    if (state.mode === '2p') {
      endTwoPlayerTurn(state, actor);
    } else {
      endHumanTurn(state);
    }
  }
}

function discardCardForAction(state: GameState, player: PlayerId, action: HumanAction): void {
  if (action.type === 'scout') {
    let idx = action.cardIdx;
    for (const removed of action.extraDiscards) {
      if (removed < action.cardIdx) idx -= 1;
    }
    discardCard(state, player, idx);
    return;
  }
  discardCard(state, player, action.cardIdx);
}

function refillHand(state: GameState, player: PlayerId): void {
  const hand = state.players[player].hand;
  while (hand.length < 8 && state.drawPile.length > 0) {
    hand.push(state.drawPile.pop()!);
  }
}

function endTwoPlayerTurn(state: GameState, actor: PlayerId): void {
  refillHand(state, actor);
  if (state.turnIndex === 0) {
    state.turnIndex = 1;
    state.activePlayer = state.turnOrder[1];
    state.actionsLeft = state.era === 'canal' && state.turn === 1 ? 1 : 2;
    return;
  }
  endTwoPlayerRound(state);
}

function endTwoPlayerRound(state: GameState): void {
  // Official turn order rule: least money spent goes first next round; ties preserve relative order
  state.turnOrder = [...state.turnOrder].sort(
    (a, b) => state.players[a].spent - state.players[b].spent,
  ) as [PlayerId, PlayerId];
  state.players[0].spent = 0;
  state.players[1].spent = 0;

  const isLastRound =
    state.drawPile.length === 0 &&
    state.players[0].hand.length === 0 &&
    state.players[1].hand.length === 0;

  if (!isLastRound) {
    for (const pid of state.turnOrder) {
      const income = levelForSpace(state.players[pid].incomeSpace);
      state.players[pid].money += income;
      if (income !== 0) {
        log(state, `Income (${playerLabel(state, pid)}): ${income >= 0 ? '+' : ''}£${income}.`);
      }
      if (state.players[pid].money < 0) shortfall(state, pid);
    }
    state.turn += 1;
    state.turnIndex = 0;
    state.activePlayer = state.turnOrder[0];
    state.actionsLeft = 2;
    return;
  }
  endEra(state);
}

function endHumanTurn(state: GameState): void {
  refillHand(state, HUMAN);
  // Automa turn
  runAutomaTurn(state);
  endRound(state);
}

function endRound(state: GameState): void {
  state.players[HUMAN].spent = 0;
  const isLastRound = state.drawPile.length === 0 && state.players[HUMAN].hand.length === 0;
  if (!isLastRound) {
    const income = levelForSpace(state.players[HUMAN].incomeSpace);
    state.players[HUMAN].money += income;
    if (income !== 0) log(state, `Income: ${income >= 0 ? '+' : ''}£${income}.`);
    if (state.players[HUMAN].money < 0) shortfall(state, HUMAN);
    state.turn += 1;
    state.actionsLeft = 2;
    return;
  }
  endEra(state);
}

/** Forced tile sales when income leaves `player` below £0. */
function shortfall(state: GameState, player: PlayerId = HUMAN): void {
  const p = state.players[player];
  log(state, `Shortfall! ${playerLabel(state, player)} owes £${-p.money}.`);
  const owned: { city: keyof typeof state.board; slot: number; value: number }[] = [];
  for (const [city, slots] of Object.entries(state.board)) {
    slots.forEach((tile, slot) => {
      if (tile?.owner === player) {
        owned.push({
          city: city as keyof typeof state.board,
          slot,
          value: Math.floor(tileSpec(tile.industry, tile.level).cost / 2),
        });
      }
    });
  }
  owned.sort((a, b) => a.value - b.value);
  for (const o of owned) {
    if (p.money >= 0) break;
    const tile = state.board[o.city][o.slot]!;
    state.board[o.city][o.slot] = null;
    p.money += o.value;
    log(state, `Forced sale (${playerLabel(state, player)}): ${tile.industry} L${tile.level} in ${o.city} for £${o.value}.`);
  }
  if (p.money < 0) {
    const penalty = Math.min(p.vp, -p.money);
    p.vp -= penalty;
    log(state, `Still short (${playerLabel(state, player)}): -${penalty} VP.`);
    p.money = 0;
  }
}

export function compareTwoPlayerWinner(state: GameState): 0 | 1 | 'tie' {
  const p0 = state.players[0];
  const p1 = state.players[1];
  if (p0.vp !== p1.vp) return p0.vp > p1.vp ? 0 : 1;
  const inc0 = levelForSpace(p0.incomeSpace);
  const inc1 = levelForSpace(p1.incomeSpace);
  if (inc0 !== inc1) return inc0 > inc1 ? 0 : 1;
  if (p0.money !== p1.money) return p0.money > p1.money ? 0 : 1;
  return 'tie';
}

function endEra(state: GameState): void {
  log(state, `--- End of ${state.era === 'canal' ? 'Canal' : 'Rail'} Era ---`);
  scoreLinks(state);
  scoreFlippedIndustries(state);
  if (state.era === 'canal') {
    canalEraCleanup(state);
    state.era = 'rail';
    state.turn = 1;
    state.turnIndex = 0;
    state.activePlayer = state.mode === '2p' ? state.turnOrder[0] : HUMAN;
    state.actionsLeft = 2;
    dealEra(state, 'rail');
    if (state.mode !== '2p') {
      state.automaDeck = [];
      state.automaDiscard = [];
      resetAutomaDeck(state);
    }
    log(state, 'Rail Era begins.');
    return;
  }
  state.gameOver = true;
  const [p0, p1] = state.players;
  if (state.mode === '2p') {
    const winner = compareTwoPlayerWinner(state);
    const verdict =
      winner === 'tie' ? 'Tie!' : `${playerLabel(state, winner)} wins!`;
    log(
      state,
      `Final: ${playerLabel(state, 0)} ${p0.vp} VP — ${playerLabel(state, 1)} ${p1.vp} VP. ${verdict}`,
    );
  } else {
    log(
      state,
      `Final: you ${p0.vp} VP — Automa ${p1.vp} VP. ${p0.vp > p1.vp ? 'You win!' : p0.vp === p1.vp ? 'Tie.' : 'Automa wins.'}`,
    );
  }
}

function resetAutomaDeck(state: GameState): void {
  state.automaDeck = buildAutomaDeck(state.rng);
}
