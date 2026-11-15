import type { CityId, Era, IndustryType, LocationId, MerchantId, MerchantTileKind, PlayerId } from './types';
import { CITIES, INDUSTRY_CARDS, LOCATION_CARDS, MERCHANTS, MERCHANT_TILE_GROUPS, MERCHANT_TILE_MIX } from './data/board';
import { INDUSTRY_TRACKS } from './data/industries';
import { COAL_MARKET, IRON_MARKET } from './market';
import { STARTING_SPACE } from './income';
import { MAUTOMA_CARDS, MAUTOMA_MATS, type MautomaDifficulty } from './mautoma/cards';
import { makeRng, shuffle, type Rng } from './rng';

export const HUMAN: PlayerId = 0;
export const AUTOMA: PlayerId = 1;

export type Card =
  | { kind: 'location'; city: CityId }
  | { kind: 'industry'; industries: readonly IndustryType[] }
  | { kind: 'wildLocation' }
  | { kind: 'wildIndustry' };

export interface PlacedTile {
  owner: PlayerId;
  industry: IndustryType;
  level: number;
  flipped: boolean;
  resources: number;
}

export interface MerchantState {
  id: MerchantId;
  tiles: MerchantTileKind[];
  /** Beer barrel available per non-blank tile slot */
  beer: boolean[];
}

export type GameMode = 'solo' | '2p';

export interface PlayerState {
  money: number;
  spent: number;
  incomeSpace: number;
  vp: number;
  hand: Card[];
  discard: Card[];
  /** Tiles remaining on the mat: count per level (index = level-1) */
  mat: Record<IndustryType, number[]>;
}

export interface GameState {
  mode: GameMode;
  seed: number;
  rng: Rng;
  difficulty: MautomaDifficulty;
  era: Era;
  /** Round number within the era, 1-based */
  turn: number;
  /** Active player ID (0 in solo; 0 or 1 in 2p) */
  activePlayer: PlayerId;
  /** Turn order for the current round ([0, 1] in solo; spent-ordered in 2p) */
  turnOrder: [PlayerId, PlayerId];
  /** Index inside turnOrder (0 = first player of round, 1 = second player of round) */
  turnIndex: 0 | 1;
  /** Optional display labels for Player 0 and Player 1 */
  playerNames?: [string, string];
  /** Actions remaining for the active player this turn */
  actionsLeft: number;
  players: [PlayerState, PlayerState];
  board: Record<CityId, (PlacedTile | null)[]>;
  /** linkId -> owner, or null if unbuilt */
  links: Record<string, PlayerId | null>;
  coalCubes: number;
  ironCubes: number;
  merchants: MerchantState[];
  drawPile: Card[];
  /** Mautoma deck: card ids in draw order (index 0 = next to draw) */
  automaDeck: number[];
  automaDiscard: number[];
  automaLastCity: CityId | null;
  automaFirstTurnDone: boolean;
  gameOver: boolean;
  log: string[];
}

export function isHumanRulesPlayer(state: GameState, player: PlayerId): boolean {
  return state.mode === '2p' ? true : player === HUMAN;
}

export function activePlayerId(state: GameState): PlayerId {
  return state.mode === '2p' ? (state.activePlayer ?? HUMAN) : HUMAN;
}

export function playerLabel(state: GameState, player: PlayerId): string {
  const custom = state.playerNames?.[player]?.trim();
  if (custom) return custom;
  if (state.mode === '2p') return player === 0 ? 'Player 1' : 'Player 2';
  return player === HUMAN ? 'You' : 'Automa';
}

export function emptyBoard(): Record<CityId, (PlacedTile | null)[]> {
  const board = {} as Record<CityId, (PlacedTile | null)[]>;
  for (const city of Object.values(CITIES)) {
    board[city.id] = city.slots.map(() => null);
  }
  return board;
}

function standardMat(): Record<IndustryType, number[]> {
  const mat = {} as Record<IndustryType, number[]>;
  for (const [industry, tiles] of Object.entries(INDUSTRY_TRACKS)) {
    const track: number[] = [];
    for (const t of tiles) track[t.level - 1] = t.count;
    mat[industry as IndustryType] = track;
  }
  return mat;
}

function automaMat(difficulty: MautomaDifficulty): Record<IndustryType, number[]> {
  const mat = {} as Record<IndustryType, number[]>;
  for (const [industry, counts] of Object.entries(MAUTOMA_MATS[difficulty])) {
    mat[industry as IndustryType] = [...counts];
  }
  return mat;
}

/** Build the 2-player human deck (the Mautoma uses the 2p card set). */
export function buildHumanDeck(rng: Rng): Card[] {
  const cards: Card[] = [];
  for (const [city, count] of Object.entries(LOCATION_CARDS[2])) {
    for (let i = 0; i < count; i++) cards.push({ kind: 'location', city: city as CityId });
  }
  for (const group of INDUSTRY_CARDS[2]) {
    for (let i = 0; i < group.count; i++) cards.push({ kind: 'industry', industries: group.industries });
  }
  return shuffle(rng, cards);
}

/**
 * Mautoma deck prep (§5.2): bottom-up 4A+3B+3C, then 1 each, then the
 * remaining 9; the deck is then flipped, so draw order is: the ten
 * (4A+3B+3C) first, then the three, then the nine.
 */
export function buildAutomaDeck(rng: Rng): number[] {
  const byGroup = (g: string) => MAUTOMA_CARDS.filter((c) => c.group === g).map((c) => c.id);
  const pools = { a: shuffle(rng, byGroup('a')), b: shuffle(rng, byGroup('b')), c: shuffle(rng, byGroup('c')) };
  const ten = shuffle(rng, [...pools.a.splice(0, 4), ...pools.b.splice(0, 3), ...pools.c.splice(0, 3)]);
  const three = shuffle(rng, [pools.a.shift()!, pools.b.shift()!, pools.c.shift()!]);
  const nine = shuffle(rng, [...pools.a, ...pools.b, ...pools.c]);
  return [...ten, ...three, ...nine];
}

function setupMerchants(rng: Rng): MerchantState[] {
  const mix = shuffle(rng, MERCHANT_TILE_MIX[2]);
  const merchants: MerchantState[] = [];
  let i = 0;
  for (const group of MERCHANT_TILE_GROUPS[2]) {
    const spec = MERCHANTS[group.location];
    const tiles = mix.slice(i, i + group.count);
    i += group.count;
    merchants.push({
      id: spec.id,
      tiles,
      beer: tiles.map((t) => t !== 'blank'),
    });
  }
  return merchants;
}

export interface NewGameOptions {
  mode?: GameMode;
  playerNames?: [string, string];
  firstPlayer?: PlayerId;
}

/**
 * Deal an era's hand + draw pile from the seeded official 40-card 2-player deck:
 * - In '2p' mode (official Roxley 2-player rules):
 *   * Canal Era: 1 card per player is placed face-down into their personal discard
 *     pile (38 cards in play -> 8 to P1, 8 to P2, 22 in drawPile; 1 action on Round 1).
 *   * Rail Era: all 40 cards in play -> 8 to P1, 8 to P2, 24 in drawPile (2 actions all 10 rounds).
 * - In 'solo' mode (Mautoma solo setup): 19 canal / 20 rail cards.
 */
export function dealEra(state: GameState, era: Era): void {
  const deck = buildHumanDeck(state.rng);
  if (state.mode === '2p') {
    const [first, second] = state.turnOrder;
    state.players[0].hand = [];
    state.players[1].hand = [];
    state.players[0].discard = [];
    state.players[1].discard = [];
    if (era === 'canal') {
      state.players[first].discard.push(deck[0]);
      state.players[second].discard.push(deck[1]);
      state.players[first].hand = deck.slice(2, 10);
      state.players[second].hand = deck.slice(10, 18);
      state.drawPile = deck.slice(18);
    } else {
      state.players[first].hand = deck.slice(0, 8);
      state.players[second].hand = deck.slice(8, 16);
      state.drawPile = deck.slice(16);
    }
    return;
  }
  const eraCount = era === 'canal' ? 19 : 20;
  const eraCards = deck.slice(0, eraCount);
  state.players[HUMAN].hand = eraCards.slice(0, 8);
  state.players[HUMAN].discard = [];
  state.drawPile = eraCards.slice(8);
}

export function newGame(
  seed: number,
  difficulty: MautomaDifficulty = 'easy',
  opts?: NewGameOptions,
): GameState {
  const rng = makeRng(seed);
  const mode: GameMode = opts?.mode ?? 'solo';
  const merchants = setupMerchants(rng);

  if (mode === '2p') {
    const seededOrder = shuffle(rng, [0, 1] as const) as [PlayerId, PlayerId];
    const turnOrder: [PlayerId, PlayerId] =
      opts?.firstPlayer === 0 || opts?.firstPlayer === 1
        ? [opts.firstPlayer, (1 - opts.firstPlayer) as PlayerId]
        : seededOrder;
    const state: GameState = {
      mode: '2p',
      seed,
      rng,
      difficulty,
      era: 'canal',
      turn: 1,
      activePlayer: turnOrder[0],
      turnOrder,
      turnIndex: 0,
      playerNames: opts?.playerNames,
      actionsLeft: 1, // Official Canal Era Round 1: 1 action per player
      players: [
        { money: 17, spent: 0, incomeSpace: STARTING_SPACE, vp: 0, hand: [], discard: [], mat: standardMat() },
        { money: 17, spent: 0, incomeSpace: STARTING_SPACE, vp: 0, hand: [], discard: [], mat: standardMat() },
      ],
      board: emptyBoard(),
      links: {},
      coalCubes: COAL_MARKET.initialCubes,
      ironCubes: IRON_MARKET.initialCubes,
      merchants,
      drawPile: [],
      automaDeck: [],
      automaDiscard: [],
      automaLastCity: null,
      automaFirstTurnDone: false,
      gameOver: false,
      log: [],
    };
    dealEra(state, 'canal');
    state.log.push(
      `New 2-player game — seed ${seed}. Turn order: ${playerLabel(state, turnOrder[0])} then ${playerLabel(state, turnOrder[1])}. Canal Era begins.`,
    );
    return state;
  }

  const state: GameState = {
    mode: 'solo',
    seed,
    rng,
    difficulty,
    era: 'canal',
    turn: 1,
    activePlayer: HUMAN,
    turnOrder: [HUMAN, AUTOMA],
    turnIndex: 0,
    playerNames: opts?.playerNames,
    actionsLeft: 1, // first turn: one action only
    players: [
      { money: 17, spent: 0, incomeSpace: STARTING_SPACE, vp: 0, hand: [], discard: [], mat: standardMat() },
      { money: 0, spent: 0, incomeSpace: STARTING_SPACE, vp: 0, hand: [], discard: [], mat: automaMat(difficulty) },
    ],
    board: emptyBoard(),
    links: {},
    coalCubes: COAL_MARKET.initialCubes,
    ironCubes: IRON_MARKET.initialCubes,
    merchants,
    drawPile: [],
    automaDeck: buildAutomaDeck(rng),
    automaDiscard: [],
    automaLastCity: null,
    automaFirstTurnDone: false,
    gameOver: false,
    log: [],
  };
  dealEra(state, 'canal');
  state.log.push(`New game — ${difficulty} difficulty, seed ${seed}. Canal Era begins.`);
  return state;
}

export function log(state: GameState, message: string): void {
  state.log.push(message);
}

export function cityName(id: LocationId): string {
  return id in CITIES ? CITIES[id as CityId].name : MERCHANTS[id as MerchantId].name;
}
