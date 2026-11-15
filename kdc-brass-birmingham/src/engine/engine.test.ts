import { describe, expect, it } from 'vitest';
import { AUTOMA, HUMAN, newGame } from './state';
import { applyHumanAction } from './game';
import { legalBuilds, legalDevelops, legalNetworks } from './options';
import { developCost, networkDoubleOption } from './actions';
import { levelForSpace } from './income';

describe('game setup', () => {
  it('deals 8 cards and an 11-card canal draw pile', () => {
    const state = newGame(42, 'easy');
    expect(state.players[HUMAN].hand).toHaveLength(8);
    expect(state.drawPile).toHaveLength(11);
    expect(state.actionsLeft).toBe(1);
    expect(state.players[HUMAN].money).toBe(17);
    expect(levelForSpace(state.players[HUMAN].incomeSpace)).toBe(0);
  });

  it('prepares a 22-card automa deck with the 10/3/9 structure', () => {
    const state = newGame(7, 'medium');
    expect(state.automaDeck).toHaveLength(22);
    expect(new Set(state.automaDeck).size).toBe(22);
  });

  it('is deterministic for a given seed', () => {
    const a = newGame(123, 'hard');
    const b = newGame(123, 'hard');
    expect(a.automaDeck).toEqual(b.automaDeck);
    expect(a.players[HUMAN].hand).toEqual(b.players[HUMAN].hand);
    expect(a.merchants).toEqual(b.merchants);
  });

  it('gives the Automa a difficulty-adjusted mat', () => {
    const easy = newGame(1, 'easy');
    const hard = newGame(1, 'hard');
    expect(easy.players[AUTOMA].mat.coal[0]).toBe(0);
    expect(easy.players[AUTOMA].mat.cotton[0]).toBe(2);
    expect(hard.players[AUTOMA].mat.goods[0]).toBe(0);
    expect(hard.players[AUTOMA].mat.iron[0]).toBe(0);
  });
});

describe('first turn', () => {
  it('offers builds on the first turn and runs a full round after one action', () => {
    const state = newGame(42, 'easy');
    const builds = legalBuilds(state);
    expect(builds.length).toBeGreaterThan(0);
    const before = state.players[HUMAN].hand.length;
    applyHumanAction(state, { type: 'build', cardIdx: builds[0].cardIdx, option: builds[0].option });
    // One action on turn 1 → automa ran, income round happened, turn 2 begins
    expect(state.turn).toBe(2);
    expect(state.actionsLeft).toBe(2);
    expect(state.players[HUMAN].hand.length).toBe(8);
    expect(before).toBe(8);
  });

  it('canal era: no rail-only links are offered', () => {
    const state = newGame(42, 'easy');
    // Give the player a network first so links are legal at all
    const builds = legalBuilds(state);
    applyHumanAction(state, { type: 'build', cardIdx: builds[0].cardIdx, option: builds[0].option });
    for (const choice of legalNetworks(state)) {
      expect(choice.option.linkIds.every((id) => !['belper-leek', 'coventry-nuneaton'].includes(id))).toBe(true);
    }
  });
});

describe('loan', () => {
  it('adds £30 and drops three income levels', () => {
    const state = newGame(42, 'easy');
    applyHumanAction(state, { type: 'loan', cardIdx: 0 });
    // Turn 1 had a single action; the round then ended with income -3
    expect(state.players[HUMAN].money).toBe(17 + 30 - 3);
    expect(levelForSpace(state.players[HUMAN].incomeSpace)).toBe(-3);
  });
});

describe('same-industry double develop', () => {
  it('allows developing the same industry twice for 2 iron and blocks lightbulb pottery', () => {
    const state = newGame(42, 'easy');
    state.actionsLeft = 2; // Keep turn open so Automa doesn't run after action 1
    const cottonChoice = legalDevelops(state).find((d) => d.industries[0] === 'cotton');
    expect(cottonChoice?.maxCount).toBe(2);

    // Pottery L1 has a lightbulb (canDevelop === false), so pottery shouldn't be developable at start
    const potteryChoice = legalDevelops(state).find((d) => d.industries[0] === 'pottery');
    expect(potteryChoice).toBeUndefined();
    expect(() => applyHumanAction(state, { type: 'develop', cardIdx: 0, industries: ['pottery'] })).toThrow();

    // Even if Pottery L1 is removed, Pottery L2 (count 1) is followed by L3 lightbulb -> maxCount === 1
    state.players[HUMAN].mat.pottery[0] = 0;
    const potteryL2Choice = legalDevelops(state).find((d) => d.industries[0] === 'pottery');
    expect(potteryL2Choice?.maxCount).toBe(1);
    expect(developCost(state, HUMAN, ['pottery', 'pottery'])).toBeNull();

    // Develop cotton twice in a single action (L1 has 3 tiles -> drops from 3 to 1; 2 iron consumed)
    const beforeCottonL1 = state.players[HUMAN].mat.cotton[0];
    const beforeIron = state.ironCubes;
    const cost = developCost(state, HUMAN, ['cotton', 'cotton']);
    expect(cost).not.toBeNull();

    applyHumanAction(state, { type: 'develop', cardIdx: 0, industries: ['cotton', 'cotton'] });
    expect(state.players[HUMAN].mat.cotton[0]).toBe(beforeCottonL1 - 2);
    expect(state.ironCubes).toBe(beforeIron - 2);
  });
});

describe('rail era double rail', () => {
  it('supports sequential 2-rail extension for £15 + 2 coal + 1 beer and flips exhausted brewery', () => {
    const state = newGame(42, 'easy');
    state.era = 'rail';
    state.actionsLeft = 2;
    state.players[HUMAN].money = 30;

    // Give human an anchor iron works in Birmingham and an own brewery in Farm North (disconnected!)
    state.board.birmingham[2] = {
      owner: HUMAN,
      industry: 'iron',
      level: 1,
      flipped: false,
      resources: 4,
    };
    state.board.farmNorth[0] = {
      owner: HUMAN,
      industry: 'brewery',
      level: 2,
      flipped: false,
      resources: 1, // Last barrel -> should flip!
    };

    // Put coal mines on Coventry and Nuneaton so coal is free from the network, or let Birmingham->Coventry->Nuneaton buy from market via Oxford
    // Birmingham-Oxford reaches Oxford merchant -> market coal available once linked to Oxford, or put coal mine in Birmingham/Coventry!
    state.board.coventry[1] = {
      owner: AUTOMA,
      industry: 'coal',
      level: 2,
      flipped: false,
      resources: 2,
    };

    // Notice 'coventry-nuneaton' is NOT adjacent to human network yet — it only becomes adjacent AFTER 'birmingham-coventry' is placed!
    const opt = networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'coventry-nuneaton');
    expect(opt).not.toBeNull();
    expect(opt!.linkIds).toEqual(['birmingham-coventry', 'coventry-nuneaton']);
    expect(opt!.moneyCost).toBe(15);
    expect(opt!.totalCost).toBe(15); // Both coal cubes came free from Coventry's coal mine!
    expect(opt!.beerSource).toEqual({ kind: 'own', city: 'farmNorth', slot: 0 });

    // Also surfaced in legalNetworks
    const allLegal = legalNetworks(state);
    expect(
      allLegal.some(
        (c) =>
          c.option.linkIds.length === 2 &&
          c.option.linkIds.includes('birmingham-coventry') &&
          c.option.linkIds.includes('coventry-nuneaton'),
      ),
    ).toBe(true);

    applyHumanAction(state, { type: 'network', cardIdx: 0, option: opt! });
    expect(state.links['birmingham-coventry']).toBe(HUMAN);
    expect(state.links['coventry-nuneaton']).toBe(HUMAN);
    expect(state.players[HUMAN].money).toBe(15);
    expect(state.board.coventry[1]!.resources).toBe(0);
    expect(state.board.coventry[1]!.flipped).toBe(true);
    expect(state.board.farmNorth[0]!.resources).toBe(0);
    expect(state.board.farmNorth[0]!.flipped).toBe(true);
  });

  it('rejects double rail in canal era, when cash < £15, or when only merchant beer exists', () => {
    const state = newGame(42, 'easy');
    state.era = 'rail';
    state.actionsLeft = 2;
    state.players[HUMAN].money = 30;
    state.board.birmingham[2] = { owner: HUMAN, industry: 'iron', level: 1, flipped: false, resources: 4 };
    state.board.coventry[1] = { owner: HUMAN, industry: 'coal', level: 2, flipped: false, resources: 3 };

    // Ensure Oxford merchant has beer, and connect Birmingham to Oxford on link 2 — merchant beer MUST NOT satisfy double rail!
    for (const m of state.merchants) {
      m.beer = m.tiles.map(() => true);
    }
    expect(networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'birmingham-oxford')).toBeNull();

    // Add an opponent brewery in Walsall (not connected to coventry or oxford) -> still null
    state.board.walsall[1] = { owner: AUTOMA, industry: 'brewery', level: 2, flipped: false, resources: 2 };
    expect(networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'birmingham-oxford')).toBeNull();

    // Connect opponent brewery to link 2 endpoint ('birmingham-walsall' as link 2!) -> now legal!
    expect(networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'birmingham-walsall')).not.toBeNull();

    // Cash < £15 -> null
    state.players[HUMAN].money = 14;
    expect(networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'birmingham-walsall')).toBeNull();

    // Canal era -> null even with £30
    state.players[HUMAN].money = 30;
    state.era = 'canal';
    expect(networkDoubleOption(state, HUMAN, 'birmingham-coventry', 'birmingham-walsall')).toBeNull();
  });
});

describe('official 2-player mode & seeded deck', () => {
  it('uses the exact 40-card official 2p deck (38 in Canal with 2 unseen discards, 40 in Rail) deterministically from seed', () => {
    const a = newGame(202611, 'easy', { mode: '2p', playerNames: ['Gemini', 'Brass PPO'] });
    const b = newGame(202611, 'easy', { mode: '2p', playerNames: ['Gemini', 'Brass PPO'] });
    const c = newGame(999999, 'easy', { mode: '2p' });

    expect(a.turnOrder).toEqual(b.turnOrder);
    expect(a.players[0].hand).toEqual(b.players[0].hand);
    expect(a.players[1].hand).toEqual(b.players[1].hand);
    expect(a.drawPile).toEqual(b.drawPile);
    expect(a.merchants).toEqual(b.merchants);
    expect(a.drawPile).not.toEqual(c.drawPile);

    // Canal setup: 8 + 8 + 1 + 1 + 22 = 40 official 2p cards
    expect(a.players[0].hand).toHaveLength(8);
    expect(a.players[1].hand).toHaveLength(8);
    expect(a.players[0].discard).toHaveLength(1);
    expect(a.players[1].discard).toHaveLength(1);
    expect(a.drawPile).toHaveLength(22);

    const allForty = [
      ...a.players[0].hand,
      ...a.players[1].hand,
      ...a.players[0].discard,
      ...a.players[1].discard,
      ...a.drawPile,
    ];
    expect(allForty).toHaveLength(40);
    // Exclude 3p/4p northern cities and cotton/goods industry cards
    for (const card of allForty) {
      if (card.kind === 'location') {
        expect(['belper', 'derby', 'leek', 'stoke', 'stone', 'uttoxeter']).not.toContain(card.city);
      } else if (card.kind === 'industry') {
        expect(card.industries).not.toContain('cotton');
        expect(card.industries).not.toContain('goods');
      }
    }
  });

  it('reorders turn order by lowest round spending and plays a full 78-action 20-round match cleanly', () => {
    const state = newGame(12345, 'easy', { mode: '2p', firstPlayer: 0 });
    expect(state.activePlayer).toBe(0);
    expect(state.actionsLeft).toBe(1);

    // Round 1: Player 0 builds (spends > £0), Player 1 takes a loan (spends £0) -> Player 1 must go first in Round 2!
    const p0Builds = legalBuilds(state, 0);
    expect(p0Builds.length).toBeGreaterThan(0);
    applyHumanAction(state, { type: 'build', cardIdx: p0Builds[0].cardIdx, option: p0Builds[0].option });
    expect(state.activePlayer).toBe(1);
    expect(state.turn).toBe(1);
    expect(state.actionsLeft).toBe(1);

    applyHumanAction(state, { type: 'loan', cardIdx: 0 });
    expect(state.turn).toBe(2);
    expect(state.turnOrder).toEqual([1, 0]);
    expect(state.activePlayer).toBe(1);
    expect(state.actionsLeft).toBe(2);

    // Play remaining 76 actions (total 78 actions = 38 Canal + 40 Rail) to completion
    let totalActions = 2;
    let sawRailSetup = false;
    while (!state.gameOver && totalActions < 100) {
      if (state.era === 'rail' && state.turn === 1 && state.turnIndex === 0 && state.actionsLeft === 2 && !sawRailSetup) {
        sawRailSetup = true;
        // Rail setup uses all 40 official 2p cards: 8 + 8 + 24 = 40
        expect(state.players[0].hand).toHaveLength(8);
        expect(state.players[1].hand).toHaveLength(8);
        expect(state.drawPile).toHaveLength(24);
      }
      const pid = state.activePlayer;
      const builds = legalBuilds(state, pid);
      const nets = legalNetworks(state, pid);
      if (builds.length > 0) {
        applyHumanAction(state, { type: 'build', cardIdx: builds[0].cardIdx, option: builds[0].option });
      } else if (nets.length > 0) {
        applyHumanAction(state, { type: 'network', cardIdx: 0, option: nets[0].option });
      } else {
        applyHumanAction(state, { type: 'pass', cardIdx: 0 });
      }
      totalActions += 1;
    }

    expect(sawRailSetup).toBe(true);
    expect(state.gameOver).toBe(true);
    expect(totalActions).toBe(78);
  });
});
