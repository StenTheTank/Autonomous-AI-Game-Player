import type { CityId, IndustryType, PlayerId } from './types';
import { CITIES } from './data/board';
import { INDUSTRY_TRACKS, tileSpec } from './data/industries';
import { activePlayerId, type Card, type GameState } from './state';
import {
  beerSources, buildOption, canScout, cardAllowsBuild, developCost, networkDoubleOption, networkSingleOption,
  sellableBuildings, type BeerSource, type BuildOption, type NetworkOption, type SellableBuilding,
} from './actions';
import { canTakeLoan } from './income';
import { LINKS } from './data/board';

export interface BuildChoice {
  cardIdx: number;
  option: BuildOption;
}

/** All legal (card, build) combinations for `player` right now. */
export function legalBuilds(state: GameState, player: PlayerId = activePlayerId(state)): BuildChoice[] {
  const hand = state.players[player].hand;
  const result: BuildChoice[] = [];
  const cities = Object.keys(CITIES) as CityId[];
  const options = new Map<string, BuildOption | null>();
  for (const city of cities) {
    for (const industry of Object.keys(INDUSTRY_TRACKS) as IndustryType[]) {
      const key = `${city}:${industry}`;
      options.set(key, buildOption(state, player, city, industry));
    }
  }
  hand.forEach((card, cardIdx) => {
    for (const [key, option] of options) {
      if (!option) continue;
      const [city, industry] = key.split(':') as [CityId, IndustryType];
      if (cardAllowsBuild(state, player, card, city, industry)) {
        result.push({ cardIdx, option });
      }
    }
  });
  return result;
}

export interface NetworkChoice {
  cardIdx: number;
  option: NetworkOption;
}

export function legalNetworks(state: GameState, player: PlayerId = activePlayerId(state)): NetworkChoice[] {
  if (state.players[player].hand.length === 0) return [];
  const result: NetworkChoice[] = [];
  const singles: string[] = [];
  for (const link of LINKS) {
    const option = networkSingleOption(state, player, link.id);
    if (option) {
      result.push({ cardIdx: 0, option });
      singles.push(link.id);
    }
  }
  if (state.era === 'rail' && state.players[player].money >= 15 && singles.length > 0) {
    const bestDoubleByPair = new Map<string, NetworkOption>();
    const railLinks = LINKS.filter((l) => l.rail && state.links[l.id] == null);
    for (const linkId1 of singles) {
      for (const link2 of railLinks) {
        if (link2.id === linkId1) continue;
        const opt = networkDoubleOption(state, player, linkId1, link2.id);
        if (!opt) continue;
        const pairKey = [linkId1, link2.id].sort().join('+');
        const prev = bestDoubleByPair.get(pairKey);
        if (!prev || opt.totalCost < prev.totalCost) {
          bestDoubleByPair.set(pairKey, opt);
        }
      }
    }
    for (const option of bestDoubleByPair.values()) {
      result.push({ cardIdx: 0, option });
    }
  }
  return result;
}

export interface SellChoice {
  cardIdx: number;
  sale: SellableBuilding;
  beer: BeerSource[];
}

/** Beer plan for one sale: merchant beer first (bonus), then own, then opponent's. */
export function planBeerForSale(state: GameState, sale: SellableBuilding, player: PlayerId = activePlayerId(state)): BeerSource[] | null {
  const beer: BeerSource[] = [];
  const used = new Set<string>();
  const keyOf = (s: BeerSource) =>
    s.kind === 'merchant' ? `m:${s.merchantIdx}:${s.beerIdx}` : `${s.kind}:${s.city}:${s.slot}`;
  const usedFromTile = new Map<string, number>();
  for (let i = 0; i < sale.beerNeeded; i++) {
    const sources = beerSources(state, player, sale.city, sale.merchantIdx).filter((s) => {
      const key = keyOf(s);
      if (s.kind === 'merchant') return !used.has(key);
      const tile = state.board[s.city][s.slot]!;
      return (usedFromTile.get(key) ?? 0) < tile.resources;
    });
    if (sources.length === 0) return null;
    const hasMerchantAlready = beer.some((b) => b.kind === 'merchant');
    const pick =
      sources.find((s) => s.kind === 'merchant' && !hasMerchantAlready) ??
      sources.find((s) => s.kind === 'own') ??
      sources[0];
    if (pick.kind === 'merchant' && hasMerchantAlready) return null;
    beer.push(pick);
    const key = keyOf(pick);
    used.add(key);
    if (pick.kind !== 'merchant') usedFromTile.set(key, (usedFromTile.get(key) ?? 0) + 1);
  }
  return beer;
}

export function legalSells(state: GameState, player: PlayerId = activePlayerId(state)): SellChoice[] {
  if (state.players[player].hand.length === 0) return [];
  return sellableBuildings(state, player)
    .map((sale) => ({ cardIdx: 0, sale, beer: planBeerForSale(state, sale, player) }))
    .filter((c): c is SellChoice => c.beer !== null);
}

export interface DevelopChoice {
  cardIdx: number;
  industries: IndustryType[];
  maxCount: 1 | 2;
}

export function maxDevelopableForIndustry(state: GameState, industry: IndustryType, player: PlayerId = activePlayerId(state)): 0 | 1 | 2 {
  const track = state.players[player].mat[industry];
  const firstIdx = track.findIndex((c) => c > 0);
  if (firstIdx < 0) return 0;
  const firstLevel = firstIdx + 1;
  if (!tileSpec(industry, firstLevel).canDevelop) return 0;
  if (!developCost(state, player, [industry])) return 0;

  let secondLevel: number | null = null;
  if (track[firstIdx] >= 2) {
    secondLevel = firstLevel;
  } else {
    for (let j = firstIdx + 1; j < track.length; j++) {
      if (track[j] > 0) {
        secondLevel = j + 1;
        break;
      }
    }
  }
  if (
    secondLevel !== null &&
    tileSpec(industry, secondLevel).canDevelop &&
    developCost(state, player, [industry, industry]) !== null
  ) {
    return 2;
  }
  return 1;
}

export function legalDevelops(state: GameState, player: PlayerId = activePlayerId(state)): DevelopChoice[] {
  if (state.players[player].hand.length === 0) return [];
  const result: DevelopChoice[] = [];
  for (const industry of Object.keys(INDUSTRY_TRACKS) as IndustryType[]) {
    const maxCount = maxDevelopableForIndustry(state, industry, player);
    if (maxCount === 1 || maxCount === 2) {
      result.push({ cardIdx: 0, industries: [industry], maxCount });
    }
  }
  return result;
}

export function canLoan(state: GameState, player: PlayerId = activePlayerId(state)): boolean {
  return state.players[player].hand.length > 0 && canTakeLoan(state.players[player].incomeSpace);
}

export function scoutAllowed(state: GameState, player: PlayerId = activePlayerId(state)): boolean {
  return canScout(state, player);
}

export function handCards(state: GameState, player: PlayerId = activePlayerId(state)): Card[] {
  return state.players[player].hand;
}
