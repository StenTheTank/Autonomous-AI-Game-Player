import React from 'react';
import { Swords, Trophy, Sparkles, Coins, TrendingUp, Layers, Hash } from 'lucide-react';

function SeatScoreCard({ player, seatIndex, isWinner }) {
  if (!player) return null;
  const isBlue = seatIndex === 0;
  const isApi = player.role === 'api';
  const isHuman = player.role === 'human';
  const activeBorder = player.is_active
    ? isHuman
      ? 'border-emerald-400 shadow-[0_0_22px_rgba(52,211,153,0.28)] bg-emerald-950/35'
      : isBlue
      ? 'border-sky-400 shadow-[0_0_20px_rgba(56,189,248,0.22)] bg-sky-950/35'
      : 'border-amber-400 shadow-[0_0_20px_rgba(251,191,36,0.22)] bg-amber-950/35'
    : 'border-slate-800/90 bg-slate-950/75';

  return (
    <div
      className={`rounded-xl border p-3 transition-all flex flex-col justify-between gap-2 ${activeBorder}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${
                isBlue ? 'bg-blue-500' : 'bg-amber-500'
              }`}
            />
            <span className="text-xs font-bold text-white truncate">
              P{seatIndex + 1} · {player.label}
            </span>
            <span
              className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                isHuman
                  ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                  : isApi
                  ? 'bg-sky-500/15 border-sky-500/40 text-sky-300'
                  : 'bg-indigo-500/15 border-indigo-500/40 text-indigo-300'
              }`}
            >
              {isHuman ? 'YOU (HUMAN)' : isApi ? 'API MODEL' : 'LOCAL PPO'}
            </span>
          </div>
          <div className="text-[10px] font-mono text-slate-400 truncate mt-0.5">
            {isHuman ? 'Interactive Click-to-Play' : player.model_name}
          </div>
        </div>

        {isWinner ? (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-400/60 text-[10px] font-bold text-emerald-200 shrink-0">
            <Trophy size={11} className="text-emerald-300" />
            WINNER
          </span>
        ) : player.is_active ? (
          <span
            className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold shrink-0 border ${
              isHuman
                ? 'bg-emerald-500/25 border-emerald-400/60 text-emerald-100'
                : isBlue
                ? 'bg-sky-500/20 border-sky-400/50 text-sky-200'
                : 'bg-amber-500/20 border-amber-400/50 text-amber-200'
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full animate-ping ${
                isHuman ? 'bg-emerald-300' : isBlue ? 'bg-sky-300' : 'bg-amber-300'
              }`}
            />
            {isHuman ? 'YOUR TURN' : 'ACTIVE TURN'}
          </span>
        ) : null}
      </div>

      <div className="flex items-baseline justify-between gap-2 pt-0.5">
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-extrabold text-white tabular-nums">{player.vp}</span>
          <span className="text-xs font-semibold text-slate-400">VP</span>
          {player.projected_era_vp > 0 && (
            <span
              className="text-xs font-semibold text-emerald-400 tabular-nums"
              title={`Flipped Tiles: +${player.flipped_vp} VP · Links: +${player.link_vp} VP`}
            >
              +{player.projected_era_vp} on board
            </span>
          )}
        </div>
        {player.unflipped_risk_vp > 0 && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/25 text-amber-300 tabular-nums"
            title="Potential VP on unflipped tiles (0 VP until flipped!)"
          >
            {player.unflipped_risk_vp} unflipped
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-1.5 pt-1 border-t border-slate-800/80 text-[11px]">
        <div className="bg-slate-900/90 rounded-lg px-2 py-1 border border-slate-800/70">
          <div className="text-[9px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
            <Coins size={10} /> Cash (Spent)
          </div>
          <div className="font-bold text-slate-200 tabular-nums">
            £{player.money}{' '}
            <span className="text-slate-400 font-normal">(£{player.spent})</span>
          </div>
        </div>
        <div className="bg-slate-900/90 rounded-lg px-2 py-1 border border-slate-800/70">
          <div className="text-[9px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
            <TrendingUp size={10} /> Income
          </div>
          <div
            className={`font-bold tabular-nums ${
              player.income_pounds >= 0 ? 'text-emerald-300' : 'text-rose-300'
            }`}
          >
            {player.income_pounds >= 0 ? `+£${player.income_pounds}` : `-£${Math.abs(player.income_pounds)}`}
            <span className="text-slate-500 font-normal"> /rd</span>
          </div>
        </div>
        <div className="bg-slate-900/90 rounded-lg px-2 py-1 border border-slate-800/70">
          <div className="text-[9px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
            <Layers size={10} /> Hand
          </div>
          <div className="font-bold text-slate-200 tabular-nums">{player.hand_count} cards</div>
        </div>
      </div>
    </div>
  );
}

export default function DuelScoreboardBanner({ duelState }) {
  if (!duelState || !duelState.active) return null;

  const players = duelState.players || [];
  const p0 = players[0];
  const p1 = players[1];
  const eraLabel = duelState.era === 'rail' ? 'RAIL ERA' : 'CANAL ERA';
  const order = duelState.turn_order || [0, 1];
  const orderText = order
    .map((s, idx) => `${idx === 0 ? '1st' : '2nd'}: P${s + 1}`)
    .join(' → ');

  return (
    <section className="bg-gradient-to-r from-slate-900/95 via-slate-900/90 to-slate-900/95 border border-indigo-500/35 rounded-2xl p-3 shadow-xl">
      <div className="grid grid-cols-1 lg:grid-cols-11 gap-3 items-stretch">
        <div className="lg:col-span-4">
          <SeatScoreCard
            player={p0}
            seatIndex={0}
            isWinner={duelState.game_over && duelState.winner_seat === 0}
          />
        </div>

        {/* Center Match Telemetry */}
        <div className="lg:col-span-3 rounded-xl bg-slate-950/85 border border-slate-800/90 px-3 py-2.5 flex flex-col items-center justify-between text-center gap-1.5">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-indigo-300">
            <Swords size={13} className="text-indigo-400" />
            <span>
              {duelState.match_mode === 'brass_human_vs_ai'
                ? 'Official 2P · You vs AI'
                : 'Official 2P Brass Duel'}
            </span>
          </div>

          {duelState.game_over ? (
            <div className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-400/45 text-xs font-bold text-emerald-200 flex items-center gap-1.5">
              <Sparkles size={14} className="text-emerald-300 shrink-0" />
              <span>{duelState.winner_summary || 'Match Complete!'}</span>
            </div>
          ) : (
            <div>
              <div className="text-sm font-extrabold text-white tracking-tight">
                {eraLabel} · ROUND {duelState.turn} / 10
              </div>
              <div className="text-[11px] text-slate-300 mt-0.5">
                Actions remaining this turn:{' '}
                <span className="font-bold text-sky-300">{duelState.actions_left}</span>
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-center gap-1.5 text-[10px] text-slate-300">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-indigo-500/15 border border-indigo-500/35 text-indigo-200 font-mono">
              <Hash size={10} />
              Seed {duelState.seed}
            </span>
            <span className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 font-mono text-slate-300">
              Deck: {duelState.draw_pile}
            </span>
            <span
              className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300"
              title="Next round turn order goes to whoever spends less £ this round"
            >
              {orderText}
            </span>
          </div>
        </div>

        <div className="lg:col-span-4">
          <SeatScoreCard
            player={p1}
            seatIndex={1}
            isWinner={duelState.game_over && duelState.winner_seat === 1}
          />
        </div>
      </div>
    </section>
  );
}
