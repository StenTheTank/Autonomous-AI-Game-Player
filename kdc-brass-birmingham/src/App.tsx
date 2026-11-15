import { useState } from 'react';
import './ui/theme.css';
import { AUTOMA, HUMAN, activePlayerId, playerLabel, type GameMode, type GameState, type NewGameOptions } from './engine/state';
import { compareTwoPlayerWinner } from './engine/game';
import { levelForSpace } from './engine/income';
import { COAL_MARKET, IRON_MARKET, nextBuyPrice } from './engine/market';
import { CITIES } from './engine/data/board';
import type { MautomaDifficulty } from './engine/mautoma/cards';
import { BoardMap } from './ui/BoardMap';
import { cardLabel, useActionFlow } from './ui/ActionPanel';
import { useGame, useTheme } from './ui/useGame';
import { HelpButton } from './ui/Help';

export default function App() {
  const game = useGame();
  const { theme, toggle } = useTheme();
  const actor = game.state ? activePlayerId(game.state) : HUMAN;

  return (
    <div className="app">
      <div className="topbar">
        <h1>{game.state?.mode === '2p' ? 'Brass Birmingham · 2P Arena' : 'Brass Birmingham Solo'}</h1>
        {game.state && (
          <span className="stat" data-testid="era-turn">
            {game.state.era === 'canal' ? 'Canal' : 'Rail'} Era — Round {game.state.turn}/10 —{' '}
            {game.state.mode === '2p' ? `${playerLabel(game.state, actor)} · ` : ''}
            {game.state.actionsLeft} action{game.state.actionsLeft === 1 ? '' : 's'} left · Seed #{game.state.seed}
          </span>
        )}
        <div className="spacer" />
        {game.state && (
          <>
            <button onClick={game.undo} disabled={!game.canUndo} data-testid="undo">
              ⎌ Undo
            </button>
            <button onClick={game.reset} data-testid="new-game">
              New game
            </button>
          </>
        )}
        <HelpButton />
        <button onClick={toggle} data-testid="theme-toggle" aria-label="Toggle dark mode">
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </div>
      {game.state ? <GameScreen state={game.state} dispatch={game.dispatch} /> : <SetupScreen onStart={game.start} />}
      <footer className="footer" data-testid="footer">
        <small>State auto-saves to this browser. Use Undo to revert an applied action.</small>
        <small className="disclaimer">
          Unofficial fan project for solo play. Not affiliated with or endorsed by Roxley Games or the designers of
          Brass: Birmingham. No game art or rules text is reproduced. The Mautoma solo variant is by Mauro Gibertoni
          (mautoma.com).
        </small>
        <small className="footer-meta">
          v{__APP_VERSION__} ·{' '}
          <a href="https://github.com/KDC-Solo/brass-birmingham" target="_blank" rel="noreferrer">
            Source &amp; issues on GitHub
          </a>
        </small>
      </footer>
    </div>
  );
}

function SetupScreen({
  onStart,
}: {
  onStart: (seed: number, difficulty: MautomaDifficulty, opts?: NewGameOptions) => void;
}) {
  const [mode, setMode] = useState<GameMode>('solo');
  const [difficulty, setDifficulty] = useState<MautomaDifficulty>('easy');
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1_000_000));
  const [p1Name, setP1Name] = useState('Player 1');
  const [p2Name, setP2Name] = useState('Player 2');

  return (
    <div className="setup panel" data-testid="setup">
      <h2 style={{ margin: 0 }}>{mode === '2p' ? 'New 2-Player Official Match' : 'New solo game'}</h2>
      <p style={{ margin: 0, color: 'var(--muted)' }}>
        {mode === '2p'
          ? 'Official 2-player Brass: Birmingham ruleset (seeded 40-card 2p deck, turn-order spending, private active hands).'
          : 'You play the standard rules against the Mautoma, a card-driven opponent (fan solo variant by Mauro Gibertoni).'}
      </p>
      <label>
        Mode
        <select value={mode} onChange={(e) => setMode(e.target.value as GameMode)} data-testid="game-mode">
          <option value="solo">Solo vs Mautoma</option>
          <option value="2p">2-Player Official Rules (Head-to-Head / AI Duel)</option>
        </select>
      </label>
      {mode === 'solo' ? (
        <label>
          Difficulty
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value as MautomaDifficulty)} data-testid="difficulty">
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </label>
      ) : (
        <>
          <label>
            Player 1 (Blue)
            <input type="text" value={p1Name} onChange={(e) => setP1Name(e.target.value)} data-testid="p1-name" />
          </label>
          <label>
            Player 2 (Amber)
            <input type="text" value={p2Name} onChange={(e) => setP2Name(e.target.value)} data-testid="p2-name" />
          </label>
        </>
      )}
      <label>
        Seed
        <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} data-testid="seed" />
      </label>
      <button
        className="primary"
        onClick={() =>
          onStart(seed, difficulty, {
            mode,
            playerNames: mode === '2p' ? [p1Name || 'Player 1', p2Name || 'Player 2'] : undefined,
          })
        }
        data-testid="start-game"
      >
        Start game
      </button>
    </div>
  );
}

function GameScreen({ state, dispatch }: { state: GameState; dispatch: (a: never) => string | null }) {
  const flow = useActionFlow(state, dispatch as never);
  const actor = activePlayerId(state);
  const activePlayerState = state.players[actor];
  const human = state.players[HUMAN];
  const automa = state.players[AUTOMA];
  const isTwoPlayer = state.mode === '2p';
  const hideSeatParam = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('hideSeat') : null;
  const isHandMasked = isTwoPlayer && (hideSeatParam === String(actor));

  return (
    <>
      <div className="game-layout">
        <div className="board-wrap">
          <BoardMap
            state={state}
            highlightCities={flow.highlightCities}
            highlightLinks={flow.highlightLinks}
            onCityClick={(city) => {
              const choice = flow.builds.find((b) => b.option.city === city);
              if (choice) flow.chooseBuild(choice);
            }}
            onLinkClick={(linkId) => {
              const choice = flow.networks.find((n) => n.option.linkIds.length === 1 && n.option.linkIds[0] === linkId);
              if (choice) flow.chooseNetwork(choice);
            }}
          />
        </div>

        <div className="side">
          <div className="panel" data-testid="player-panel" data-active-player={actor}>
            {isTwoPlayer ? (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {([0, 1] as const).map((pid) => {
                    const p = state.players[pid];
                    const isTurn = actor === pid && !state.gameOver;
                    const orderRank = state.turnOrder[0] === pid ? '1st' : '2nd';
                    return (
                      <div
                        key={pid}
                        data-testid={`p${pid}-summary`}
                        style={{
                          padding: '6px 8px',
                          borderRadius: 6,
                          border: isTurn ? '1.5px solid var(--highlight)' : '1px solid var(--border)',
                          background: isTurn ? 'var(--panel-2)' : 'transparent',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                          <span>
                            <span className={pid === 0 ? 'badge-human' : 'badge-automa'}>
                              {playerLabel(state, pid)}
                            </span>{' '}
                            <small style={{ color: 'var(--muted)' }}>({orderRank})</small>
                            {isTurn && (
                              <strong style={{ marginLeft: 6, color: 'var(--accent)', fontSize: 11 }} data-testid="active-turn-badge">
                                ▶ TURN
                              </strong>
                            )}
                          </span>
                          <b>{p.vp} VP</b>
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
                          £{p.money} (spent £{p.spent}) · inc {levelForSpace(p.incomeSpace) >= 0 ? '+' : ''}
                          {levelForSpace(p.incomeSpace)} · hand {p.hand.length}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="stat-row" style={{ marginTop: 6 }}>
                  <span className="stat">
                    Draw pile <b data-testid="draw-pile">{state.drawPile.length}</b>
                  </span>
                  <span className="stat">
                    Turn order <b>{playerLabel(state, state.turnOrder[0])} → {playerLabel(state, state.turnOrder[1])}</b>
                  </span>
                </div>
              </>
            ) : (
              <>
                <h3>
                  <span className="badge-human">You</span> · £{human.money} · income {levelForSpace(human.incomeSpace)} · {human.vp} VP
                </h3>
                <div className="stat-row">
                  <span className="stat">
                    Draw pile <b data-testid="draw-pile">{state.drawPile.length}</b>
                  </span>
                  <span className="stat">
                    <span className="badge-automa">Automa</span> <b data-testid="automa-vp">{automa.vp} VP</b>
                  </span>
                  <span className="stat">
                    Automa deck <b>{state.automaDeck.length}</b>
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="panel">
            <h3>Markets</h3>
            <div className="stat-row">
              <span className="stat" data-testid="coal-market">
                ⚫ Coal £<b>{nextBuyPrice(COAL_MARKET, state.coalCubes)}</b> ({state.coalCubes})
              </span>
              <span className="stat" data-testid="iron-market">
                🔩 Iron £<b>{nextBuyPrice(IRON_MARKET, state.ironCubes)}</b> ({state.ironCubes})
              </span>
            </div>
          </div>

          <div className="panel" data-testid="action-panel">
            <h3>Actions</h3>
            <div className="actions">
              {(['build', 'network', 'sell', 'develop', 'loan', 'scout', 'pass'] as const).map((a) => (
                <button
                  key={a}
                  data-testid={`action-${a}`}
                  disabled={!flow.availability[a] || state.gameOver}
                  className={flow.flow.action === a ? 'selected' : ''}
                  onClick={() => flow.chooseAction(a)}
                >
                  {a[0].toUpperCase() + a.slice(1)}
                </button>
              ))}
            </div>

            {flow.flow.action && (
              <p style={{ margin: '8px 0 4px', fontSize: 13, color: 'var(--muted)' }} data-testid="flow-hint">
                {flow.flow.cardIdx === null
                  ? flow.flow.action === 'scout'
                    ? 'Pick the action card, then 2 more to discard.'
                    : `Pick a card to ${flow.flow.action}.`
                  : flowHint(flow.flow.action)}
                <button style={{ marginLeft: 8, fontSize: 12, padding: '2px 8px' }} onClick={flow.reset} data-testid="flow-cancel">
                  cancel
                </button>
              </p>
            )}
            {flow.error && (
              <p style={{ color: 'var(--danger)', fontSize: 13 }} data-testid="flow-error">
                {flow.error}
              </p>
            )}

            <h3 style={{ marginTop: 10 }}>
              {isTwoPlayer
                ? isHandMasked
                  ? `${playerLabel(state, actor)}'s Hand (${activePlayerState.hand.length} hidden)`
                  : `${playerLabel(state, actor)}'s Hand`
                : 'Hand'}
            </h3>
            <div className="hand" data-testid="hand">
              {activePlayerState.hand.map((card, i) => (
                <span
                  key={i}
                  className={[
                    'card-chip',
                    !isHandMasked && card.kind.startsWith('wild') ? 'wild' : '',
                    flow.flow.cardIdx === i || flow.flow.scoutExtras.includes(i) ? 'selected' : '',
                  ].join(' ')}
                  style={{ opacity: flow.flow.action && !flow.cardsSelectable.has(i) ? 0.4 : 1 }}
                  data-testid={`hand-card-${i}`}
                  onClick={() => flow.cardsSelectable.has(i) && flow.chooseCard(i)}
                >
                  {isHandMasked ? `🂠 Hidden #${i + 1}` : cardLabel(card)}
                </span>
              ))}
            </div>

            {flow.flow.action === 'build' && flow.flow.cardIdx !== null && (
              <div className="option-list" style={{ marginTop: 8 }} data-testid="build-options">
                {flow.builds.map((b, i) => (
                  <button key={i} onClick={() => flow.chooseBuild(b)}>
                    {CITIES[b.option.city].name}: {b.option.industry} L{b.option.level} — £{b.option.totalCost}
                    {b.option.overbuild ? ' (overbuild)' : ''}
                  </button>
                ))}
                {flow.builds.length === 0 && <span style={{ color: 'var(--muted)' }}>No legal builds with this card.</span>}
              </div>
            )}

            {flow.flow.action === 'network' && flow.flow.cardIdx !== null && (
              <div className="option-list" style={{ marginTop: 8 }} data-testid="network-options">
                {flow.networks.map((n, i) => (
                  <button key={i} onClick={() => flow.chooseNetwork(n)}>
                    {n.option.linkIds.join(' + ')} — £{n.option.totalCost}
                  </button>
                ))}
              </div>
            )}

            {flow.flow.action === 'sell' && flow.flow.cardIdx !== null && (
              <div className="option-list" style={{ marginTop: 8 }} data-testid="sell-options">
                {flow.sells.map((s, i) => {
                  const selected = flow.flow.sales.some((x) => x.sale.city === s.sale.city && x.sale.slot === s.sale.slot);
                  const tile = state.board[s.sale.city][s.sale.slot]!;
                  return (
                    <button key={i} className={selected ? 'selected' : ''} onClick={() => flow.toggleSale(s)}>
                      {CITIES[s.sale.city].name}: {tile.industry} L{tile.level} (beer ×{s.sale.beerNeeded})
                    </button>
                  );
                })}
                <button className="primary" disabled={flow.flow.sales.length === 0} onClick={flow.confirmSell} data-testid="confirm-sell">
                  Sell {flow.flow.sales.length} building{flow.flow.sales.length === 1 ? '' : 's'}
                </button>
              </div>
            )}

            {flow.flow.action === 'develop' && flow.flow.cardIdx !== null && (
              <div className="option-list" style={{ marginTop: 8 }} data-testid="develop-options">
                {flow.develops.map((d, i) => {
                  const ind = d.industries[0];
                  const selCount = flow.flow.develops.filter((x) => x === ind).length;
                  return (
                    <button
                      key={i}
                      data-industry={ind}
                      data-max-count={d.maxCount}
                      className={selCount > 0 ? 'selected' : ''}
                      onClick={() => flow.toggleDevelop(ind)}
                    >
                      {ind}{selCount === 2 ? ' ×2' : ''}
                    </button>
                  );
                })}
                <button
                  className="primary"
                  disabled={flow.flow.develops.length === 0}
                  onClick={flow.confirmDevelop}
                  data-testid="confirm-develop"
                >
                  Develop {flow.flow.develops.length} tile{flow.flow.develops.length === 1 ? '' : 's'}
                </button>
              </div>
            )}
          </div>

          <div className="panel">
            <h3>Log</h3>
            <div className="log" data-testid="log">
              {[...state.log].reverse().map((line, i) => (
                <div key={state.log.length - i} className={i < 6 ? 'recent' : ''}>
                  {line}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {state.gameOver && (
        <div className="gameover" data-testid="gameover">
          <div className="panel">
            <h2>Game over</h2>
            {isTwoPlayer ? (
              <>
                <p>
                  <span className="badge-human">
                    {playerLabel(state, 0)}: {human.vp} VP (£{human.money}, inc {levelForSpace(human.incomeSpace)})
                  </span>{' '}
                  —{' '}
                  <span className="badge-automa">
                    {playerLabel(state, 1)}: {automa.vp} VP (£{automa.money}, inc {levelForSpace(automa.incomeSpace)})
                  </span>
                </p>
                <p style={{ fontSize: 22 }}>
                  {(() => {
                    const w = compareTwoPlayerWinner(state);
                    return w === 'tie' ? '🤝 Exact Tie!' : `🏆 ${playerLabel(state, w)} wins!`;
                  })()}
                </p>
              </>
            ) : (
              <>
                <p>
                  <span className="badge-human">You: {human.vp} VP</span> — <span className="badge-automa">Automa: {automa.vp} VP</span>
                </p>
                <p style={{ fontSize: 22 }}>{human.vp > automa.vp ? '🏆 You win!' : human.vp === automa.vp ? '🤝 Tie!' : '🤖 The Automa wins.'}</p>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function flowHint(action: string): string {
  switch (action) {
    case 'build':
      return 'Choose a highlighted city or an option below.';
    case 'network':
      return 'Choose a highlighted link or an option below.';
    case 'sell':
      return 'Toggle buildings to sell, then confirm.';
    case 'develop':
      return 'Pick 1–2 industries, then confirm.';
    default:
      return '';
  }
}
