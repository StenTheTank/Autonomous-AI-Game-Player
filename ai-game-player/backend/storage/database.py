import json
import os
import sqlite3
import time
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_game_player.db")


class SessionDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    game_name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_reason TEXT DEFAULT '',
                    config_json TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    strategy TEXT DEFAULT '',
                    lessons_json TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS session_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    observation TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    expected_result TEXT NOT NULL,
                    evaluation TEXT DEFAULT '',
                    visual_delta_pct REAL DEFAULT 0.0,
                    goal_status TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS saved_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rl_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    episode_num INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    your_vp INTEGER NOT NULL,
                    automa_vp INTEGER NOT NULL,
                    won INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    policy_loss REAL DEFAULT 0.0,
                    value_loss REAL DEFAULT 0.0,
                    entropy REAL DEFAULT 0.0,
                    checkpoint_tag TEXT DEFAULT ''
                );
                """
            )
            conn.commit()

    @staticmethod
    def _strip_sensitive_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(config)
        cleaned.pop("api_key", None)
        cleaned.pop("openai_api_key", None)
        cleaned.pop("anthropic_api_key", None)
        return cleaned

    def create_session(self, session_id: str, config: Dict[str, Any], stats: Dict[str, Any]) -> None:
        now = time.time()
        safe_config = self._strip_sensitive_fields(config)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, created_at, updated_at, game_name, url, provider, model,
                    status, status_reason, config_json, stats_json, strategy, lessons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    now,
                    safe_config.get("game_name", "Browser Game"),
                    safe_config.get("url", ""),
                    safe_config.get("provider", "openai"),
                    safe_config.get("model", ""),
                    "starting",
                    "",
                    json.dumps(safe_config),
                    json.dumps(stats),
                    "",
                    "[]",
                ),
            )
            conn.commit()

    def update_session(
        self,
        session_id: str,
        status: str,
        stats: Dict[str, Any],
        status_reason: str = "",
        strategy: str = "",
        lessons: Optional[List[str]] = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, status = ?, status_reason = ?, stats_json = ?,
                    strategy = ?, lessons_json = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    status,
                    status_reason,
                    json.dumps(stats),
                    strategy,
                    json.dumps(lessons or []),
                    session_id,
                ),
            )
            conn.commit()

    def record_step(
        self,
        session_id: str,
        iteration: int,
        observation: str,
        reasoning: str,
        action: Dict[str, Any],
        expected_result: str,
        evaluation: str = "",
        visual_delta_pct: float = 0.0,
        goal_status: str = "in_progress",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_steps (
                    session_id, iteration, timestamp, observation, reasoning,
                    action_json, expected_result, evaluation, visual_delta_pct,
                    goal_status, tokens_used, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    iteration,
                    time.time(),
                    observation,
                    reasoning,
                    json.dumps(action),
                    expected_result,
                    evaluation,
                    visual_delta_pct,
                    goal_status,
                    tokens_used,
                    cost_usd,
                ),
            )
            conn.commit()

    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for r in rows:
                results.append(
                    {
                        "session_id": r["session_id"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "game_name": r["game_name"],
                        "url": r["url"],
                        "provider": r["provider"],
                        "model": r["model"],
                        "status": r["status"],
                        "status_reason": r["status_reason"],
                        "config": json.loads(r["config_json"]),
                        "stats": json.loads(r["stats_json"]),
                        "strategy": r["strategy"],
                        "lessons": json.loads(r["lessons_json"]),
                    }
                )
            return results

    def get_session_detail(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            step_rows = conn.execute(
                "SELECT * FROM session_steps WHERE session_id = ? ORDER BY iteration ASC",
                (session_id,),
            ).fetchall()
            steps = [
                {
                    "iteration": s["iteration"],
                    "timestamp": s["timestamp"],
                    "observation": s["observation"],
                    "reasoning": s["reasoning"],
                    "action": json.loads(s["action_json"]),
                    "expected_result": s["expected_result"],
                    "evaluation": s["evaluation"],
                    "visual_delta_pct": s["visual_delta_pct"],
                    "goal_status": s["goal_status"],
                    "tokens_used": s["tokens_used"],
                    "cost_usd": s["cost_usd"],
                }
                for s in step_rows
            ]
            return {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "game_name": row["game_name"],
                "url": row["url"],
                "provider": row["provider"],
                "model": row["model"],
                "status": row["status"],
                "status_reason": row["status_reason"],
                "config": json.loads(row["config_json"]),
                "stats": json.loads(row["stats_json"]),
                "strategy": row["strategy"],
                "lessons": json.loads(row["lessons_json"]),
                "steps": steps,
            }

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM session_steps WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0

    def save_profile(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        safe_config = self._strip_sensitive_fields(config)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO saved_profiles (name, created_at, config_json)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET created_at = excluded.created_at, config_json = excluded.config_json
                """,
                (name.strip(), now, json.dumps(safe_config)),
            )
            conn.commit()
        return {"name": name.strip(), "created_at": now, "config": safe_config}

    def list_profiles(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM saved_profiles ORDER BY created_at DESC"
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "config": json.loads(r["config_json"]),
                }
                for r in rows
            ]

    def delete_profile(self, profile_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM saved_profiles WHERE id = ?", (profile_id,))
            conn.commit()
            return cur.rowcount > 0

    def record_rl_episode(
        self,
        run_id: str,
        episode_num: int,
        your_vp: int,
        automa_vp: int,
        won: bool,
        total_reward: float,
        steps: int,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
        checkpoint_tag: str = "",
    ) -> Dict[str, Any]:
        ts = time.time()
        won_int = 1 if won else 0
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO rl_episodes (
                    run_id, episode_num, timestamp, your_vp, automa_vp, won,
                    total_reward, steps, policy_loss, value_loss, entropy, checkpoint_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(episode_num),
                    ts,
                    int(your_vp),
                    int(automa_vp),
                    won_int,
                    round(float(total_reward), 4),
                    int(steps),
                    round(float(policy_loss), 5),
                    round(float(value_loss), 5),
                    round(float(entropy), 5),
                    checkpoint_tag,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
        return {
            "id": row_id,
            "run_id": run_id,
            "episode_num": int(episode_num),
            "timestamp": ts,
            "your_vp": int(your_vp),
            "automa_vp": int(automa_vp),
            "won": bool(won),
            "total_reward": round(float(total_reward), 4),
            "steps": int(steps),
            "policy_loss": round(float(policy_loss), 5),
            "value_loss": round(float(value_loss), 5),
            "entropy": round(float(entropy), 5),
            "checkpoint_tag": checkpoint_tag,
        }

    def get_rl_training_summary(self, limit: int = 200) -> Dict[str, Any]:
        with self._connect() as conn:
            agg = conn.execute(
                "SELECT COUNT(*) AS cnt, MAX(your_vp) AS max_vp FROM rl_episodes"
            ).fetchone()
            total_count = int(agg["cnt"] or 0) if agg else 0
            best_vp = int(agg["max_vp"] or 0) if agg else 0

            rows = conn.execute(
                "SELECT * FROM rl_episodes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        episodes = [
            {
                "id": r["id"],
                "run_id": r["run_id"],
                "episode_num": r["episode_num"],
                "timestamp": r["timestamp"],
                "your_vp": r["your_vp"],
                "automa_vp": r["automa_vp"],
                "won": bool(r["won"]),
                "total_reward": r["total_reward"],
                "steps": r["steps"],
                "policy_loss": r["policy_loss"],
                "value_loss": r["value_loss"],
                "entropy": r["entropy"],
                "checkpoint_tag": r["checkpoint_tag"],
            }
            for r in reversed(rows)
        ]

        window = episodes[-20:] if episodes else []
        rolling_avg_vp = round(sum(e["your_vp"] for e in window) / len(window), 1) if window else 0.0
        rolling_automa_vp = round(sum(e["automa_vp"] for e in window) / len(window), 1) if window else 0.0
        rolling_win_rate = round(100.0 * sum(1 for e in window if e["won"]) / len(window), 1) if window else 0.0

        return {
            "total_episodes_recorded": total_count,
            "best_vp": best_vp,
            "rolling_avg_vp": rolling_avg_vp,
            "rolling_automa_vp": rolling_automa_vp,
            "rolling_win_rate_pct": rolling_win_rate,
            "episodes": episodes,
        }

    def clear_rl_episodes(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM rl_episodes")
            conn.commit()
            return int(cur.rowcount or 0)

