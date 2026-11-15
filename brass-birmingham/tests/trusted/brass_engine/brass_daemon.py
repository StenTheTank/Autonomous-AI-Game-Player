"""Persistent local Brass: Birmingham engine daemon (static webapp + headless Chromium + JSON RPC)."""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import socket
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright

from brass_engine.brass_bridge import (
    describe_macro_action,
    enumerate_legal_actions,
    execute_macro_action,
    extract_brass_state,
    inject_fast_training_css,
    start_new_brass_game,
)
from brass_engine.brass_features import compute_board_summary
from brass_engine.formatter import format_legal_actions, format_status_hud
from brass_engine.scoring import (
    DEFAULT_EVAL_DIFFICULTY,
    DEFAULT_EVAL_SEED,
    action_signature,
    compute_brass_harbor_reward,
    compute_step_chain_hash,
    parse_record_signature,
    serialize_action_record,
)


class QuietStaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def start_local_static_server(webapp_dir: Path, port: int = 0) -> Tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(QuietStaticHandler, directory=str(webapp_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{actual_port}/index.html"


async def replay_records_on_page(
    page: Any,
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, List[str]]:
    """
    Deterministically replay recorded trajectory signatures on a freshly initialized page.
    Stops immediately if any recorded signature is not genuinely legal at that step.
    """
    warnings: List[str] = []
    valid_steps = 0
    state = await extract_brass_state(page)

    for idx, rec in enumerate(records):
        if state.get("game_over"):
            warnings.append(f"Record #{idx} ignored because game was already over at step {valid_steps}.")
            break
        legal = await enumerate_legal_actions(page, state)
        target_sig = parse_record_signature(rec)
        matched: Optional[Dict[str, Any]] = None
        for cand in legal:
            if action_signature(cand) == target_sig:
                matched = cand
                break
        if matched is None:
            warnings.append(f"Illegal or mismatched action signature at record #{idx}: {target_sig}")
            break
        exec_res = await execute_macro_action(page, matched)
        if not exec_res.get("ok"):
            warnings.append(f"Engine rejected execution at record #{idx}: {describe_macro_action(matched)}")
            break
        valid_steps += 1
        state = await extract_brass_state(page)

    return state, valid_steps, warnings


class BrassEngineSession:
    def __init__(
        self,
        webapp_dir: Path,
        trajectory_path: Path,
        seed: int = DEFAULT_EVAL_SEED,
        difficulty: str = DEFAULT_EVAL_DIFFICULTY,
        ledger_path: Optional[Path] = None,
    ) -> None:
        self.webapp_dir = webapp_dir
        self.trajectory_path = trajectory_path
        self.seed = int(seed)
        self.difficulty = str(difficulty)
        default_ledger = (
            Path("/var/tmp/.brass_ledger.jsonl")
            if os.name != "nt"
            else (Path(tempfile.gettempdir()) / f".brass_ledger_{self.seed}.jsonl")
        )
        self.ledger_path = ledger_path or Path(os.environ.get("BRASS_LEDGER_PATH", str(default_ledger)))

        self._lock = asyncio.Lock()
        self._static_server: Optional[http.server.ThreadingHTTPServer] = None
        self._webapp_url: str = ""
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

        self.valid_steps: int = 0
        self.chain_hash: str = "GENESIS"
        self.state: Dict[str, Any] = {}
        self.cached_step: int = -1
        self.cached_legal: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        return out

    async def start(self) -> None:
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._static_server, self._webapp_url = start_local_static_server(self.webapp_dir, port=0)

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        self._context = await self._browser.new_context(viewport={"width": 1280, "height": 800})
        self._page = await self._context.new_page()
        await self._page.goto(self._webapp_url, wait_until="domcontentloaded")
        await inject_fast_training_css(self._page)
        await start_new_brass_game(self._page, seed=self.seed, difficulty=self.difficulty)

        traj_records = self._read_jsonl_records(self.trajectory_path)
        ledger_records = self._read_jsonl_records(self.ledger_path)

        # Anti-rewind protection: if `/app/trajectory.jsonl` was deleted/truncated to attempt a mulligan,
        # restore the authentic committed history from the daemon ledger!
        existing_records = ledger_records if len(ledger_records) > len(traj_records) else traj_records

        if existing_records:
            self.state, self.valid_steps, _ = await replay_records_on_page(self._page, existing_records)
            self.history = existing_records[: self.valid_steps]
            self.chain_hash = "GENESIS"
            for idx, rec in enumerate(self.history):
                sig_list = rec.get("signature") or []
                self.chain_hash = compute_step_chain_hash(self.chain_hash, idx, sig_list, self.seed)
                rec["chain_hash"] = self.chain_hash
            # Re-sync trajectory file if it was tampered or truncated
            if len(traj_records) != len(self.history):
                with self.trajectory_path.open("w", encoding="utf-8", newline="\n") as f:
                    for rec in self.history:
                        f.write(json.dumps(rec) + "\n")
        else:
            self.state = await extract_brass_state(self._page)
            self.valid_steps = 0
            self.chain_hash = "GENESIS"
            self.history = []

        await self._refresh_legal_unlocked()

    async def _refresh_legal_unlocked(self) -> List[Dict[str, Any]]:
        if self.cached_step != self.valid_steps:
            self.cached_legal = await enumerate_legal_actions(self._page, self.state)
            self.cached_step = self.valid_steps
        return self.cached_legal

    async def get_status_payload(self) -> Dict[str, Any]:
        async with self._lock:
            self.state = await extract_brass_state(self._page)
            summary = compute_board_summary(self.state)
            score = compute_brass_harbor_reward(self.state, self.valid_steps, records=self.history)
            hud = format_status_hud(self.state, self.valid_steps, records=self.history)
            return {
                "ok": True,
                "step": self.valid_steps,
                "difficulty": self.difficulty,
                "state": self.state,
                "board_summary": {k: v for k, v in summary.items() if k != "city_link_dots"},
                "harbor_score_preview": score,
                "hud": hud,
            }

    async def get_legal_payload(self) -> Dict[str, Any]:
        async with self._lock:
            self.state = await extract_brass_state(self._page)
            legal = await self._refresh_legal_unlocked()
            formatted = format_legal_actions(legal, self.state)
            enriched = []
            for idx, act in enumerate(legal):
                item = dict(act)
                item["action_id"] = idx
                item["description"] = describe_macro_action(act)
                enriched.append(item)
            return {
                "ok": True,
                "step": self.valid_steps,
                "game_over": bool(self.state.get("game_over", False)),
                "count": len(enriched),
                "actions": enriched,
                "formatted": formatted,
            }

    async def play_action(self, action_id: int) -> Dict[str, Any]:
        async with self._lock:
            self.state = await extract_brass_state(self._page)
            if self.state.get("game_over"):
                score = compute_brass_harbor_reward(self.state, self.valid_steps, records=self.history)
                return {
                    "ok": False,
                    "error": "Game is already over (`game_over: true`).",
                    "step": self.valid_steps,
                    "game_over": True,
                    "harbor_score_preview": score,
                    "hud": format_status_hud(self.state, self.valid_steps, records=self.history),
                }

            legal = await self._refresh_legal_unlocked()
            if action_id < 0 or action_id >= len(legal):
                return {
                    "ok": False,
                    "error": f"Invalid action_id={action_id}. Valid range is 0..{max(0, len(legal) - 1)}.",
                    "step": self.valid_steps,
                    "count": len(legal),
                    "formatted_legal": format_legal_actions(legal, self.state),
                }

            chosen = legal[action_id]
            desc = describe_macro_action(chosen)
            exec_res = await execute_macro_action(self._page, chosen)
            if not exec_res.get("ok"):
                return {
                    "ok": False,
                    "error": f"Engine rejected action #{action_id}: {desc}",
                    "step": self.valid_steps,
                }

            rec = serialize_action_record(
                self.valid_steps,
                chosen,
                desc,
                prev_hash=self.chain_hash,
                seed=self.seed,
            )
            self.chain_hash = rec["chain_hash"]
            with self.trajectory_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as lf:
                lf.write(json.dumps(rec) + "\n")

            self.history.append(rec)
            self.valid_steps += 1
            self.state = await extract_brass_state(self._page)
            next_legal = await self._refresh_legal_unlocked()
            summary = compute_board_summary(self.state)
            score = compute_brass_harbor_reward(self.state, self.valid_steps, records=self.history)

            next_enriched = []
            for idx, act in enumerate(next_legal):
                item = dict(act)
                item["action_id"] = idx
                item["description"] = describe_macro_action(act)
                next_enriched.append(item)

            return {
                "ok": True,
                "executed_step": self.valid_steps - 1,
                "executed_action_id": action_id,
                "executed_description": desc,
                "step": self.valid_steps,
                "game_over": bool(self.state.get("game_over", False)),
                "state": self.state,
                "board_summary": {k: v for k, v in summary.items() if k != "city_link_dots"},
                "harbor_score_preview": score,
                "hud": format_status_hud(self.state, self.valid_steps, records=self.history),
                "next_legal_count": len(next_enriched),
                "next_actions": next_enriched,
                "formatted_next_legal": format_legal_actions(next_legal, self.state),
            }

    async def close(self) -> None:
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        finally:
            if self._static_server:
                self._static_server.shutdown()


async def handle_http_client(
    session: BrassEngineSession,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not request_line:
            writer.close()
            return
        parts = request_line.decode("utf-8", errors="replace").strip().split(" ")
        method = parts[0].upper() if len(parts) > 0 else "GET"
        path = parts[1] if len(parts) > 1 else "/"

        content_length = 0
        while True:
            hdr = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not hdr or hdr in (b"\r\n", b"\n"):
                break
            hdr_str = hdr.decode("utf-8", errors="replace").strip()
            if hdr_str.lower().startswith("content-length:"):
                content_length = int(hdr_str.split(":", 1)[1].strip())

        body_bytes = b""
        if content_length > 0:
            body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=10.0)

        status_code = 200
        if method == "GET" and path == "/health":
            payload: Dict[str, Any] = {
                "ok": True,
                "step": session.valid_steps,
                "game_over": bool(session.state.get("game_over", False)),
            }
        elif method == "GET" and path == "/status":
            payload = await session.get_status_payload()
        elif method == "GET" and path == "/legal":
            payload = await session.get_legal_payload()
        elif method == "GET" and path == "/history":
            payload = {
                "ok": True,
                "step": session.valid_steps,
                "history": session.history,
            }
        elif method == "POST" and path == "/play":
            req_json = json.loads(body_bytes.decode("utf-8") or "{}")
            action_id = int(req_json.get("action_id", -1))
            payload = await session.play_action(action_id)
            if not payload.get("ok"):
                status_code = 400
        else:
            status_code = 404
            payload = {"ok": False, "error": f"Unknown endpoint {method} {path}"}
    except Exception as exc:
        status_code = 500
        payload = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}

    resp_bytes = (json.dumps(payload) + "\n").encode("utf-8")
    reason = "OK" if status_code == 200 else "Error"
    header = (
        f"HTTP/1.1 {status_code} {reason}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(resp_bytes)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8")
    writer.write(header + resp_bytes)
    await writer.drain()
    writer.close()


async def run_daemon() -> None:
    host = os.environ.get("BRASS_DAEMON_HOST", "127.0.0.1")
    port = int(os.environ.get("BRASS_DAEMON_PORT", "18765"))
    webapp_dir = Path(
        os.environ.get(
            "BRASS_WEBAPP_DIR",
            str(Path(__file__).resolve().parent.parent / "webapp"),
        )
    )
    trajectory_path = Path(os.environ.get("BRASS_TRAJECTORY_PATH", "/app/trajectory.jsonl"))
    seed = int(os.environ.get("BRASS_EVAL_SEED", str(DEFAULT_EVAL_SEED)))
    difficulty = os.environ.get("BRASS_EVAL_DIFFICULTY", DEFAULT_EVAL_DIFFICULTY)

    session = BrassEngineSession(
        webapp_dir=webapp_dir,
        trajectory_path=trajectory_path,
        seed=seed,
        difficulty=difficulty,
    )
    await session.start()

    server = await asyncio.start_server(
        lambda r, w: handle_http_client(session, r, w),
        host=host,
        port=port,
    )
    try:
        async with server:
            await server.serve_forever()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(run_daemon())
