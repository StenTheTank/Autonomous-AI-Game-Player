"""Custom Harbor baseline & adversarial-exploit agents for verifying Brass: Birmingham reward integrity."""

from __future__ import annotations

from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class StopEarlyAgent(BaseAgent):
    """Plays 19 Canal Era turns (building & scoring Canal VP) and halts before completing the Rail Era."""

    capabilities = AgentCapabilities(windows=True)

    @staticmethod
    @override
    def name() -> str:
        return "stop-early"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        script = (
            "python3 -c '"
            "import sys; sys.path.insert(0, \"/opt/brass\"); import brass_cli; "
            "brass_cli.ensure_daemon_running(); "
            "[brass_cli._http_json(\"POST\", \"/play\", {\"action_id\": 0}) for _ in range(19)]"
            "'"
        )
        await environment.exec(command=script)


class PassSpamAgent(BaseAgent):
    """Completes all 39 turns by passing every turn (reaches game_over=True with 0 VP)."""

    capabilities = AgentCapabilities(windows=True)

    @staticmethod
    @override
    def name() -> str:
        return "pass-spam"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        script = (
            "python3 -c '"
            "import sys; sys.path.insert(0, \"/opt/brass\"); import brass_cli; "
            "brass_cli.ensure_daemon_running()\n"
            "for _ in range(45):\n"
            "    st = brass_cli._http_json(\"GET\", \"/status\")[\"state\"]\n"
            "    if st.get(\"game_over\"): break\n"
            "    acts = brass_cli._http_json(\"GET\", \"/legal\").get(\"actions\") or []\n"
            "    p_i = next((i for i, a in enumerate(acts) if a.get(\"action_type\") == \"pass\"), len(acts) - 1)\n"
            "    if brass_cli._http_json(\"POST\", \"/play\", {\"action_id\": int(p_i)}).get(\"game_over\"): break\n"
            "'"
        )
        await environment.exec(command=script)
