import asyncio
import inspect
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Type, TypeVar
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from models.schemas import (
    BrassMacroDecision,
    GameConfig,
    LLMResponse,
    StrategicPlan,
    VisionGroundingResponse,
    ZoomRefinement,
)
from agent.prompts import (
    TACTICAL_SYSTEM_PROMPT,
    STRATEGIC_SYSTEM_PROMPT,
    VISION_GROUNDING_SYSTEM_PROMPT,
    ZOOM_REFINEMENT_SYSTEM_PROMPT,
    build_game_briefing,
    build_strategic_briefing,
)

T = TypeVar("T", bound=BaseModel)

# Approximate USD cost per 1M tokens: (input_per_1m, output_per_1m)
MODEL_PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-7-sonnet-latest": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "gemini-3.8-pro": (1.25, 10.00),
    "gemini-3.8-flash": (0.15, 0.60),
    "gemini-3": (0.50, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_lower = (model or "").lower()
    rates = (2.50, 10.00)  # conservative default
    for key, val in MODEL_PRICING_PER_1M.items():
        if key in model_lower:
            rates = val
            break
    in_cost = (prompt_tokens / 1_000_000.0) * rates[0]
    out_cost = (completion_tokens / 1_000_000.0) * rates[1]
    return round(in_cost + out_cost, 6)


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object cleanly even if wrapped in markdown fences."""
    if not raw_text:
        raise ValueError("Empty response from LLM.")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # Try finding outermost {...}
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1], strict=False)
        raise


class LLMCallTelemetry(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 1


class LLMManager:
    def __init__(self, config: GameConfig):
        self.config = config
        self.game_briefing = build_game_briefing(config)
        self.strategic_briefing = build_strategic_briefing(config)
        self.openai_client = None
        self.anthropic_client = None

        if config.provider == "openai":
            kwargs: Dict[str, Any] = {"api_key": config.api_key}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self.openai_client = AsyncOpenAI(**kwargs)
        elif config.provider == "gemini":
            base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.openai_client = AsyncOpenAI(api_key=config.api_key, base_url=base_url)
        elif config.provider == "anthropic":
            self.anthropic_client = AsyncAnthropic(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def _redact_error(self, err: Exception) -> str:
        msg = str(err)
        if self.config.api_key and len(self.config.api_key) > 4:
            msg = msg.replace(self.config.api_key, "[REDACTED_API_KEY]")
        return msg

    async def list_available_models(self) -> Dict[str, Any]:
        """Fetch live model IDs available to this API key directly from the provider."""
        try:
            if self.config.provider in ("openai", "gemini"):
                res = await asyncio.wait_for(
                    self.openai_client.models.list(),
                    timeout=self.config.api_timeout_sec,
                )
                ids = []
                for m in getattr(res, "data", []):
                    mid = getattr(m, "id", "")
                    if not mid:
                        continue
                    if mid.startswith("models/"):
                        mid = mid[len("models/") :]
                    if self.config.provider == "gemini":
                        if "gemini" in mid.lower() and "embedding" not in mid.lower() and "tts" not in mid.lower():
                            ids.append(mid)
                    else:
                        if any(k in mid.lower() for k in ("gpt-4", "gpt-5", "o3", "o4")) and "audio" not in mid.lower() and "realtime" not in mid.lower():
                            ids.append(mid)
                ids = sorted(set(ids), reverse=True)
                return {"ok": True, "models": ids}
            elif self.config.provider == "anthropic":
                res = await asyncio.wait_for(
                    self.anthropic_client.models.list(),
                    timeout=self.config.api_timeout_sec,
                )
                ids = [getattr(m, "id", "") for m in getattr(res, "data", []) if getattr(m, "id", "")]
                return {"ok": True, "models": ids}
            return {"ok": False, "error": f"Unsupported provider: {self.config.provider}"}
        except Exception as e:
            return {"ok": False, "error": self._redact_error(e)}

    async def test_connection(self) -> Dict[str, Any]:
        """Lightweight text ping to verify API key and selected model availability."""
        try:
            raw_text, p_tok, c_tok = await asyncio.wait_for(
                self._raw_vision_completion(
                    system_prompt="Respond strictly with raw JSON: {\"ok\": true}",
                    user_prompt="Verify connection by returning {\"ok\": true}.",
                    image_base64=None,
                ),
                timeout=self.config.api_timeout_sec,
            )
            cost = estimate_cost_usd(self.config.model, p_tok, c_tok)
            return {
                "ok": True,
                "provider": self.config.provider,
                "model": self.config.model,
                "tokens": p_tok + c_tok,
                "cost_usd": cost,
                "sample": raw_text[:120],
            }
        except Exception as e:
            return {
                "ok": False,
                "error": self._redact_error(e),
            }

    async def get_strategic_plan(
        self, image_base64: str, strategic_context: str
    ) -> Tuple[StrategicPlan, LLMCallTelemetry]:
        system_prompt = f"{STRATEGIC_SYSTEM_PROMPT}\n\n=== GAME RULES & OBJECTIVES ===\n{self.strategic_briefing}"
        user_prompt = (
            f"=== PURE GAME-DOMAIN PROGRESSION ===\n{strategic_context}\n\n"
            "Analyze the current unannotated game screenshot strictly as a game strategist (ignoring UI button positions or past click ease) and return an updated StrategicPlan JSON."
        )
        return await self._validated_vision_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=image_base64,
            schema_cls=StrategicPlan,
        )

    async def get_next_action(
        self, image_base64: str, session_context: str
    ) -> Tuple[LLMResponse, LLMCallTelemetry]:
        """
        Query Vision LLM for next tactical action.
        Note: Returns (LLMResponse, LLMCallTelemetry) tuple; see get_next_action_only for simple callers.
        """
        system_prompt = f"{TACTICAL_SYSTEM_PROMPT}\n\n=== GAME CONFIGURATION ===\n{self.game_briefing}"
        user_prompt = (
            f"{session_context}\n\n"
            "Inspect the attached 1280x720 screenshot carefully and output your next tactical action as strict JSON."
        )
        return await self._validated_vision_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=image_base64,
            schema_cls=LLMResponse,
        )

    async def get_visual_grounding(
        self, grounding_image_base64: str, context_hint: str
    ) -> Tuple[VisionGroundingResponse, LLMCallTelemetry]:
        """
        Screenshot-based visual analysis pass for Canvas/WebGL games.
        Identifies interactive game objects, board tiles/nodes, and clickable areas with confidence scores.
        """
        system_prompt = f"{VISION_GROUNDING_SYSTEM_PROMPT}\n\n=== GAME CONTEXT ===\n{self.game_briefing}"
        user_prompt = (
            f"{context_hint}\n\n"
            "Detect currently interactive Canvas/WebGL targets on this 1280x720 screenshot and return raw JSON."
        )
        return await self._validated_vision_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=grounding_image_base64,
            schema_cls=VisionGroundingResponse,
            max_correction_retries=1,
        )

    async def get_zoom_refinement(
        self, zoom_image_base64: str, target_description: str
    ) -> Tuple[ZoomRefinement, LLMCallTelemetry]:
        """Fine-grained 4x magnifier pass that locks onto the exact cell ('A1'..'L9') + 9-point sub_pos."""
        user_prompt = (
            f"INTENDED CLICK TARGET:\n{target_description}\n\n"
            "Locate this exact clickable target inside the 4x magnified A1..L9 grid and return raw JSON."
        )
        return await self._validated_vision_call(
            system_prompt=ZOOM_REFINEMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_base64=zoom_image_base64,
            schema_cls=ZoomRefinement,
            max_correction_retries=1,
        )

    async def choose_brass_macro_action(
        self,
        image_base64: Optional[str],
        state_summary_text: str,
        legal_actions: List[Dict[str, Any]],
        player_name: str,
        opponent_name: str,
    ) -> Tuple[BrassMacroDecision, LLMCallTelemetry]:
        """
        Multimodal 2-Player Brass: Birmingham turn chooser for head-to-head AI Duels.
        Sees the live 1280x720 board screenshot + public board state + ONLY its own private hand,
        and selects 1 legal macro-action index [0 .. K-1] with zero RL advisor leakage.
        """
        system_prompt = (
            f"You are '{player_name}', an expert grandmaster playing an official 2-player match of "
            f"Brass: Birmingham against '{opponent_name}'.\n\n"
            "=== OFFICIAL 2-PLAYER BRASS: BIRMINGHAM RULES ===\n"
            "1. DECK & ROUNDS: Uses the official 40-card 2-player deck (northern 3p/4p cities & cotton/goods industry cards excluded). "
            "Canal Era deals 38 playable cards (Round 1 = 1 action/player, Rounds 2-10 = 2 actions/player). "
            "Rail Era deals all 40 cards (Rounds 1-10 = 2 actions/player).\n"
            "2. ERA SCORING: At the end of Canal Era (Round 10) and Rail Era (Round 10):\n"
            "   - Links score 1 VP per link-dot (●) across their two endpoints. ONLY FLIPPED industry tiles and active Merchant ports (2 ●) contribute link-dots!\n"
            "   - FLIPPED industry tiles score their printed VP. Unflipped consumer tiles (Cotton/Goods/Pottery) and un-depleted mines/breweries score 0 VP!\n"
            "   - Canal -> Rail Transition: All Canal links and Level 1 board tiles are removed; Level 2+ board tiles SURVIVE and score AGAIN at the end of Rail Era!\n"
            "3. TURN ORDER: At the end of each round, turn order for the next round goes to whoever SPENT LESS money (£ Spent ascending; ties preserve relative order).\n"
            "4. RESOURCES & SPECIAL MOVES:\n"
            "   - Iron can be consumed free from ANY iron works on the board (no network required); Coal requires network connection to closest coal mine or Merchant market.\n"
            "   - Sell flips 1 or more connected unflipped Cotton/Goods/Pottery tiles in a single action (using your own Brewery anywhere, a connected opponent Brewery, or connected Merchant beer).\n"
            "   - Develop removes 1-2 lowest tiles from your mat (1 Iron each, including 2x on the same industry; Pottery L1/L3 lightbulb tiles cannot be developed).\n"
            "   - In Rail Era, Network can build 1 rail (£5 + Coal) OR a Double Rail in 1 action (£15 + 2 Coal + 1 Beer).\n\n"
            "Respond STRICTLY with raw JSON matching this schema:\n"
            "{\n"
            '  "observation": "1 concise sentence assessing board, cash/turn-order, and VP race.",\n'
            '  "reasoning_summary": "1-2 concise sentences explaining why your chosen move maximizes VP / tempo.",\n'
            '  "chosen_index": 0,\n'
            '  "current_subgoal": "Short next-turn objective."\n'
            "}"
        )
        actions_block = "\n".join(
            f"  [{idx}] {act.get('description', act.get('action_type', 'move'))}"
            for idx, act in enumerate(legal_actions)
        )
        user_prompt = (
            f"{state_summary_text}\n\n"
            f"=== LEGAL MACRO-ACTIONS FOR {player_name} (choose 0..{max(0, len(legal_actions) - 1)}) ===\n"
            f"{actions_block}\n\n"
            f"Inspect the board screenshot and state above, and return strict JSON with `chosen_index` in 0..{max(0, len(legal_actions) - 1)}."
        )
        decision, tel = await self._validated_vision_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=image_base64 or "",
            schema_cls=BrassMacroDecision,
            max_correction_retries=2,
        )
        if legal_actions:
            decision.chosen_index = max(0, min(len(legal_actions) - 1, int(decision.chosen_index)))
        else:
            decision.chosen_index = 0
        return decision, tel

    async def _validated_vision_call(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        schema_cls: Type[T],
        max_correction_retries: int = 2,
    ) -> Tuple[T, LLMCallTelemetry]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        calls = 0
        current_prompt = user_prompt
        last_err: Exception = ValueError("Unknown validation failure")
        token_multiplier = 1

        for attempt in range(max_correction_retries + 1):
            try:
                raw_text, p_tok, c_tok = await asyncio.wait_for(
                    self._raw_vision_completion(
                        system_prompt,
                        current_prompt,
                        image_base64,
                        token_multiplier=token_multiplier,
                    ),
                    timeout=self.config.api_timeout_sec,
                )
                calls += 1
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"LLM API request timed out after {self.config.api_timeout_sec}s."
                )
            except Exception as e:
                raise RuntimeError(f"LLM API error: {self._redact_error(e)}")

            try:
                parsed_dict = extract_json_object(raw_text)
                validated = schema_cls(**parsed_dict)
                telemetry = LLMCallTelemetry(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    cost_usd=estimate_cost_usd(
                        self.config.model, total_prompt_tokens, total_completion_tokens
                    ),
                    api_calls=calls,
                )
                return validated, telemetry
            except (json.JSONDecodeError, ValidationError, ValueError) as val_err:
                last_err = val_err
                err_str = str(val_err)
                # If output was cut off mid-JSON (e.g. thinking model exhausted max_tokens), double token budget
                if "Unterminated" in err_str or "Expecting" in err_str or not raw_text.rstrip().endswith("}"):
                    token_multiplier *= 2
                if attempt < max_correction_retries:
                    current_prompt = (
                        f"{user_prompt}\n\n"
                        f"CORRECTION REQUIRED: Your previous JSON response was invalid or truncated ({err_str[:240]}).\n"
                        f"Keep every text field VERY concise (1 short sentence max) and return ONLY valid raw JSON."
                    )

        raise RuntimeError(f"LLM returned invalid action JSON after retries: {last_err}")

    async def _raw_vision_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str | None,
        token_multiplier: int = 1,
    ) -> Tuple[str, int, int]:
        if self.config.provider in ("openai", "gemini"):
            return await self._call_openai_compatible(
                system_prompt, user_prompt, image_base64, token_multiplier=token_multiplier
            )
        elif self.config.provider == "anthropic":
            return await self._call_anthropic(
                system_prompt, user_prompt, image_base64, token_multiplier=token_multiplier
            )
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str | None,
        token_multiplier: int = 1,
    ) -> Tuple[str, int, int]:
        user_content = [{"type": "text", "text": user_prompt}]
        if image_base64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high",
                    },
                }
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content if image_base64 else user_prompt},
        ]

        model_id = self.config.model.strip()
        if self.config.provider == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/") :]

        model_lower = model_id.lower()
        is_reasoning_series = (
            model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("o4")
        )
        is_thinking_model = is_reasoning_series or self.config.provider == "gemini" or "gemini" in model_lower

        # Gemini 2.5+/3.x and o-series count internal thinking tokens toward max_tokens;
        # enforce at least 4096 so thinking never truncates the JSON response.
        base_tokens = max(self.config.max_tokens, 4096) if is_thinking_model else self.config.max_tokens
        effective_max_tokens = min(16384, base_tokens * token_multiplier)

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }

        if not is_reasoning_series:
            kwargs["max_tokens"] = effective_max_tokens
            kwargs["temperature"] = self.config.temperature
            if self.config.provider in ("openai", "gemini"):
                kwargs["response_format"] = {"type": "json_object"}
            if self.config.provider == "gemini":
                # Keep thinking concise for fast real-time game loops
                kwargs["reasoning_effort"] = "low"
        else:
            kwargs["max_completion_tokens"] = effective_max_tokens

        try:
            response = await self.openai_client.chat.completions.create(**kwargs)
        except Exception as api_err:
            err_msg = str(api_err).lower()
            # Graceful fallback if a model doesn't support reasoning_effort or response_format
            if "reasoning_effort" in kwargs and ("reasoning" in err_msg or "unsupported" in err_msg or "unknown" in err_msg or "400" in err_msg):
                kwargs.pop("reasoning_effort", None)
                try:
                    response = await self.openai_client.chat.completions.create(**kwargs)
                except Exception:
                    kwargs.pop("response_format", None)
                    response = await self.openai_client.chat.completions.create(**kwargs)
            elif "response_format" in kwargs and ("response_format" in err_msg or "json_object" in err_msg):
                kwargs.pop("response_format", None)
                response = await self.openai_client.chat.completions.create(**kwargs)
            else:
                raise

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0
        return content, p_tok, c_tok

    async def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str | None,
        token_multiplier: int = 1,
    ) -> Tuple[str, int, int]:
        content_blocks = []
        if image_base64:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64,
                    },
                }
            )
        content_blocks.append(
            {
                "text": user_prompt + "\n\nRespond strictly with raw JSON only.",
                "type": "text",
            }
        )

        effective_max_tokens = min(8192, self.config.max_tokens * token_multiplier)

        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content_blocks}],
            "max_tokens": effective_max_tokens,
        }
        # Newer anthropic SDKs (e.g. 2.8.0+) removed `temperature` from AsyncMessages.create()
        if "temperature" in inspect.signature(self.anthropic_client.messages.create).parameters:
            create_kwargs["temperature"] = min(1.0, self.config.temperature)

        response = await self.anthropic_client.messages.create(**create_kwargs)

        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        content = "\n".join(text_parts)
        usage = getattr(response, "usage", None)
        p_tok = getattr(usage, "input_tokens", 0) or 0
        c_tok = getattr(usage, "output_tokens", 0) or 0
        return content, p_tok, c_tok
