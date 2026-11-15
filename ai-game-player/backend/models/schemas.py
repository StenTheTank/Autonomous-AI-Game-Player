import math
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from urllib.parse import urlparse


ActionType = Literal[
    "click_target",
    "click_coordinates",
    "hover_target",
    "key_press",
    "key_hold",
    "key_release",
    "key_combo",
    "mouse_click",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "mouse_drag",
    "scroll",
    "wait",
    "request_observation",
    "declare_success",
    "declare_failure",
]


KEY_ALIASES: Dict[str, str] = {
    "SPACE": "Space",
    "SPACEBAR": "Space",
    " ": "Space",
    "ENTER": "Enter",
    "RETURN": "Enter",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "UP": "ArrowUp",
    "ARROWUP": "ArrowUp",
    "ARROW_UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "ARROWDOWN": "ArrowDown",
    "ARROW_DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "ARROWLEFT": "ArrowLeft",
    "ARROW_LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "ARROWRIGHT": "ArrowRight",
    "ARROW_RIGHT": "ArrowRight",
    "TAB": "Tab",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "DEL": "Delete",
    "SHIFT": "Shift",
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
}


def normalize_key_name(key: str) -> str:
    if not key:
        return key
    stripped = key.strip()
    upper = stripped.upper()
    if upper in KEY_ALIASES:
        return KEY_ALIASES[upper]
    # Single alphanumeric characters should be lowercase for Playwright unless Shift is desired
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.lower()
    return stripped


def macro_cell_to_coords(cell: str, width: int = 1280, height: int = 720, cols: int = 16, rows: int = 9) -> tuple[int, int]:
    """Convert a macro grid code like 'E7' (Col A..P, Row 1..9) to center pixel (x, y)."""
    cleaned = (cell or "").strip().upper()
    if len(cleaned) < 2 or not cleaned[0].isalpha() or not cleaned[1:].isdigit():
        raise ValueError(f"Invalid grid_cell '{cell}'. Expected format 'A1'..'P9'.")
    col_idx = ord(cleaned[0]) - ord("A")
    row_idx = int(cleaned[1:]) - 1
    if not (0 <= col_idx < cols and 0 <= row_idx < rows):
        raise ValueError(f"grid_cell '{cell}' out of bounds (A..{chr(ord('A') + cols - 1)}, 1..{rows}).")
    cell_w = width / cols
    cell_h = height / rows
    cx = int(round((col_idx + 0.5) * cell_w))
    cy = int(round((row_idx + 0.5) * cell_h))
    return cx, cy


def coords_to_macro_cell(x: int, y: int, width: int = 1280, height: int = 720, cols: int = 16, rows: int = 9) -> str:
    """Convert viewport pixel (x, y) into its 16x9 macro cell label ('A1'..'P9')."""
    col_idx = max(0, min(cols - 1, int(x) * cols // max(1, width)))
    row_idx = max(0, min(rows - 1, int(y) * rows // max(1, height)))
    return f"{chr(ord('A') + col_idx)}{row_idx + 1}"


def normalize_target_id(raw_id: Any) -> Optional[str]:
    """Normalize target IDs like 'target_002', '#target_002', '2', or 'settlement_03'."""
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    if not s:
        return None
    if s.startswith("#"):
        s = s[1:].strip()
    if s.isdigit():
        return f"target_{int(s):03d}"
    low = s.lower()
    if low.startswith("target_"):
        suffix = low[len("target_") :]
        if suffix.isdigit():
            return f"target_{int(suffix):03d}"
    return s


class InteractionTarget(BaseModel):
    id: str = Field(..., description="Unique target ID, e.g. 'target_001'.")
    label: str = Field(..., description="Descriptive human/LLM-readable label for the target.")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x, y, width, height] in viewport pixels.")
    center: List[int] = Field(..., min_length=2, max_length=2, description="[cx, cy] center coordinates in viewport pixels.")
    source: Literal["dom", "vision", "cv"] = Field(..., description="Detection pipeline source.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0).")
    screenshot_version: int = Field(1, description="Screenshot version this target was detected on.")
    timestamp: float = Field(0.0, description="Unix timestamp when target was detected.")
    role: Optional[str] = Field(None, description="Optional element role/tag (e.g. 'button', 'node', 'tile').")
    dom_selector_hint: Optional[str] = Field(None, description="Optional internal selector hint for live DOM re-verification.")


class InteractionMap(BaseModel):
    screenshot_version: int = 0
    timestamp: float = 0.0
    viewport: List[int] = Field(default_factory=lambda: [1280, 720])
    device_pixel_ratio: float = 1.0
    has_canvas_or_webgl: bool = False
    regenerated: bool = True
    regeneration_reason: str = "initial"
    targets: List[InteractionTarget] = Field(default_factory=list)


class VisionDetectedObject(BaseModel):
    label: str = Field(..., description="Descriptive label (e.g., 'Empty settlement spot', 'Roll Dice button').")
    grid_cell: Optional[str] = Field(None, description="16x9 cell ('A1'..'P9') containing the object's center.")
    bbox: Optional[List[int]] = Field(None, description="[x, y, width, height] in 1280x720 viewport pixels.")
    center: Optional[List[int]] = Field(None, description="[cx, cy] in 1280x720 viewport pixels.")
    cv_ref_id: Optional[str] = Field(None, description="Optional CV candidate ID ('cv_01', ...) if matching a CV box.")
    confidence: float = Field(0.85, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0.")


class VisionGroundingResponse(BaseModel):
    scene_summary: str = Field("", description="Brief 1-sentence visual summary of interactive game elements.")
    targets: List[VisionDetectedObject] = Field(default_factory=list)


class Action(BaseModel):
    type: ActionType = Field(..., description="Structured action type to execute in the browser.")
    target_id: Optional[str] = Field(None, description="Target ID from the current Interaction Map (e.g., 'target_002').")
    from_target_id: Optional[str] = Field(None, description="Starting target ID from Interaction Map for mouse_drag.")
    key: Optional[str] = Field(None, description="Keyboard key name (e.g., 'Space', 'ArrowUp', 'w', 'Enter').")
    keys: Optional[List[str]] = Field(None, description="Multiple simultaneous keys for key_combo (e.g., ['ArrowRight', 'Space']).")
    duration_ms: Optional[int] = Field(None, ge=0, le=15000, description="Duration in milliseconds for key_press, key_hold, mouse_drag, or wait.")
    element_id: Optional[int] = Field(None, ge=1, description="Legacy numeric clickable UI element ID (#1, #2, ...).")
    grid_cell: Optional[str] = Field(None, description="Macro screen grid cell ('A1'..'P9') containing the target.")
    from_grid_cell: Optional[str] = Field(None, description="Starting macro grid cell ('A1'..'P9') for mouse_drag.")
    x: Optional[int] = Field(None, description="Target X pixel coordinate within viewport (0..1280).")
    y: Optional[int] = Field(None, description="Target Y pixel coordinate within viewport (0..720).")
    from_x: Optional[int] = Field(None, description="Starting X coordinate for mouse_drag.")
    from_y: Optional[int] = Field(None, description="Starting Y coordinate for mouse_drag.")
    button: Optional[Literal["left", "right", "middle"]] = Field("left", description="Mouse button.")
    clicks: Optional[int] = Field(1, ge=1, le=3, description="Number of clicks (1 for single, 2 for double).")
    scroll_x: Optional[int] = Field(0, description="Horizontal scroll delta in pixels.")
    scroll_y: Optional[int] = Field(0, description="Vertical scroll delta in pixels (positive = down, negative = up).")

    @model_validator(mode="before")
    @classmethod
    def accept_action_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            # Support {"action": "click_target", "target_id": "target_002"} shorthand
            if "type" not in out and "action" in out and isinstance(out["action"], str):
                out["type"] = out.pop("action")
            return out
        return data

    @field_validator("button", mode="before")
    @classmethod
    def normalize_button(cls, v: Any) -> str:
        if v is None:
            return "left"
        s = str(v).lower().strip()
        return s if s in ("left", "right", "middle") else "left"

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.key:
            self.key = normalize_key_name(self.key)
        if self.keys:
            self.keys = [normalize_key_name(k) for k in self.keys if k and str(k).strip()]

        if self.target_id:
            self.target_id = normalize_target_id(self.target_id)
        elif self.element_id is not None:
            self.target_id = f"target_{int(self.element_id):03d}"

        if self.from_target_id:
            self.from_target_id = normalize_target_id(self.from_target_id)

        if self.grid_cell:
            self.grid_cell = self.grid_cell.strip().upper()
            cell_cx, cell_cy = macro_cell_to_coords(self.grid_cell)
            if self.x is None or self.y is None:
                self.x, self.y = cell_cx, cell_cy
            else:
                # Guard against 0..1000 normalized coordinates (e.g. Gemini outputting x=285, y=465 for (365, 335) in E5)
                raw_dist = math.hypot(self.x - cell_cx, self.y - cell_cy)
                norm_x = int(round((self.x / 1000.0) * 1280))
                norm_y = int(round((self.y / 1000.0) * 720))
                norm_dist = math.hypot(norm_x - cell_cx, norm_y - cell_cy)
                if 0 <= self.x <= 1000 and 0 <= self.y <= 1000 and raw_dist > 58 and norm_dist + 20 < raw_dist and norm_dist <= 75:
                    self.x, self.y = norm_x, norm_y
                elif raw_dist > 95:
                    self.x, self.y = cell_cx, cell_cy
        if self.from_grid_cell:
            self.from_grid_cell = self.from_grid_cell.strip().upper()
            if self.from_x is None or self.from_y is None:
                self.from_x, self.from_y = macro_cell_to_coords(self.from_grid_cell)

        if self.type in ("key_press", "key_hold", "key_release"):
            if not self.key and self.keys:
                self.key = self.keys[0]
            if not self.key:
                raise ValueError(f"Action '{self.type}' requires a non-empty 'key' field.")
        elif self.type == "key_combo":
            if not self.keys and self.key:
                parts = [normalize_key_name(p) for p in self.key.replace("+", " ").split() if p]
                self.keys = parts
            if not self.keys:
                raise ValueError("Action 'key_combo' requires a non-empty 'keys' list.")
        elif self.type in ("click_target", "hover_target"):
            if not self.target_id:
                if self.x is not None and self.y is not None:
                    # Graceful upgrade if LLM wrote click_target with x, y
                    self.type = "click_coordinates" if self.type == "click_target" else "mouse_move"
                else:
                    raise ValueError(f"Action '{self.type}' requires 'target_id' from the Interaction Map.")
        elif self.type == "click_coordinates":
            if self.x is None or self.y is None:
                if self.target_id:
                    self.type = "click_target"
                else:
                    raise ValueError("Action 'click_coordinates' requires integer 'x' and 'y' coordinates.")
        elif self.type in ("mouse_click", "mouse_move"):
            if not self.target_id and self.element_id is None and (self.x is None or self.y is None):
                raise ValueError(
                    f"Action '{self.type}' requires 'target_id', 'grid_cell' ('A1'..'P9'), or ('x', 'y') coordinates."
                )
        elif self.type == "mouse_drag":
            if not self.target_id and (self.x is None or self.y is None):
                raise ValueError("Action 'mouse_drag' requires 'target_id', 'grid_cell', or ('x', 'y') coordinates.")
        return self


class ZoomRefinement(BaseModel):
    target_visible: bool = Field(True, description="True if the intended target is visible inside this zoomed crop.")
    cell: str = Field("F5", description="Cell code ('A1'..'L9') containing the exact center of the target.")
    sub_pos: Literal[
        "center",
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ] = Field("center", description="9-point position inside the chosen 20x20px cell.")
    observation: str = Field("", description="Brief confirmation of where the target sits in the zoomed view.")


class LLMResponse(BaseModel):
    observation: str = Field(..., min_length=1, description="Concise visual analysis of the current game screen.")
    previous_action_evaluation: Optional[str] = Field(
        None,
        description="Assessment of whether the previous action achieved its expected result."
    )
    reasoning_summary: str = Field(..., min_length=1, description="Tactical and strategic reasoning for choosing this action.")
    action: Action = Field(..., description="The structured browser action to perform next.")
    expected_result: str = Field(..., min_length=1, description="Concrete visual change expected after executing this action.")
    goal_status: Literal["in_progress", "success", "failure"] = Field(
        "in_progress",
        description="Overall objective status."
    )
    learned_lesson: Optional[str] = Field(
        None,
        description="Optional short rule, pattern, or mistake learned from recent observations."
    )
    current_subgoal: Optional[str] = Field(
        None,
        description="Immediate sub-objective currently being pursued."
    )
    progress_percent: Optional[int] = Field(
        None, ge=0, le=100, description="Estimated progress percentage toward main objective (0-100)."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_action_payload(cls, data: Any) -> Any:
        """Accept top-level {'action': 'click_target', 'target_id': '...'} in addition to nested action object."""
        if isinstance(data, dict) and isinstance(data.get("action"), str):
            out = dict(data)
            action_obj: Dict[str, Any] = {"type": out.pop("action")}
            for k in (
                "target_id",
                "from_target_id",
                "element_id",
                "grid_cell",
                "from_grid_cell",
                "x",
                "y",
                "from_x",
                "from_y",
                "key",
                "keys",
                "duration_ms",
                "button",
                "clicks",
                "scroll_x",
                "scroll_y",
            ):
                if k in out:
                    action_obj[k] = out.pop(k)
            out["action"] = action_obj
            return out
        return data


class StrategicPlan(BaseModel):
    game_state_summary: str = Field(..., description="High-level assessment of current game phase and situation.")
    current_strategy: str = Field(..., description="Overall winning strategy and tactical approach.")
    sub_goals: List[str] = Field(default_factory=list, description="Ordered list of concrete short-term sub-goals.")
    known_mechanics: List[str] = Field(default_factory=list, description="Confirmed game mechanics observed so far.")
    lessons_learned: List[str] = Field(default_factory=list, description="Mistakes to avoid and effective tactics discovered.")
    recommended_action_cadence_ms: Optional[int] = Field(None, description="Suggested action duration if adjustment needed.")


class BrassMacroDecision(BaseModel):
    observation: str = Field("Evaluated 2P Brass: Birmingham board.", description="Concise 1-sentence reading of board & economy.")
    reasoning_summary: str = Field(..., min_length=1, description="Strategic reasoning for choosing this macro-action.")
    chosen_index: int = Field(..., ge=0, description="0-based index of the chosen legal macro-action.")
    current_subgoal: Optional[str] = Field(None, description="Short sub-objective for the next 1-2 turns.")

    @model_validator(mode="before")
    @classmethod
    def normalize_macro_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if "chosen_index" not in out:
                for alias in ("action_index", "move_index", "index", "choice"):
                    if alias in out:
                        out["chosen_index"] = out[alias]
                        break
            if "reasoning_summary" not in out:
                for r_alias in ("reasoning", "rationale", "explanation"):
                    if r_alias in out and out[r_alias]:
                        out["reasoning_summary"] = out[r_alias]
                        break
            if not out.get("observation"):
                out["observation"] = "Evaluated live 2P Brass: Birmingham board state."
            return out
        return data


class GameConfig(BaseModel):
    # LLM / Policy configuration
    provider: Literal["openai", "anthropic", "gemini", "brass_rl"] = "openai"
    api_key: str = Field("", description="Provider API key (kept in memory only unless explicitly opted in; optional for brass_rl).")
    model: str = "gpt-4o"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=150, le=16384)
    base_url: Optional[str] = Field(None, description="Optional custom OpenAI-compatible base URL.")

    # Head-to-Head 2P Duel & Human-vs-AI configuration
    match_mode: Literal["single", "brass_2p_duel", "brass_human_vs_ai"] = "single"
    opponent_model: str = Field("brass-ppo-best", description="Local PyTorch PPO checkpoint for 2P Duel opponent.")
    api_player_seat: Literal[0, 1] = Field(0, description="Seat index for the API Challenger in 2P Duel (0 = Player 1, 1 = Player 2).")
    human_player_seat: Literal[0, 1] = Field(0, description="Seat index for the Human player in Human vs AI mode (0 = Player 1, 1 = Player 2).")
    game_seed: int = Field(202611, ge=0, le=999999999, description="Deterministic seed for official 2P Brass deck & setup.")

    # Game configuration
    url: str = Field(..., min_length=4)
    game_name: str = "Browser Game"
    game_description: str = ""
    rules: str = ""
    keyboard_controls: str = ""
    mouse_controls: str = ""
    controls: str = ""  # Backward-compatible combined controls field
    objective: str = "Play the game effectively and achieve the highest score or win state."
    secondary_objectives: str = ""
    winning_conditions: str = ""
    losing_conditions: str = ""
    additional_instructions: str = ""

    # Agent & safety configuration
    strategy_preference: Literal["balanced", "aggressive", "cautious", "exploratory", "speedrun"] = "balanced"
    default_action_duration_ms: int = Field(180, ge=20, le=5000)
    max_action_duration_ms: int = Field(4000, ge=100, le=10000)
    observation_interval_ms: int = Field(800, ge=100, le=15000)
    max_iterations: int = Field(100, ge=1, le=2000)
    max_session_duration_sec: int = Field(900, ge=15, le=7200)
    max_budget_usd: float = Field(2.00, ge=0.01, le=100.0)
    strategic_planning_interval: int = Field(8, ge=2, le=50)
    max_consecutive_failures: int = Field(5, ge=1, le=20)
    api_timeout_sec: float = Field(35.0, ge=5.0, le=120.0)
    show_coordinate_grid_to_llm: bool = True
    precision_zoom_click: bool = True
    enable_vision_grounding: bool = True
    enable_cv_detection: bool = True
    enable_rl_advisor: bool = True
    min_target_confidence: float = Field(0.45, ge=0.1, le=0.99)
    headless: bool = False

    @model_validator(mode="after")
    def validate_provider_api_key(self) -> "GameConfig":
        if self.match_mode == "brass_2p_duel" and self.provider == "brass_rl":
            raise ValueError(
                "In 2P Duel mode, select an API provider (Gemini, OpenAI, or Anthropic) for the API Challenger."
            )
        if self.provider != "brass_rl" and not (self.api_key and self.api_key.strip()):
            raise ValueError(f"Provider '{self.provider}' requires a non-empty API key.")
        return self

    @field_validator("url")
    @classmethod
    def validate_safe_url(cls, v: str) -> str:
        url = v.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Game URL must start with http:// or https:// for browser safety.")
        if not parsed.netloc:
            raise ValueError("Game URL must include a valid host.")
        return url

    def sanitized_dict(self) -> Dict[str, Any]:
        """Return configuration dictionary with API key completely removed for safe logging/storage."""
        data = self.model_dump()
        data.pop("api_key", None)
        return data


class SessionStats(BaseModel):
    current_iteration: int = 0
    max_iterations: int = 100
    total_actions: int = 0
    elapsed_seconds: float = 0.0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    max_budget_usd: float = 2.0
    consecutive_failures: int = 0
    last_visual_delta_pct: float = 0.0
    progress_percent: int = 0


class SessionState(BaseModel):
    session_id: str = ""
    status: Literal[
        "idle", "starting", "planning", "running", "paused", "completed", "failed", "stopped", "error"
    ] = "idle"
    status_reason: str = ""
    browser_status: str = "disconnected"
    current_url: str = ""
    game_name: str = ""
    model: str = ""
    provider: str = ""
    current_objective: str = ""
    current_subgoal: str = ""
    current_strategy: str = ""
    sub_goals: List[str] = Field(default_factory=list)
    known_mechanics: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    last_observation: Optional[str] = None
    last_reasoning: Optional[str] = None
    last_expected_result: Optional[str] = None
    last_action: Optional[Dict[str, Any]] = None
    last_action_status: str = "idle"
    mouse_x: int = 640
    mouse_y: int = 360
    detected_elements_count: int = 0
    interaction_map: Optional[Dict[str, Any]] = None
    click_history: List[Dict[str, Any]] = Field(default_factory=list)
    duel_state: Optional[Dict[str, Any]] = None
    stats: SessionStats = Field(default_factory=SessionStats)


class RLTrainRequest(BaseModel):
    workers: int = Field(6, ge=1, le=12, description="Number of parallel headless Chromium tabs.")
    training_mode: Literal["self_play", "solo_mautoma"] = Field(
        "self_play",
        description="self_play = 2P population-vs-population matches; solo_mautoma = solo vs Mautoma AI.",
    )
    generations: int = Field(30, ge=1, le=5000, description="Number of evolutionary generations to run.")
    population_size: int = Field(12, ge=4, le=64, description="Number of candidate model genomes per generation.")
    games_per_individual: int = Field(
        2, ge=1, le=6, description="Paired evaluation games played per candidate genome each generation."
    )
    mutation_sigma: float = Field(
        0.05, ge=0.005, le=0.5, description="Gaussian weight & bias mutation standard deviation."
    )
    mutation_rate: float = Field(
        0.20, ge=0.01, le=1.0, description="Fraction of weights & biases mutated per offspring."
    )
    elite_count: int = Field(2, ge=1, le=16, description="Top elite genomes preserved unchanged into next generation.")
    resume_checkpoint: bool = Field(True, description="Include saved best checkpoint as an elite seed in Generation 1.")
    episodes: Optional[int] = Field(None, ge=1, le=50000, description="Optional total-games alias converted to generations.")

    @model_validator(mode="before")
    @classmethod
    def map_legacy_episodes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if "generations" not in out and out.get("episodes"):
                pop = int(out.get("population_size", 12))
                gpi = int(out.get("games_per_individual", 2))
                out["generations"] = max(1, math.ceil(int(out["episodes"]) / max(1, pop * gpi)))
            return out
        return data


class RLTrainingStatus(BaseModel):
    running: bool = False
    algorithm: str = "genetic_algorithm"
    training_mode: Literal["self_play", "solo_mautoma"] = "self_play"
    run_id: str = ""
    device: str = "cpu"
    workers: int = 0
    current_generation: int = 0
    target_generations: int = 0
    population_size: int = 12
    games_per_individual: int = 2
    elite_count: int = 2
    mutation_sigma: float = 0.05
    mutation_rate: float = 0.20
    generation_progress: str = "0 / 0"
    generation_best_vp: float = 0.0
    generation_mean_vp: float = 0.0
    champion_avg_vp: float = 0.0
    champion_fitness: float = 0.0
    target_episodes: int = 0
    completed_episodes: int = 0
    rolling_avg_vp: float = 0.0
    rolling_automa_vp: float = 0.0
    rolling_win_rate_pct: float = 0.0
    best_vp: int = 0
    best_rolling_vp: float = 0.0
    games_per_min: float = 0.0
    latest_policy_loss: float = 0.0
    latest_value_loss: float = 0.0
    latest_entropy: float = 0.0
    has_best_checkpoint: bool = False
    has_latest_checkpoint: bool = False
    status_message: str = "Idle"
    recent_episodes: List[Dict[str, Any]] = Field(default_factory=list)

