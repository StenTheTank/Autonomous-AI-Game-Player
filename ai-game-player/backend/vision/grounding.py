import base64
from collections import deque
import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from models.schemas import (
    InteractionMap,
    InteractionTarget,
    VisionGroundingResponse,
    coords_to_macro_cell,
    macro_cell_to_coords,
)

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# Optional OpenCV import: verify NumPy 2.x ABI compatibility first so legacy OpenCV builds
# (< 4.10 with NumPy >= 2.0) don't emit C-extension traceback noise on startup.
cv2 = None
try:
    import importlib.metadata as _meta

    _np_major = int(np.__version__.split(".")[0]) if np is not None else 1
    _cv_ok = True
    if _np_major >= 2:
        for _dist in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
            try:
                _ver_parts = [int(p) for p in _meta.version(_dist).split(".")[:2] if p.isdigit()]
                if _ver_parts and tuple(_ver_parts) < (4, 10):
                    _cv_ok = False
                    break
            except Exception:
                pass
    if _cv_ok:
        import cv2  # type: ignore
except Exception:
    cv2 = None


class VisualGroundingEngine:
    """
    General-purpose Hybrid Visual Grounding & Interaction Map Engine.
    Combines:
      1. Playwright DOM & Accessibility Tree bounding-box inspection (source='dom')
      2. Computer Vision salient marker / contour detection via OpenCV or NumPy+Pillow (source='cv')
      3. Vision LLM semantic object & board grounding (source='vision')
    Maintains screenshot versioning and invalidates stale targets automatically upon layout/screen changes.
    """

    def __init__(self, viewport_width: int = 1280, viewport_height: int = 720):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.screenshot_version: int = 0
        self._last_layout_state: Optional[Dict[str, Any]] = None
        self._cached_vision_targets: List[Dict[str, Any]] = []
        self._cached_vision_version: int = 0
        self.current_map: InteractionMap = InteractionMap(
            screenshot_version=0,
            viewport=[viewport_width, viewport_height],
        )

    def next_screenshot_version(self) -> int:
        self.screenshot_version += 1
        return self.screenshot_version

    def invalidate(self, reason: str = "manual_invalidation") -> None:
        """Force full regeneration of visual grounding on next observation."""
        self._cached_vision_targets = []
        self._last_layout_state = None
        self.current_map.regenerated = True
        self.current_map.regeneration_reason = reason

    def check_regeneration_needed(
        self,
        layout_meta: Dict[str, Any],
        dom_targets: List[Dict[str, Any]],
        visual_delta_pct: float,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine whether cached visual targets are stale due to navigation, scrolling,
        resizing, DPR shift, modal/dialog change, DOM change, or visual transition.
        """
        dom_sig = "|".join(
            f"{d.get('tag')}:{d.get('label', '')[:16]}@{d.get('x', 0) // 16},{d.get('y', 0) // 16}"
            for d in dom_targets[:18]
        )
        current_state = {
            "url": layout_meta.get("url", ""),
            "vw": layout_meta.get("viewport_w", self.viewport_width),
            "vh": layout_meta.get("viewport_h", self.viewport_height),
            "dpr": round(float(layout_meta.get("dpr", 1.0)), 2),
            "sx": int(layout_meta.get("scroll_x", 0)),
            "sy": int(layout_meta.get("scroll_y", 0)),
            "dialogs": layout_meta.get("dialog_signature", ""),
            "dom_sig": dom_sig,
        }

        if force or self._last_layout_state is None:
            self._last_layout_state = current_state
            return True, "initial_observation"

        prev = self._last_layout_state
        reason: Optional[str] = None
        if current_state["url"] != prev["url"]:
            reason = "navigation_url_changed"
        elif current_state["vw"] != prev["vw"] or current_state["vh"] != prev["vh"] or current_state["dpr"] != prev["dpr"]:
            reason = "viewport_or_dpr_resized"
        elif abs(current_state["sx"] - prev["sx"]) > 4 or abs(current_state["sy"] - prev["sy"]) > 4:
            reason = "viewport_scrolled"
        elif current_state["dialogs"] != prev["dialogs"]:
            reason = "dialog_or_modal_changed"
        elif visual_delta_pct >= 2.0:
            reason = f"screen_changed_{visual_delta_pct:.1f}pct"
        elif current_state["dom_sig"] != prev["dom_sig"]:
            reason = "dom_elements_updated"

        self._last_layout_state = current_state
        if reason:
            self._cached_vision_targets = []
            return True, reason
        return False, "stable_layout"

    def detect_cv_targets(
        self,
        raw_image: Image.Image,
        dom_targets: List[Dict[str, Any]],
        canvas_rects: Optional[List[List[int]]] = None,
        max_targets: int = 14,
    ) -> List[Dict[str, Any]]:
        """
        Computer-vision salient interactive region detector for Canvas/WebGL games.
        Uses OpenCV (cv2) if functional, otherwise runs a fast vectorized NumPy + Pillow
        local color-contrast & edge-ring connected-components detector.
        """
        if raw_image is None or np is None:
            return []

        w, h = self.viewport_width, self.viewport_height
        if raw_image.size != (w, h):
            raw_image = raw_image.resize((w, h))

        # Work on 2x downsampled grid (640x360) for <12ms execution speed
        sw, sh = w // 2, h // 2
        small_img = raw_image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")

        # Build valid region mask (include canvas/svg areas if present; exclude DOM buttons & top HUD)
        allowed = np.zeros((sh, sw), dtype=bool)
        if canvas_rects:
            for rect in canvas_rects:
                rx, ry, rw, rh = [int(v) for v in rect[:4]]
                x0 = max(2, rx // 2)
                y0 = max(2, ry // 2)
                x1 = min(sw - 2, (rx + rw) // 2)
                y1 = min(sh - 2, (ry + rh) // 2)
                if x1 > x0 and y1 > y0:
                    allowed[y0:y1, x0:x1] = True
        else:
            allowed[6 : sh - 6, 6 : sw - 6] = True

        # Mask out top cursor HUD pill and already-detected DOM bounding boxes (padded by BoxBlur radius)
        allowed[0:22, sw // 2 - 75 : sw // 2 + 75] = False
        for d in dom_targets:
            bx, by, bw, bh = d.get("bbox", [d["x"] - 12, d["y"] - 12, 24, 24])
            x0 = max(0, (bx - 18) // 2)
            y0 = max(0, (by - 18) // 2)
            x1 = min(sw, (bx + bw + 18) // 2)
            y1 = min(sh, (by + bh + 18) // 2)
            allowed[y0:y1, x0:x1] = False

        if not np.any(allowed):
            return []

        if cv2 is not None:
            try:
                return self._detect_with_opencv(small_img, allowed, max_targets)
            except Exception:
                pass

        return self._detect_with_numpy_pillow(small_img, allowed, max_targets)

    def _detect_with_opencv(
        self, small_img: Image.Image, allowed: "np.ndarray", max_targets: int
    ) -> List[Dict[str, Any]]:
        arr = np.array(small_img, dtype=np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 55, 150)
        edges[~allowed] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: List[Tuple[float, List[int], str]] = []
        for cnt in contours:
            sx, sy, sw_box, sh_box = cv2.boundingRect(cnt)
            bw, bh = sw_box * 2, sh_box * 2
            if bw < 14 or bh < 14 or bw > 200 or bh > 140:
                continue
            aspect = bw / float(max(1, bh))
            if aspect < 0.28 or aspect > 4.2:
                continue
            area = cv2.contourArea(cnt)
            fill_ratio = area / float(max(1, sw_box * sh_box))
            conf = min(0.85, max(0.55, 0.52 + 0.32 * fill_ratio))
            label = (
                "Canvas circular marker"
                if 0.72 <= aspect <= 1.38 and bw <= 52
                else "Canvas interactive object"
            )
            raw_boxes.append((conf, [sx * 2, sy * 2, bw, bh], label))

        return self._nms_and_format_cv(raw_boxes, max_targets)

    def _detect_with_numpy_pillow(
        self, small_img: Image.Image, allowed: "np.ndarray", max_targets: int
    ) -> List[Dict[str, Any]]:
        """
        Pure NumPy + Pillow local contrast & edge salience detector.
        Finds glowing rings, highlighted board intersections, tokens, and distinct Canvas buttons.
        """
        arr = np.asarray(small_img, dtype=np.float32)
        local_bg = np.asarray(small_img.filter(ImageFilter.BoxBlur(7)), dtype=np.float32)
        edge_img = np.asarray(small_img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)

        # Color/luminance pop relative to immediate 14px neighborhood + crisp edge strength
        color_diff = np.sqrt(np.sum((arr - local_bg) ** 2, axis=2))
        salience = (0.65 * color_diff + 0.35 * edge_img) * allowed.astype(np.float32)

        valid_vals = salience[allowed]
        if valid_vals.size < 100:
            return []

        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        if std_val < 3.5:
            return []

        thresh = mean_val + 2.25 * std_val
        mask = (salience >= thresh) & allowed

        # Light 3x3 dilation so thin circular rings connect into single components
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        dilated = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        ) & allowed

        visited = np.zeros_like(dilated, dtype=bool)
        ys, xs = np.nonzero(dilated)
        sh, sw = dilated.shape
        raw_boxes: List[Tuple[float, List[int], str]] = []

        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if visited[start_y, start_x]:
                continue
            q = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            pix_count = 0
            peak_sal = 0.0

            while q:
                cy, cx = q.popleft()
                pix_count += 1
                sv = float(salience[cy, cx])
                if sv > peak_sal:
                    peak_sal = sv
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= sh:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if 0 <= nx < sw and not visited[ny, nx] and dilated[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

            bw = (max_x - min_x + 1) * 2
            bh = (max_y - min_y + 1) * 2
            if pix_count < 8 or bw < 12 or bh < 12 or bw > 190 or bh > 130:
                continue
            aspect = bw / float(max(1, bh))
            if aspect < 0.28 or aspect > 4.2:
                continue

            z_score = (peak_sal - mean_val) / max(1.0, std_val)
            conf = round(min(0.84, max(0.55, 0.48 + 0.065 * z_score)), 2)
            if 0.72 <= aspect <= 1.38 and bw <= 48:
                label = "Highlighted board node / marker"
            elif aspect > 1.7:
                label = "Canvas button / bar region"
            else:
                label = "Salient interactive region"

            raw_boxes.append((conf, [min_x * 2, min_y * 2, bw, bh], label))

        return self._nms_and_format_cv(raw_boxes, max_targets)

    def _nms_and_format_cv(
        self, raw_boxes: List[Tuple[float, List[int], str]], max_targets: int
    ) -> List[Dict[str, Any]]:
        raw_boxes.sort(key=lambda item: item[0], reverse=True)
        kept: List[Dict[str, Any]] = []
        for conf, bbox, label in raw_boxes:
            bx, by, bw, bh = bbox
            cx = bx + bw // 2
            cy = by + bh // 2
            if any(math.hypot(cx - k["center"][0], cy - k["center"][1]) < 26 for k in kept):
                continue
            cv_id = f"cv_{len(kept) + 1:02d}"
            kept.append(
                {
                    "cv_ref_id": cv_id,
                    "label": label,
                    "bbox": [bx, by, bw, bh],
                    "center": [cx, cy],
                    "source": "cv",
                    "confidence": round(float(conf), 2),
                    "role": "canvas_hotspot",
                }
            )
            if len(kept) >= max_targets:
                break
        return kept

    def render_vision_grounding_image(
        self, raw_image: Image.Image, cv_candidates: List[Dict[str, Any]]
    ) -> str:
        """
        Render an annotated 1280x720 image for the Vision LLM grounding pass with
        16x9 A1..P9 cell labels, pixel ruler ticks, and CV candidate boxes (cv_01..).
        """
        w, h = self.viewport_width, self.viewport_height
        base = raw_image.resize((w, h)).convert("RGBA") if raw_image.size != (w, h) else raw_image.convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cols, rows = 16, 9
        cw, ch = w // cols, h // rows
        for c in range(1, cols):
            gx = c * cw
            draw.line([(gx, 0), (gx, h)], fill=(56, 189, 248, 48), width=1)
        for r in range(1, rows):
            gy = r * ch
            draw.line([(0, gy), (w, gy)], fill=(56, 189, 248, 48), width=1)

        for r in range(rows):
            for c in range(cols):
                code = f"{chr(ord('A') + c)}{r + 1}"
                tx, ty = c * cw + 3, r * ch + 2
                draw.rectangle([(tx - 1, ty - 1), (tx + 16, ty + 10)], fill=(2, 6, 23, 130))
                draw.text((tx, ty), code, fill=(125, 249, 255, 210))

        # Draw CV candidate boxes in amber so Vision LLM can semantically label them
        for cand in cv_candidates:
            bx, by, bw, bh = cand["bbox"]
            ref = cand["cv_ref_id"]
            draw.rectangle([(bx, by), (bx + bw, by + bh)], outline=(251, 191, 36, 215), width=1)
            ly = max(2, by - 12)
            draw.rectangle([(bx, ly), (bx + 34, ly + 11)], fill=(2, 6, 23, 190))
            draw.text((bx + 2, ly + 1), ref, fill=(253, 224, 71, 255))

        combined = Image.alpha_composite(base, overlay).convert("RGB")
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=78)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def resolve_vision_detections(
        self,
        vision_resp: VisionGroundingResponse,
        cv_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert Vision LLM detections into viewport-consistent target dicts:
        - Snaps to exact CV contour when cv_ref_id matches or center is within 24px
        - Cross-validates coordinates against 16x9 grid_cell to eliminate 0..1000 scale drift!
        """
        cv_by_id = {c["cv_ref_id"].lower(): c for c in cv_candidates}
        resolved: List[Dict[str, Any]] = []
        w, h = self.viewport_width, self.viewport_height

        for obj in vision_resp.targets[:20]:
            label = (obj.label or "Interactive game target").strip()[:64]
            conf = round(float(obj.confidence), 2)

            # 1. Exact CV reference match -> sensor fusion (Vision label + CV pixel bbox)
            if obj.cv_ref_id and obj.cv_ref_id.strip().lower() in cv_by_id:
                cv_item = cv_by_id[obj.cv_ref_id.strip().lower()]
                resolved.append(
                    {
                        "label": label,
                        "bbox": list(cv_item["bbox"]),
                        "center": list(cv_item["center"]),
                        "source": "vision",
                        "confidence": min(0.98, max(conf, cv_item["confidence"] + 0.08)),
                        "role": "canvas_object",
                    }
                )
                continue

            # 2. Resolve cell anchor if provided
            cell_anchor: Optional[Tuple[int, int]] = None
            if obj.grid_cell:
                try:
                    cell_anchor = macro_cell_to_coords(obj.grid_cell, width=w, height=h)
                except ValueError:
                    cell_anchor = None

            cx: Optional[int] = None
            cy: Optional[int] = None
            bw, bh = 32, 32

            if obj.center and len(obj.center) >= 2:
                cx, cy = int(obj.center[0]), int(obj.center[1])
            elif obj.bbox and len(obj.bbox) >= 4:
                bx, by, bw_in, bh_in = [int(v) for v in obj.bbox[:4]]
                bw = max(12, min(320, bw_in))
                bh = max(12, min(240, bh_in))
                cx = bx + bw // 2
                cy = by + bh // 2

            if obj.bbox and len(obj.bbox) >= 4:
                bw = max(12, min(320, int(obj.bbox[2])))
                bh = max(12, min(240, int(obj.bbox[3])))

            # Guard against 0..1000 normalized coordinates using grid_cell cross-check!
            if cx is not None and cy is not None and cell_anchor is not None:
                raw_dist = math.hypot(cx - cell_anchor[0], cy - cell_anchor[1])
                norm_cx = int(round((cx / 1000.0) * w))
                norm_cy = int(round((cy / 1000.0) * h))
                norm_dist = math.hypot(norm_cx - cell_anchor[0], norm_cy - cell_anchor[1])
                if raw_dist > 95 and norm_dist < 75:
                    cx, cy = norm_cx, norm_cy
                    bw = max(12, int(round((bw / 1000.0) * w)))
                    bh = max(12, int(round((bh / 1000.0) * h)))
                elif raw_dist > 130:
                    # Trust the visual cell label directly if pixel estimate drifted far outside the cell
                    cx, cy = cell_anchor

            if (cx is None or cy is None) and cell_anchor is not None:
                cx, cy = cell_anchor

            if cx is None or cy is None:
                continue

            cx = max(4, min(w - 5, cx))
            cy = max(4, min(h - 5, cy))

            # Snap to nearby CV contour if within 24px for pixel-exact centering
            nearest_cv = min(
                cv_candidates,
                key=lambda c: math.hypot(cx - c["center"][0], cy - c["center"][1]),
                default=None,
            )
            if nearest_cv and math.hypot(cx - nearest_cv["center"][0], cy - nearest_cv["center"][1]) <= 24:
                cx, cy = nearest_cv["center"]
                bbox = list(nearest_cv["bbox"])
                conf = min(0.97, max(conf, nearest_cv["confidence"] + 0.06))
            else:
                bx = max(0, min(w - bw, cx - bw // 2))
                by = max(0, min(h - bh, cy - bh // 2))
                bbox = [bx, by, bw, bh]

            resolved.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "center": [cx, cy],
                    "source": "vision",
                    "confidence": conf,
                    "role": "canvas_object",
                }
            )

        self._cached_vision_targets = resolved
        self._cached_vision_version = self.screenshot_version
        return resolved

    def build_interaction_map(
        self,
        dom_targets: List[Dict[str, Any]],
        cv_targets: List[Dict[str, Any]],
        vision_targets: Optional[List[Dict[str, Any]]],
        layout_meta: Dict[str, Any],
        regenerated: bool,
        regeneration_reason: str,
        max_total_targets: int = 95,
    ) -> InteractionMap:
        """
        Fuse DOM, Vision-grounded, and CV targets into a single deduplicated InteractionMap.
        Priority: DOM (confidence ~1.0) > Vision (LLM + CV fused) > CV hotspots.
        """
        now = time.time()
        ver = self.screenshot_version
        active_vision = vision_targets if vision_targets is not None else self._cached_vision_targets

        merged_raw: List[Dict[str, Any]] = []

        # 1. DOM & Accessibility targets (highest priority & exact live coordinates)
        for d in dom_targets:
            cx, cy = int(d["x"]), int(d["y"])
            bw, bh = int(d.get("w", 28)), int(d.get("h", 24))
            bx = int(d.get("bx", max(0, cx - bw // 2)))
            by = int(d.get("by", max(0, cy - bh // 2)))
            merged_raw.append(
                {
                    "label": d.get("label") or d.get("tag", "button"),
                    "bbox": [bx, by, bw, bh],
                    "center": [cx, cy],
                    "source": "dom",
                    "confidence": float(d.get("confidence", 1.0)),
                    "role": d.get("role") or d.get("tag", "button"),
                    "dom_selector_hint": d.get("selector_hint"),
                }
            )

        # 2. Vision LLM targets (skip if overlapping an existing DOM target)
        for v in active_vision:
            vcx, vcy = v["center"]
            if any(math.hypot(vcx - m["center"][0], vcy - m["center"][1]) < 20 for m in merged_raw):
                continue
            merged_raw.append(v)

        # 3. Remaining Computer Vision hotspots (skip if overlapping DOM or Vision targets)
        for c in cv_targets:
            ccx, ccy = c["center"]
            if any(math.hypot(ccx - m["center"][0], ccy - m["center"][1]) < 24 for m in merged_raw):
                continue
            merged_raw.append(c)

        targets: List[InteractionTarget] = []
        for idx, item in enumerate(merged_raw[:max_total_targets], start=1):
            targets.append(
                InteractionTarget(
                    id=f"target_{idx:03d}",
                    label=str(item["label"])[:64],
                    bbox=[int(v) for v in item["bbox"][:4]],
                    center=[int(item["center"][0]), int(item["center"][1])],
                    source=item["source"],
                    confidence=round(float(item["confidence"]), 2),
                    screenshot_version=ver,
                    timestamp=round(now, 3),
                    role=item.get("role"),
                    dom_selector_hint=item.get("dom_selector_hint"),
                )
            )

        imap = InteractionMap(
            screenshot_version=ver,
            timestamp=round(now, 3),
            viewport=[
                int(layout_meta.get("viewport_w", self.viewport_width)),
                int(layout_meta.get("viewport_h", self.viewport_height)),
            ],
            device_pixel_ratio=round(float(layout_meta.get("dpr", 1.0)), 2),
            has_canvas_or_webgl=bool(layout_meta.get("has_canvas", False)),
            regenerated=regenerated,
            regeneration_reason=regeneration_reason,
            targets=targets,
        )
        self.current_map = imap
        return imap

    def lookup_target(self, target_id: Optional[str]) -> Optional[InteractionTarget]:
        if not target_id:
            return None
        norm = target_id.strip().lower()
        for t in self.current_map.targets:
            if t.id.lower() == norm:
                return t
        # Allow numeric suffix fallback e.g. 'target_2' -> 'target_002'
        if norm.startswith("target_") and norm[7:].isdigit():
            padded = f"target_{int(norm[7:]):03d}"
            for t in self.current_map.targets:
                if t.id.lower() == padded:
                    return t
        return None

    def annotate_screenshot_for_llm(
        self,
        raw_image: Image.Image,
        interaction_map: InteractionMap,
        mouse_pos: Tuple[int, int],
        show_grid: bool = True,
    ) -> str:
        """
        Overlay InteractionMap target boxes & IDs (without covering target centers!)
        plus optional 16x9 A1..P9 reference grid and live CURSOR (x,y) crosshair.
        """
        w, h = self.viewport_width, self.viewport_height
        annotated = raw_image.resize((w, h)).convert("RGBA") if raw_image.size != (w, h) else raw_image.convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if show_grid:
            cols, rows = 16, 9
            cw, ch = w // cols, h // rows
            for c in range(1, cols):
                gx = c * cw
                draw.line([(gx, 0), (gx, h)], fill=(56, 189, 248, 42), width=1)
                # Explicit 1280px X-axis ruler tick at top edge to prevent 0..1000 coordinate drift
                tick_str = str(gx)
                tw = len(tick_str) * 6 + 3
                draw.rectangle([(gx - tw // 2, 0), (gx + tw // 2, 10)], fill=(2, 6, 23, 185))
                draw.text((gx - tw // 2 + 2, 0), tick_str, fill=(253, 224, 71, 235))
            for r in range(1, rows):
                gy = r * ch
                draw.line([(0, gy), (w, gy)], fill=(56, 189, 248, 42), width=1)
                # Explicit 720px Y-axis ruler tick at left edge to prevent 0..1000 coordinate drift
                tick_str = str(gy)
                tw = len(tick_str) * 6 + 4
                draw.rectangle([(0, gy - 5), (tw, gy + 6)], fill=(2, 6, 23, 185))
                draw.text((2, gy - 5), tick_str, fill=(253, 224, 71, 235))
            for r in range(rows):
                for c in range(cols):
                    code = f"{chr(ord('A') + c)}{r + 1}"
                    tx, ty = c * cw + 3, r * ch + (11 if r == 0 else 2)
                    if c == 0:
                        tx = 24
                    draw.rectangle([(tx - 1, ty - 1), (tx + 16, ty + 10)], fill=(2, 6, 23, 125))
                    draw.text((tx, ty), code, fill=(125, 249, 255, 195))

        # Draw target bounding boxes & compact ID badges outside centers so small game icons stay clear
        source_palette = {
            "dom": ((16, 185, 129, 220), (6, 78, 59, 225)),      # Emerald
            "vision": ((139, 92, 246, 225), (46, 16, 101, 225)), # Violet
            "cv": ((245, 158, 11, 215), (69, 26, 3, 220)),       # Amber
        }

        for t in interaction_map.targets[:95]:
            bx, by, bw, bh = t.bbox
            outline_col, bg_col = source_palette.get(t.source, source_palette["dom"])
            draw.rectangle([(bx, by), (bx + bw, by + bh)], outline=outline_col, width=2 if t.source == "dom" else 1)

            short_tag = f"#{t.id.split('_')[-1]}"  # e.g. '#001' -> compact 28px badge!
            lx = max(2, min(w - 32, bx))
            ly = by - 12 if by >= 14 else min(h - 13, by + bh + 1)
            draw.rectangle([(lx, ly), (lx + 28, ly + 11)], fill=bg_col, outline=outline_col)
            draw.text((lx + 2, ly + 1), short_tag, fill=(255, 255, 255, 255))

        # Draw prominent live mouse crosshair & cell badge (avoids 1280px vs 0..1000 confusion in VLM weights)
        mx, my = mouse_pos
        draw.line([(mx, 0), (mx, h)], fill=(244, 63, 94, 105), width=1)
        draw.line([(0, my), (w, my)], fill=(244, 63, 94, 105), width=1)
        draw.ellipse([(mx - 10, my - 10), (mx + 10, my + 10)], outline=(244, 63, 94, 255), width=2)

        cursor_cell = coords_to_macro_cell(mx, my, width=w, height=h)
        cursor_label = f"CURSOR [{cursor_cell}]"
        lx = mx - 88 if mx > 1160 else mx + 13
        ly = my + 12 if my < 32 else my - 20
        draw.rectangle([(lx - 3, ly - 2), (lx + 78, ly + 13)], fill=(15, 23, 42, 235), outline=(244, 63, 94, 235))
        draw.text((lx, ly), cursor_label, fill=(254, 240, 138, 255))

        combined = Image.alpha_composite(annotated, overlay).convert("RGB")
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=78)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
