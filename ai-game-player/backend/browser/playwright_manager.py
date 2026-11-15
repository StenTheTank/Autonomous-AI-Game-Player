import asyncio
import base64
import io
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

# Persistent, zero-interference in-browser cursor & live (X, Y) HUD script
CURSOR_HUD_INIT_SCRIPT = """
(() => {
  if (window !== window.top) return;
  if (window.__agyCursorInitialized) return;
  window.__agyCursorInitialized = true;
  window.__agyMousePos = { x: 640, y: 360 };

  function ensureHud() {
    if (document.getElementById('__agy_cursor_root')) return;
    const root = document.createElement('div');
    root.id = '__agy_cursor_root';
    root.style.cssText = 'position:fixed;inset:0;pointer-events:none !important;z-index:2147483647;overflow:hidden;';

    const topBadge = document.createElement('div');
    topBadge.id = '__agy_cursor_top_badge';
    topBadge.style.cssText = [
      'position:fixed',
      'top:6px',
      'left:50%',
      'transform:translateX(-50%)',
      'background:rgba(2,6,23,0.86)',
      'color:#38bdf8',
      'border:1px solid rgba(56,189,248,0.45)',
      'border-radius:999px',
      'padding:3px 12px',
      'font:700 12px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace',
      'pointer-events:none !important',
      'box-shadow:0 4px 14px rgba(0,0,0,0.45)',
      'letter-spacing:0.3px'
    ].join(';');
    topBadge.textContent = 'MOUSE: (640, 360)';

    const pointer = document.createElement('div');
    pointer.id = '__agy_cursor_pointer';
    pointer.style.cssText = [
      'position:fixed',
      'left:640px',
      'top:360px',
      'width:22px',
      'height:22px',
      'margin-left:-11px',
      'margin-top:-11px',
      'border:2px solid #f43f5e',
      'border-radius:50%',
      'background:rgba(244,63,94,0.16)',
      'pointer-events:none !important',
      'transition:transform 0.08s ease-out'
    ].join(';');

    const dot = document.createElement('div');
    dot.style.cssText = 'position:absolute;left:50%;top:50%;width:4px;height:4px;margin:-2px 0 0 -2px;background:#ffffff;border-radius:50%;';
    pointer.appendChild(dot);

    const tag = document.createElement('div');
    tag.id = '__agy_cursor_tag';
    tag.style.cssText = [
      'position:fixed',
      'left:654px',
      'top:348px',
      'background:rgba(15,23,42,0.92)',
      'color:#fde047',
      'border:1px solid rgba(250,204,21,0.55)',
      'border-radius:5px',
      'padding:1px 6px',
      'font:700 11px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace',
      'white-space:nowrap',
      'pointer-events:none !important'
    ].join(';');
    tag.textContent = '(640, 360)';

    root.appendChild(topBadge);
    root.appendChild(pointer);
    root.appendChild(tag);
    (document.documentElement || document.body).appendChild(root);
  }

  window.__agyUpdateCursor = (x, y, clicked) => {
    const rx = Math.round(x);
    const ry = Math.round(y);
    window.__agyMousePos = { x: rx, y: ry };
    ensureHud();
    const pointer = document.getElementById('__agy_cursor_pointer');
    const tag = document.getElementById('__agy_cursor_tag');
    const badge = document.getElementById('__agy_cursor_top_badge');
    if (pointer) {
      pointer.style.left = rx + 'px';
      pointer.style.top = ry + 'px';
      if (clicked) {
        pointer.style.transform = 'scale(1.65)';
        pointer.style.borderColor = '#34d399';
        setTimeout(() => {
          pointer.style.transform = 'scale(1)';
          pointer.style.borderColor = '#f43f5e';
        }, 220);
      }
    }
    if (tag) {
      const flipX = rx > 1130 ? rx - 82 : rx + 14;
      const flipY = ry < 28 ? ry + 14 : ry - 12;
      tag.style.left = flipX + 'px';
      tag.style.top = flipY + 'px';
      tag.textContent = `(${rx}, ${ry})`;
    }
    if (badge) {
      badge.textContent = `MOUSE: (${rx}, ${ry})  |  1280×720`;
    }
  };

  window.addEventListener('mousemove', (e) => {
    window.__agyUpdateCursor(e.clientX, e.clientY, false);
  }, { passive: true, capture: true });

  window.addEventListener('mousedown', (e) => {
    window.__agyUpdateCursor(e.clientX, e.clientY, true);
  }, { passive: true, capture: true });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureHud);
  } else {
    ensureHud();
  }
})();
"""


class BrowserManager:
    def __init__(self, headless: bool = False, max_action_duration_ms: int = 4000):
        self.headless = headless
        self.max_action_duration_ms = max_action_duration_ms
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._cdp_session: Any = None
        self._held_keys: Set[str] = set()
        self._mouse_down_buttons: Set[str] = set()
        self._last_mouse_pos: Tuple[int, int] = (VIEWPORT_WIDTH // 2, VIEWPORT_HEIGHT // 2)
        self._viewport_scale: Tuple[float, float] = (1.0, 1.0)
        self._prev_thumb: Optional[bytes] = None
        self._last_raw_image: Optional[Image.Image] = None
        self._initial_url: Optional[str] = None
        self._acknowledged_gates: Set[str] = set()

    @property
    def raw_image(self) -> Optional[Image.Image]:
        return self._last_raw_image

    @property
    def mouse_pos(self) -> Tuple[int, int]:
        return self._last_mouse_pos

    async def start(self) -> None:
        if self.playwright is None:
            self.playwright = await async_playwright().start()
        if self.browser is None or not self.browser.is_connected():
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT + 85}",
                ],
            )
        if self.context is None:
            self.context = await self.browser.new_context(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                device_scale_factor=1,
            )
        if self.page is None or self.page.is_closed():
            self.page = await self.context.new_page()

    async def ensure_alive(self) -> bool:
        """Verify page & browser are alive; recover if browser crashed."""
        if self.browser is None or not self.browser.is_connected() or self.page is None or self.page.is_closed():
            await self.start()
            if self._initial_url:
                await self.navigate(self._initial_url)
            return False
        return True

    async def navigate(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Blocked unsafe URL scheme: {parsed.scheme}")
        self._initial_url = url
        if not self.page or self.page.is_closed():
            await self.start()
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=25000)
            try:
                await self.page.wait_for_load_state("networkidle", timeout=4500)
            except Exception:
                pass
            await self._sync_cursor_hud(self._last_mouse_pos[0], self._last_mouse_pos[1], False)
            await self.page.mouse.move(self._last_mouse_pos[0], self._last_mouse_pos[1])
        except Exception as e:
            raise RuntimeError(f"Failed to load game URL '{url}': {e}")

    async def _sync_cursor_hud(self, x: int, y: int, clicked: bool = False) -> None:
        if not self.page or self.page.is_closed():
            return
        try:
            # Inject HUD only into the top-level document (never into sandboxed about:blank iframes)
            has_hud = await self.page.evaluate("() => Boolean(window.__agyCursorInitialized)")
            if not has_hud:
                await self.page.evaluate(CURSOR_HUD_INIT_SCRIPT)
            await self.page.evaluate(
                "([x, y, c]) => { if (window.__agyUpdateCursor) window.__agyUpdateCursor(x, y, c); }",
                [x, y, clicked],
            )
        except Exception:
            pass

    async def _pull_live_mouse_pos(self) -> Tuple[int, int]:
        """Read current cursor position from browser in case user moved the mouse manually."""
        if not self.page or self.page.is_closed():
            return self._last_mouse_pos
        try:
            pos = await self.page.evaluate("() => window.__agyMousePos || null")
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                self._last_mouse_pos = self._clamp_coords(pos["x"], pos["y"])
        except Exception:
            pass
        return self._last_mouse_pos

    async def _eval_dom_inspection(self, js_fn_body: str, args_list: List[Any]) -> Any:
        """
        Run DOM inspection via Chromium CDP Runtime.evaluate(includeCommandLineAPI=True)
        so native V8 `getEventListeners(el)` detects vanilla addEventListener('click', ...)
        on cards, tiles, and SVG board elements even when CSS cursor is 'auto'/'default'.
        Falls back cleanly to page.evaluate in mocked/non-CDP environments.
        """
        if self.context is not None and self.page is not None and not self.page.is_closed():
            try:
                if self._cdp_session is None:
                    self._cdp_session = await self.context.new_cdp_session(self.page)
                expr = f"({js_fn_body})({json.dumps(args_list)})"
                cdp_res = await self._cdp_session.send(
                    "Runtime.evaluate",
                    {
                        "expression": expr,
                        "includeCommandLineAPI": True,
                        "returnByValue": True,
                    },
                )
                val = (cdp_res or {}).get("result", {}).get("value")
                if isinstance(val, dict):
                    return val
            except Exception:
                self._cdp_session = None
        return await self.page.evaluate(js_fn_body, args_list)

    async def inspect_page_and_dom(self, limit: int = 95) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Inspect browser DOM, ARIA accessibility attributes, native V8 click listeners,
        interactive SVG board elements, delegated UI cards/slots, bounding boxes,
        Canvas/WebGL surfaces, devicePixelRatio, scroll offsets, and active modals (<15ms).
        Normalizes all coordinates into the canonical 1280x720 viewport space.
        """
        default_meta: Dict[str, Any] = {
            "url": self.page.url if self.page and not self.page.is_closed() else "",
            "viewport_w": VIEWPORT_WIDTH,
            "viewport_h": VIEWPORT_HEIGHT,
            "dpr": 1.0,
            "scroll_x": 0,
            "scroll_y": 0,
            "has_canvas": False,
            "has_bitmap_canvas": False,
            "has_svg": False,
            "canvas_rects": [],
            "dialog_signature": "",
            "disabled_labels": [],
        }
        if not self.page or self.page.is_closed():
            return [], default_meta
        try:
            payload = await self._eval_dom_inspection(
                """([maxItems, canonW, canonH]) => {
                    const hud = document.getElementById('__agy_cursor_root');
                    if (hud) hud.style.display = 'none';

                    const vw = window.innerWidth || canonW;
                    const vh = window.innerHeight || canonH;
                    const dpr = window.devicePixelRatio || 1;
                    const scaleX = canonW / Math.max(1, vw);
                    const scaleY = canonH / Math.max(1, vh);

                    function safeClassStr(el) {
                        if (!el) return '';
                        if (typeof el.className === 'string') return el.className;
                        return el.getAttribute ? (el.getAttribute('class') || '') : '';
                    }

                    function cleanClassHint(cls) {
                        if (!cls) return '';
                        return cls
                            .split(/\\s+/)
                            .map(tok => tok.replace(/[_-][a-zA-Z0-9]{5,}$/, ''))
                            .filter(tok => tok.length >= 2 && tok.length <= 28 && !/^(css|sc|tw|svelte)-/i.test(tok))
                            .slice(0, 2)
                            .join(' ');
                    }

                    // 1. Detect Canvas (bitmap) and SVG game surfaces
                    const canvasRects = [];
                    let hasBitmapCanvas = false;
                    let hasSvg = false;
                    for (const c of Array.from(document.querySelectorAll('canvas, svg'))) {
                        const cr = c.getBoundingClientRect();
                        if (cr.width >= 160 && cr.height >= 120) {
                            const tag = c.tagName.toLowerCase();
                            if (tag === 'canvas') hasBitmapCanvas = true;
                            if (tag === 'svg') hasSvg = true;
                            canvasRects.push([
                                Math.max(0, Math.round(cr.left * scaleX)),
                                Math.max(0, Math.round(cr.top * scaleY)),
                                Math.min(canonW, Math.round(cr.width * scaleX)),
                                Math.min(canonH, Math.round(cr.height * scaleY))
                            ]);
                        }
                    }

                    // 2. Detect open dialogs / modals for staleness invalidation
                    const dialogs = Array.from(
                        document.querySelectorAll('dialog[open], [role="dialog"], [aria-modal="true"]')
                    ).filter(d => {
                        const r = d.getBoundingClientRect();
                        return r.width > 40 && r.height > 40;
                    });
                    const dialogSig = dialogs
                        .map(d => `${d.tagName}:${(d.getAttribute('aria-label') || d.id || '').slice(0, 24)}:${Math.round(d.getBoundingClientRect().width)}`)
                        .join('|');

                    // 3. Collect Standard HTML/ARIA Controls + Pointer-Cursor & Interactive SVG Game Nodes
                    const primarySelectors = [
                        'button',
                        'a[href]',
                        'input[type="button"]',
                        'input[type="submit"]',
                        'input[type="checkbox"]',
                        'input[type="radio"]',
                        'select',
                        '[role="button"]',
                        '[role="tab"]',
                        '[role="menuitem"]',
                        '[role="option"]',
                        '[role="switch"]',
                        '[tabindex]:not([tabindex="-1"])',
                        '[onclick]'
                    ].join(',');

                    const seen = new Set();
                    const rawCandidates = [];
                    const rawDisabledLabels = [];

                    function collectSvgTextIndex(doc, offsetX, offsetY) {
                        const labels = [];
                        try {
                            for (const t of Array.from(doc.querySelectorAll('svg text, svg foreignObject'))) {
                                const txt = (t.textContent || '').replace(/\\s+/g, ' ').trim();
                                if (!txt || txt.length < 2 || txt.length > 36 || /^[+\\-]?\\d+$/.test(txt)) continue;
                                const tr = t.getBoundingClientRect();
                                if (tr.width < 4 || tr.height < 4) continue;
                                labels.push({
                                    text: txt,
                                    cx: Math.round((offsetX + tr.left + tr.width / 2) * scaleX),
                                    cy: Math.round((offsetY + tr.top + tr.height / 2) * scaleY)
                                });
                            }
                        } catch (_) {}
                        return labels;
                    }

                    function nearestSvgLabels(svgLabels, cx, cy, maxDist, maxCount) {
                        const hits = [];
                        for (let i = 0; i < svgLabels.length; i++) {
                            const lb = svgLabels[i];
                            const d = Math.hypot(lb.cx - cx, lb.cy - cy);
                            if (d <= maxDist) {
                                hits.push({ text: lb.text, d });
                            }
                        }
                        hits.sort((a, b) => a.d - b.d);
                        const uniq = [];
                        for (const h of hits) {
                            if (!uniq.includes(h.text)) {
                                uniq.push(h.text);
                                if (uniq.length >= maxCount) break;
                            }
                        }
                        return uniq;
                    }

                    function isElementDisabled(el, style) {
                        if (!el) return true;
                        if (el.disabled || (el.hasAttribute && el.hasAttribute('disabled'))) return true;
                        if (el.getAttribute && el.getAttribute('aria-disabled') === 'true') return true;
                        try {
                            if (el.matches && el.matches(':disabled')) return true;
                        } catch (_) {}
                        if (style && style.cursor === 'not-allowed') return true;
                        const cls = safeClassStr(el);
                        if (/(?:^|[-_\\s])(disabled|is-disabled|btn-disabled|cursor-not-allowed|unplayable)(?:$|[-_\\s])/i.test(cls)) {
                            return true;
                        }
                        return false;
                    }

                    function getEffectiveState(el) {
                        let effOpacity = 1.0;
                        let curr = el;
                        let depth = 0;
                        while (curr && curr !== document.body && curr !== document.documentElement && depth < 5) {
                            const cTag = (curr.tagName || '').toLowerCase();
                            if (cTag === 'svg' || cTag === 'canvas' || cTag === 'body' || cTag === 'html') break;
                            const cst = window.getComputedStyle(curr);
                            const op = parseFloat(cst.opacity || '1');
                            if (!Number.isNaN(op)) effOpacity *= op;
                            if (depth === 0 || cTag === 'button' || curr.getBoundingClientRect().width <= 380) {
                                if (isElementDisabled(curr, cst)) {
                                    return { disabled: true, opacity: effOpacity };
                                }
                            }
                            curr = curr.parentElement;
                            depth++;
                        }
                        return { disabled: false, opacity: effOpacity };
                    }

                    function accessibleLabel(el, tag, w, h, cx, cy, svgLabels) {
                        const ariaLabel = el.getAttribute && el.getAttribute('aria-label');
                        if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim().slice(0, 62);
                        const labelledBy = el.getAttribute && el.getAttribute('aria-labelledby');
                        if (labelledBy) {
                            const ref = document.getElementById(labelledBy.split(/\\s+/)[0]);
                            if (ref && (ref.innerText || ref.textContent)) return (ref.innerText || ref.textContent).trim().slice(0, 62);
                        }
                        const titleEl = el.querySelector && el.querySelector('title');
                        if (titleEl && titleEl.textContent && titleEl.textContent.trim()) {
                            return titleEl.textContent.trim().slice(0, 62);
                        }

                        const inSvg = Boolean(el.ownerSVGElement);
                        const visibleTxt = (
                            (!inSvg ? el.innerText : '') ||
                            el.textContent ||
                            el.value ||
                            ''
                        ).toString().replace(/\\s+/g, ' ').trim();

                        const testId = el.getAttribute && el.getAttribute('data-testid');
                        const dataHint = el.getAttribute && (
                            el.getAttribute('data-city') ||
                            el.getAttribute('data-location') ||
                            el.getAttribute('data-name') ||
                            el.getAttribute('data-label') ||
                            el.getAttribute('data-slot') ||
                            el.getAttribute('data-link') ||
                            el.getAttribute('title') ||
                            el.getAttribute('alt') ||
                            el.getAttribute('placeholder') ||
                            testId ||
                            el.getAttribute('data-action')
                        );

                        // Prefer visible human-readable button/card text FIRST so 'hand-card-2' never hides 'brewery'!
                        if (visibleTxt && visibleTxt.length >= 1 && visibleTxt.length <= 58) {
                            if (testId && /^hand-card-\\d+$/i.test(testId.trim())) {
                                return `${visibleTxt} [${testId.trim()}]`.slice(0, 62);
                            }
                            return visibleTxt;
                        }

                        if (dataHint && dataHint.trim()) return dataHint.trim().slice(0, 60);
                        if (el.id && !el.id.startsWith('__')) return el.id.trim().slice(0, 50);

                        // For SVG slots/links/nodes that carry no text inside their own shape, inspect compact parent <g> or nearby <svg> city text!
                        if (inSvg) {
                            let pg = el.parentElement;
                            let pDepth = 0;
                            while (pg && pg.tagName && pg.tagName.toLowerCase() === 'g' && pDepth < 3) {
                                const pr = pg.getBoundingClientRect();
                                if (pr.width > 185 || pr.height > 150) break;
                                const pTxt = (pg.textContent || '').replace(/\\s+/g, ' ').trim();
                                if (pTxt && pTxt.length >= 2 && pTxt.length <= 42) {
                                    return `${pTxt} (SVG ${tag === 'path' || tag === 'line' ? 'link' : 'slot'})`;
                                }
                                pg = pg.parentElement;
                                pDepth++;
                            }
                            const nearby = nearestSvgLabels(svgLabels, cx, cy, tag === 'path' || tag === 'line' ? 68 : 56, tag === 'path' || tag === 'line' ? 2 : 1);
                            if (nearby.length > 0) {
                                const kind = (tag === 'path' || tag === 'line') ? 'SVG link' : 'SVG slot';
                                return `${nearby.join(' - ')} (${kind})`;
                            }
                        }

                        const clsHint = cleanClassHint(safeClassStr(el));
                        if (clsHint) {
                            return inSvg ? `SVG ${clsHint}` : clsHint;
                        }
                        if (inSvg) {
                            return `Interactive SVG ${tag} (${Math.round(w)}x${Math.round(h)})`;
                        }
                        return visibleTxt ? visibleTxt.slice(0, 55) : '';
                    }

                    function resolveHitPoint(doc, el, r) {
                        let lx = r.left + r.width / 2;
                        let ly = r.top + r.height / 2;
                        if (lx >= 0 && lx < vw && ly >= 0 && ly < vh) {
                            const topEl = doc.elementFromPoint(lx, ly);
                            if (!topEl || el === topEl || el.contains(topEl) || topEl.contains(el)) {
                                return { ok: true, lx, ly };
                            }
                            // Allow sibling inside the same compact SVG group (e.g. decorative icon or glow inside clickable <g>)
                            if (el.ownerSVGElement && topEl.ownerSVGElement && el.parentElement && el.parentElement === topEl.parentElement) {
                                const pr = el.parentElement.getBoundingClientRect();
                                if (pr.width <= 160 && pr.height <= 140) {
                                    return { ok: true, lx, ly };
                                }
                            }
                        }
                        // For curved SVG paths, lines, or hollow rings where bbox center misses the stroke:
                        if (typeof el.getPointAtLength === 'function' && typeof el.getScreenCTM === 'function') {
                            try {
                                const len = el.getTotalLength();
                                const ctm = el.getScreenCTM();
                                if (len > 0 && ctm) {
                                    for (const frac of [0.5, 0.25, 0.75]) {
                                        const pt = el.getPointAtLength(len * frac);
                                        const tx = pt.x * ctm.a + pt.y * ctm.c + ctm.e;
                                        const ty = pt.x * ctm.b + pt.y * ctm.d + ctm.f;
                                        if (tx >= 0 && tx < vw && ty >= 0 && ty < vh) {
                                            const topPt = doc.elementFromPoint(tx, ty);
                                            if (topPt && (el === topPt || el.contains(topPt) || topPt.contains(el))) {
                                                return { ok: true, lx: tx, ly: ty };
                                            }
                                        }
                                    }
                                }
                            } catch (_) {}
                        }
                        return { ok: false, lx, ly };
                    }

                    function tryAddElement(doc, el, offsetX, offsetY, isPointerDiscovery, priority, svgLabels) {
                        if (!el || (el.id && el.id.startsWith('__agy_'))) return;
                        const tag = (el.tagName || '').toLowerCase();
                        if (!tag || tag === 'canvas' || tag === 'svg' || tag === 'body' || tag === 'html') return;

                        const style = window.getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') return;
                        const effState = getEffectiveState(el);
                        if (effState.opacity < 0.08) return;
                        const isControlOrCard = !el.ownerSVGElement && (
                            tag === 'button' ||
                            el.getAttribute('role') === 'button' ||
                            el.getAttribute('role') === 'option' ||
                            Boolean(el.getAttribute && el.getAttribute('data-testid')) ||
                            interactiveClsRe.test(safeClassStr(el))
                        );
                        if (
                            style.pointerEvents === 'none' ||
                            effState.disabled ||
                            effState.opacity < 0.12 ||
                            (isControlOrCard && effState.opacity < 0.52)
                        ) {
                            if (isControlOrCard) {
                                const dr = el.getBoundingClientRect();
                                if (dr.width >= 10 && dr.height >= 10 && dr.width <= 320 && dr.top >= 0 && dr.bottom <= vh + 20) {
                                    const dlbl = accessibleLabel(el, tag, dr.width, dr.height, 0, 0, [])
                                        .replace(/\s*\[[^\]]*\]\s*$/, '')
                                        .trim();
                                    if (dlbl && dlbl.length >= 2 && dlbl.length <= 42 && !rawDisabledLabels.includes(dlbl)) {
                                        rawDisabledLabels.push(dlbl);
                                    }
                                }
                            }
                            return;
                        }

                        const r = el.getBoundingClientRect();
                        if (r.width < 8 || r.height < 8) return;
                        if (r.width > vw * 0.45 && r.height > vh * 0.45) return; // skip huge board/ocean background rects

                        const hit = resolveHitPoint(doc, el, r);
                        if (!hit.ok) return;

                        const cx = Math.round((offsetX + hit.lx) * scaleX);
                        const cy = Math.round((offsetY + hit.ly) * scaleY);
                        if (cx < 2 || cx >= canonW - 2 || cy < 2 || cy >= canonH - 2) return;

                        const key = `${Math.round(cx / 9)}:${Math.round(cy / 9)}`;
                        if (seen.has(key)) return;

                        const role = (el.getAttribute && el.getAttribute('role')) || (el.ownerSVGElement ? `svg_${tag}` : tag);
                        const label = accessibleLabel(el, tag, r.width, r.height, cx, cy, svgLabels);
                        if (!label && !isPointerDiscovery && !el.ownerSVGElement && tag !== 'button' && role !== 'button' && priority < 2) return;

                        seen.add(key);

                        const bx = Math.max(0, Math.round((offsetX + r.left) * scaleX));
                        const by = Math.max(0, Math.round((offsetY + r.top) * scaleY));
                        const bw = Math.max(8, Math.round(r.width * scaleX));
                        const bh = Math.max(8, Math.round(r.height * scaleY));

                        rawCandidates.push({
                            priority: priority || 1,
                            tag,
                            role,
                            label: (label || role).slice(0, 62),
                            x: cx,
                            y: cy,
                            bx,
                            by,
                            w: bw,
                            h: bh,
                            bbox: [bx, by, bw, bh],
                            confidence: isPointerDiscovery ? 0.97 : 1.0
                        });
                    }

                    const hasNativeListenersApi = typeof getEventListeners === 'function';

                    function hasDirectClickBinding(el) {
                        if (!el) return false;
                        if (typeof el.onclick === 'function' || typeof el.__click === 'function' || (el.hasAttribute && el.hasAttribute('onclick'))) return true;
                        // 1. Native Chromium V8 event listener table (catches vanilla addEventListener, Svelte, Solid, Angular, Lit, D3)
                        if (hasNativeListenersApi) {
                            try {
                                const evs = getEventListeners(el);
                                if (evs && (
                                    (evs.click && evs.click.length > 0) ||
                                    (evs.pointerdown && evs.pointerdown.length > 0) ||
                                    (evs.mousedown && evs.mousedown.length > 0) ||
                                    (evs.pointerup && evs.pointerup.length > 0) ||
                                    (evs.mouseup && evs.mouseup.length > 0)
                                )) {
                                    return true;
                                }
                            } catch (_) {}
                        }
                        // 2. React 17/18/19, Vue 3, Solid attached listener props on DOM/SVG nodes
                        try {
                            const keys = Object.keys(el);
                            for (let i = 0; i < keys.length; i++) {
                                const k = keys[i];
                                if (k.startsWith('__reactProps$')) {
                                    const p = el[k];
                                    if (p && !p.disabled && (typeof p.onClick === 'function' || typeof p.onPointerDown === 'function' || typeof p.onMouseDown === 'function')) {
                                        return true;
                                    }
                                } else if (k === '_vei') {
                                    const vei = el._vei;
                                    if (vei && (vei.onClick || vei.onPointerdown || vei.onMousedown)) {
                                        return true;
                                    }
                                } else if (k === '$$click') {
                                    return true;
                                }
                            }
                        } catch (_) {}
                        return false;
                    }

                    function ancestorHasDelegatedClick(el) {
                        let p = el && el.parentElement;
                        let depth = 0;
                        // Climb up to 12 levels so <svg>-root or deep <g> board delegation is detected
                        while (p && p !== document.documentElement && depth < 12) {
                            if (hasDirectClickBinding(p)) return true;
                            p = p.parentElement;
                            depth++;
                        }
                        return false;
                    }

                    function hasHighlightStroke(el, st) {
                        if (!el.ownerSVGElement) return false;
                        const stroke = (st.stroke || el.getAttribute('stroke') || '').toLowerCase();
                        if (!stroke || stroke === 'none' || stroke === 'transparent') return false;
                        const sw = parseFloat(st.strokeWidth || el.getAttribute('stroke-width') || '0');
                        if (sw < 1.5) return false;
                        // Match bright gold/amber/yellow/emerald/cyan/sky highlight strokes used for playable board slots
                        const m = stroke.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                        if (m) {
                            const rr = parseInt(m[1], 10), gg = parseInt(m[2], 10), bb = parseInt(m[3], 10);
                            const maxC = Math.max(rr, gg, bb), minC = Math.min(rr, gg, bb);
                            if (maxC >= 175 && (maxC - minC) >= 55) return true;
                        }
                        return /^#(f[0-9a-f]{5}|e[ab0-9][0-9a-f]{4}|3[48][0-9a-f]{4}|10b981|22c55e|eab308|f59e0b|fbbf24|38bdf8)/i.test(stroke);
                    }

                    const interactiveClsRe = /(?:^|[-_\\s])(card|hand-card|tile|slot|spot|choice|option|selectable|clickable|valid|available|playable|highlight|highlighted|glow|pulse|action-btn|chip|pill|industry-slot|link-slot)(?:$|[-_\\s])/i;
                    const highPriorityClsRe = /(?:^|[-_\\s])(highlight|highlighted|valid|available|playable|selectable|glow|pulse|card|hand-card)(?:$|[-_\\s])/i;

                    function collectFromDoc(doc, offsetX, offsetY) {
                        const svgLabels = collectSvgTextIndex(doc, offsetX, offsetY);

                        // Pass A: Explicit interactive HTML/ARIA controls (enabled <button>s ALWAYS get top priority 5!)
                        let primaryNodes = [];
                        try {
                            primaryNodes = Array.from(doc.querySelectorAll(primarySelectors));
                        } catch (_) {}
                        for (const el of primaryNodes) {
                            const tag = (el.tagName || '').toLowerCase();
                            const role = (el.getAttribute && el.getAttribute('role')) || '';
                            let prio = 3;
                            if (!el.ownerSVGElement && (tag === 'button' || tag === 'input' || role === 'button' || role === 'option')) {
                                prio = 5;
                            } else if (highPriorityClsRe.test(safeClassStr(el))) {
                                prio = 4;
                            }
                            tryAddElement(doc, el, offsetX, offsetY, false, prio, svgLabels);
                        }

                        // Pass B: Interactive SVG & custom DOM game elements (native addEventListener, React/Vue/Svelte props, cursor:pointer, highlighted SVG slots, or delegated cards/slots)
                        let candidates = [];
                        try {
                            candidates = Array.from(
                                doc.querySelectorAll('svg g, svg rect, svg circle, svg ellipse, svg path, svg line, svg polygon, div, span, li, article, label, img')
                            );
                        } catch (_) {}

                        const bodyCursor = doc.body ? window.getComputedStyle(doc.body).cursor : 'auto';
                        for (let el of candidates) {
                            const st = window.getComputedStyle(el);
                            const hasBoundClick = hasDirectClickBinding(el);
                            const hasPointerCursor = st.cursor === 'pointer' && bodyCursor !== 'pointer';
                            const clsStr = safeClassStr(el);

                            let isDelegatedCardOrSlot = false;
                            let isHighlightedBoardNode = false;

                            if (!hasBoundClick && !hasPointerCursor) {
                                const hasDataAttr = Boolean(el.dataset && Object.keys(el.dataset).length > 0);
                                const matchesCls = interactiveClsRe.test(clsStr);
                                if (matchesCls || (hasDataAttr && ancestorHasDelegatedClick(el))) {
                                    const r = el.getBoundingClientRect();
                                    const hasNestedBlocks = el.querySelectorAll && el.querySelectorAll('div, ul, section, table').length > 2;
                                    if (!hasNestedBlocks && r.width >= 10 && r.width <= 290 && r.height >= 10 && r.height <= 150) {
                                        isDelegatedCardOrSlot = true;
                                    }
                                } else if (el.ownerSVGElement && hasHighlightStroke(el, st) && ancestorHasDelegatedClick(el)) {
                                    const r = el.getBoundingClientRect();
                                    if (r.width >= 12 && r.width <= 170 && r.height >= 12 && r.height <= 140) {
                                        isHighlightedBoardNode = true;
                                    }
                                } else if (!el.ownerSVGElement && ancestorHasDelegatedClick(el)) {
                                    // Styled boxed pill/card inside a delegated click container (e.g. hand card div)
                                    const r = el.getBoundingClientRect();
                                    const hasBorder = parseFloat(st.borderTopWidth || '0') >= 1 && st.borderTopStyle !== 'none' && !st.borderTopColor.includes('0, 0, 0, 0');
                                    const txt = (el.innerText || '').trim();
                                    const isLeafBox = !el.querySelector || el.querySelectorAll('div, section, ul, table').length === 0;
                                    if (hasBorder && isLeafBox && txt.length >= 1 && txt.length <= 48 && r.width >= 24 && r.width <= 260 && r.height >= 18 && r.height <= 110) {
                                        isDelegatedCardOrSlot = true;
                                    }
                                }
                            }

                            if (!hasBoundClick && !hasPointerCursor && !isDelegatedCardOrSlot && !isHighlightedBoardNode) continue;

                            let curr = el;
                            // If inner <span> inside a compact clickable button or card container, bind to the card/button container itself!
                            if (!curr.ownerSVGElement && curr.tagName.toLowerCase() === 'span' && curr.parentElement) {
                                const p = curr.parentElement;
                                const pr = p.getBoundingClientRect();
                                if (pr.width >= 20 && pr.width <= 260 && pr.height >= 16 && pr.height <= 110) {
                                    const pTag = p.tagName.toLowerCase();
                                    if (pTag === 'button' || hasDirectClickBinding(p) || interactiveClsRe.test(safeClassStr(p))) {
                                        curr = p;
                                    }
                                }
                            }

                            if (!hasBoundClick && hasPointerCursor) {
                                // Climb to compact interactive group root (<g cursor="pointer"> or <div cursor="pointer">)
                                while (curr.parentElement && curr.parentElement !== doc.body) {
                                    const pTag = curr.parentElement.tagName.toLowerCase();
                                    if (pTag === 'svg' || pTag === 'body' || pTag === 'html') break;
                                    if (hasDirectClickBinding(curr.parentElement)) {
                                        curr = curr.parentElement;
                                        break;
                                    }
                                    const pStyle = window.getComputedStyle(curr.parentElement);
                                    if (pStyle.cursor !== 'pointer') break;
                                    const pr = curr.parentElement.getBoundingClientRect();
                                    if (pr.width > 180 || pr.height > 150) {
                                        break;
                                    }
                                    curr = curr.parentElement;
                                }

                                // Ignore if cursor:pointer was merely inherited from a huge full-board wrapper
                                if (curr.parentElement && !hasDirectClickBinding(curr)) {
                                    const pStyle = window.getComputedStyle(curr.parentElement);
                                    const pr = curr.parentElement.getBoundingClientRect();
                                    if (pStyle.cursor === 'pointer' && pr.width > 420 && pr.height > 320) {
                                        const inlineCursor = (curr.style && curr.style.cursor) || (curr.getAttribute && curr.getAttribute('cursor'));
                                        if (inlineCursor !== 'pointer') continue;
                                    }
                                }
                            }

                            // Priority: 4 = HTML cards/options, 3 = highlighted/bound/pointer SVG board slots, 1 = passive delegated SVG shapes
                            let candPrio = 2;
                            if (!curr.ownerSVGElement) {
                                candPrio = 4;
                            } else if (isHighlightedBoardNode || hasBoundClick || hasPointerCursor || highPriorityClsRe.test(safeClassStr(curr))) {
                                candPrio = 3;
                            } else {
                                candPrio = 1;
                            }
                            tryAddElement(doc, curr, offsetX, offsetY, !hasBoundClick, candPrio, svgLabels);
                        }
                    }

                    collectFromDoc(document, 0, 0);

                    // Inspect accessible same-origin iframes
                    for (const frameEl of Array.from(document.querySelectorAll('iframe'))) {
                        try {
                            if (frameEl.contentDocument) {
                                const fr = frameEl.getBoundingClientRect();
                                collectFromDoc(frameEl.contentDocument, fr.left, fr.top);
                            }
                        } catch (_) {}
                    }

                    // Prioritize interactive SVG/board slots, cards, and active controls so they are never truncated!
                    rawCandidates.sort((a, b) => (b.priority - a.priority) || (a.y - b.y) || (a.x - b.x));
                    const results = rawCandidates.slice(0, maxItems).map((item, idx) => ({
                        id: idx + 1,
                        tag: item.tag,
                        role: item.role,
                        label: item.label,
                        x: item.x,
                        y: item.y,
                        bx: item.bx,
                        by: item.by,
                        w: item.w,
                        h: item.h,
                        bbox: item.bbox,
                        confidence: item.confidence
                    }));

                    const activeBaseLabels = new Set(
                        results.map(r => (r.label || '').replace(/\s*\[[^\]]*\]\s*$/, '').trim().toLowerCase())
                    );
                    const disabledLabels = rawDisabledLabels
                        .filter(lbl => !activeBaseLabels.has(lbl.toLowerCase()))
                        .slice(0, 14);

                    if (hud) hud.style.display = 'block';

                    return {
                        items: results,
                        meta: {
                            url: location.href,
                            viewport_w: vw,
                            viewport_h: vh,
                            dpr,
                            scroll_x: Math.round(window.scrollX || 0),
                            scroll_y: Math.round(window.scrollY || 0),
                            has_canvas: canvasRects.length > 0,
                            has_bitmap_canvas: hasBitmapCanvas,
                            has_svg: hasSvg,
                            canvas_rects: canvasRects,
                            dialog_signature: dialogSig,
                            disabled_labels: disabledLabels
                        }
                    };
                }""",
                [limit, VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
            )
            if isinstance(payload, dict):
                meta = payload.get("meta") or default_meta
                vw = max(1, int(meta.get("viewport_w", VIEWPORT_WIDTH)))
                vh = max(1, int(meta.get("viewport_h", VIEWPORT_HEIGHT)))
                self._viewport_scale = (vw / float(VIEWPORT_WIDTH), vh / float(VIEWPORT_HEIGHT))
                items = payload.get("items")
                return (items if isinstance(items, list) else []), meta
            return [], default_meta
        except Exception:
            return [], default_meta

    async def scan_interactive_elements(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper returning interactive DOM target dicts."""
        items, _ = await self.inspect_page_and_dom(limit=limit)
        return items

    async def reverify_dom_target(
        self, expected_label: str, expected_center: List[int]
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Re-verify a DOM target's live coordinates immediately prior to clicking so
        stale coordinates are never clicked after layout shifts or modal popups.
        """
        if not self.page or self.page.is_closed():
            return None, "browser_closed"
        try:
            res = await self.page.evaluate(
                """([expLabel, expCx, expCy, canonW, canonH]) => {
                    const hud = document.getElementById('__agy_cursor_root');
                    if (hud) hud.style.display = 'none';
                    const vw = window.innerWidth || canonW;
                    const vh = window.innerHeight || canonH;
                    const cssX = expCx * (vw / canonW);
                    const cssY = expCy * (vh / canonH);

                    const normExp = (expLabel || '').replace(/\[[^\]]*\]\s*$/, '').toLowerCase().trim();
                    const elAtPoint = document.elementFromPoint(cssX, cssY);

                    function restoreHudAndReturn(payload) {
                        if (hud) hud.style.display = 'block';
                        return payload;
                    }

                    if (elAtPoint) {
                        const interactive = elAtPoint.closest('button, a[href], input, select, [role="button"], [role="tab"], [role="menuitem"], [role="option"], [data-testid], [tabindex]:not([tabindex="-1"])');
                        if (interactive && !interactive.disabled && interactive.getAttribute('aria-disabled') !== 'true') {
                            let r = interactive.getBoundingClientRect();
                            if (r.width >= 8 && r.height >= 8 && r.width < vw * 0.48 && r.height < vh * 0.48) {
                                const iTxt = (interactive.innerText || interactive.getAttribute('aria-label') || interactive.value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                                const labelConsistent = !normExp || !iTxt || iTxt === normExp || iTxt.includes(normExp) || normExp.includes(iTxt);
                                if (labelConsistent) {
                                    if (r.bottom > vh - 4 || r.top < 4) {
                                        try { interactive.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (_) {}
                                        r = interactive.getBoundingClientRect();
                                    }
                                    const liveCx = Math.round(((r.left + r.width / 2) / vw) * canonW);
                                    const liveCy = Math.round(((r.top + r.height / 2) / vh) * canonH);
                                    if (Math.hypot(liveCx - expCx, liveCy - expCy) <= 95) {
                                        return restoreHudAndReturn({ ok: true, x: liveCx, y: liveCy, status: 'verified_live' });
                                    }
                                }
                            }
                        } else if (elAtPoint.ownerSVGElement) {
                            const r = elAtPoint.getBoundingClientRect();
                            const eTag = (elAtPoint.tagName || '').toLowerCase();
                            if (eTag !== 'svg' && r.width >= 8 && r.height >= 8 && r.width < vw * 0.42 && r.height < vh * 0.42) {
                                if (Math.hypot((r.left + r.width / 2) * (canonW / vw) - expCx, (r.top + r.height / 2) * (canonH / vh) - expCy) <= 50) {
                                    return restoreHudAndReturn({ ok: true, x: expCx, y: expCy, status: 'verified_svg_live' });
                                }
                            }
                        } else {
                            // Accept non-SVG custom element ONLY if it is a compact leaf/card (NEVER a multi-button sidebar wrapper <div>!)
                            const r = elAtPoint.getBoundingClientRect();
                            const childButtons = elAtPoint.querySelectorAll ? elAtPoint.querySelectorAll('button, input, a[href], [role="button"]').length : 0;
                            if (childButtons === 0 && r.width >= 8 && r.width <= 260 && r.height >= 8 && r.height <= 120) {
                                const liveCx = Math.round(((r.left + r.width / 2) / vw) * canonW);
                                const liveCy = Math.round(((r.top + r.height / 2) / vh) * canonH);
                                if (Math.hypot(liveCx - expCx, liveCy - expCy) <= 65) {
                                    return restoreHudAndReturn({ ok: true, x: liveCx, y: liveCy, status: 'verified_compact_live' });
                                }
                            }
                        }
                    }

                    // Search by label in case button shifted slightly or sits inside a scrollable panel
                    if (normExp) {
                        const candidates = Array.from(document.querySelectorAll('button, a[href], [role="button"], [role="option"], [data-testid], input[type="button"], input[type="submit"]'));
                        let bestCand = null;
                        let bestDist = Infinity;
                        for (const c of candidates) {
                            if (c.disabled || c.getAttribute('aria-disabled') === 'true') continue;
                            const txt = (c.innerText || c.getAttribute('aria-label') || c.value || c.getAttribute('data-testid') || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                            if (txt && (txt === normExp || txt.includes(normExp) || normExp.includes(txt))) {
                                const r = c.getBoundingClientRect();
                                if (r.width >= 10 && r.height >= 10 && r.width <= 380) {
                                    const lx = Math.round(((r.left + r.width / 2) / vw) * canonW);
                                    const ly = Math.round(((r.top + r.height / 2) / vh) * canonH);
                                    const dist = Math.hypot(lx - expCx, ly - expCy);
                                    if (dist <= 210 && dist < bestDist) {
                                        bestDist = dist;
                                        bestCand = c;
                                    }
                                }
                            }
                        }
                        if (bestCand) {
                            try { bestCand.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (_) {}
                            const br = bestCand.getBoundingClientRect();
                            const lx = Math.round(((br.left + br.width / 2) / vw) * canonW);
                            const ly = Math.round(((br.top + br.height / 2) / vh) * canonH);
                            return restoreHudAndReturn({ ok: true, x: lx, y: ly, status: 'relocated_live' });
                        }
                    }
                    return restoreHudAndReturn({ ok: false, status: 'stale_or_occluded' });
                }""",
                [expected_label, int(expected_center[0]), int(expected_center[1]), VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
            )
            if isinstance(res, dict) and res.get("ok"):
                return self._clamp_coords(res["x"], res["y"]), str(res.get("status", "verified"))
            return None, str((res or {}).get("status", "stale_or_occluded"))
        except Exception:
            return (int(expected_center[0]), int(expected_center[1])), "fallback"

    async def detect_access_gate(self) -> Optional[str]:
        """Detect CAPTCHA or anti-bot challenge gates so the agent pauses instead of bypassing."""
        if not self.page or self.page.is_closed():
            return None
        try:
            # Check child frames (e.g. Cloudflare Turnstile inside closed Shadow DOM, reCAPTCHA, hCaptcha)
            for frame in self.page.frames:
                furl = (frame.url or "").lower()
                if any(
                    marker in furl
                    for marker in ("challenges.cloudflare.com", "google.com/recaptcha", "hcaptcha.com")
                ):
                    if furl not in self._acknowledged_gates:
                        self._acknowledged_gates.add(furl)
                        return f"Anti-bot / CAPTCHA verification widget detected ({furl.split('?')[0][:70]})"

            gate = await self.page.evaluate(
                """() => {
                    const title = (document.title || '').toLowerCase();
                    if (title.includes('just a moment') || title.includes('attention required') || title.includes('security check')) {
                        return 'Anti-bot / security challenge page detected (' + document.title + ')';
                    }
                    const selectors = [
                        '.cf-turnstile',
                        '.g-recaptcha',
                        '.h-captcha'
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.offsetParent !== null) {
                            return 'CAPTCHA / human verification widget detected (' + sel + ')';
                        }
                    }
                    return null;
                }"""
            )
            if gate and gate not in self._acknowledged_gates:
                self._acknowledged_gates.add(gate)
                return gate
            return None
        except Exception:
            return None

    def _clamp_coords(self, x: Optional[int], y: Optional[int]) -> Tuple[int, int]:
        cx = max(0, min(VIEWPORT_WIDTH - 1, int(x if x is not None else self._last_mouse_pos[0])))
        cy = max(0, min(VIEWPORT_HEIGHT - 1, int(y if y is not None else self._last_mouse_pos[1])))
        return cx, cy

    def _clamp_duration(self, duration_ms: Optional[int], default_ms: int = 150) -> int:
        val = default_ms if duration_ms is None else int(duration_ms)
        return max(10, min(self.max_action_duration_ms, val))

    async def capture_observation(
        self,
        add_grid_for_llm: bool = True,
        interactive_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, str, float]:
        """
        Capture current game state.
        Returns (ui_screenshot_b64, llm_screenshot_b64, visual_delta_pct).
        """
        if not self.page or self.page.is_closed():
            return "", "", 0.0

        await self._pull_live_mouse_pos()

        # Briefly hide in-DOM cursor overlay for 1 frame during raw screenshot so UI & Pillow overlays stay crisp
        try:
            await self.page.evaluate(
                "() => { const r = document.getElementById('__agy_cursor_root'); if (r) r.style.visibility = 'hidden'; }"
            )
        except Exception:
            pass

        screenshot_bytes = await self.page.screenshot(type="jpeg", quality=78)

        try:
            await self.page.evaluate(
                "() => { const r = document.getElementById('__agy_cursor_root'); if (r) r.style.visibility = 'visible'; }"
            )
        except Exception:
            pass

        ui_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        if img.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
            img = img.resize((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
        self._last_raw_image = img.copy()

        # Compute visual delta % against previous frame using a compact 64x36 grayscale thumbnail
        thumb = img.resize((64, 36)).convert("L").tobytes()
        delta_pct = 0.0
        if self._prev_thumb is not None and len(self._prev_thumb) == len(thumb):
            changed = sum(1 for a, b in zip(self._prev_thumb, thumb) if abs(a - b) > 12)
            delta_pct = round((changed / len(thumb)) * 100.0, 2)
        else:
            delta_pct = 100.0
        self._prev_thumb = thumb

        if not add_grid_for_llm:
            return ui_b64, ui_b64, delta_pct

        # Render clean 16x9 Named Macro Cell Grid (A1..P9, each 80x80px) + [#ID] button tags + CURSOR (x,y)
        annotated = img.convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = annotated.size
        cols, rows = 16, 9
        cell_w, cell_h = w // cols, h // rows  # 80x80 px

        for c in range(1, cols):
            gx = c * cell_w
            draw.line([(gx, 0), (gx, h)], fill=(56, 189, 248, 52), width=1)
        for r in range(1, rows):
            gy = r * cell_h
            draw.line([(0, gy), (w, gy)], fill=(56, 189, 248, 52), width=1)

        for r in range(rows):
            for c in range(cols):
                code = f"{chr(ord('A') + c)}{r + 1}"
                tx = c * cell_w + 3
                ty = r * cell_h + 2
                draw.rectangle([(tx - 1, ty - 1), (tx + 16, ty + 10)], fill=(2, 6, 23, 135))
                draw.text((tx, ty), code, fill=(125, 249, 255, 215))

        # Highlight genuine HTML buttons with rectangular magenta [#ID] badges (never circular dots!)
        if interactive_elements:
            for item in interactive_elements[:20]:
                ex, ey = item["x"], item["y"]
                tag_txt = f"#{item['id']}"
                bx = max(2, min(w - 28, ex - 12))
                by = max(2, min(h - 14, ey - 6))
                draw.rectangle([(bx, by), (bx + 24, by + 12)], fill=(157, 23, 77, 225), outline=(244, 114, 182, 255))
                draw.text((bx + 3, by + 1), tag_txt, fill=(255, 255, 255, 255))

        # Draw prominent full-screen crosshair & exact coordinate badge at current MOUSE (mx, my)
        mx, my = self._last_mouse_pos
        draw.line([(mx, 0), (mx, h)], fill=(244, 63, 94, 105), width=1)
        draw.line([(0, my), (w, my)], fill=(244, 63, 94, 105), width=1)
        draw.ellipse([(mx - 10, my - 10), (mx + 10, my + 10)], outline=(244, 63, 94, 255), width=2)

        cursor_label = f"CURSOR ({mx},{my})"
        lx = mx - 112 if mx > 1130 else mx + 13
        ly = my + 12 if my < 32 else my - 20
        draw.rectangle([(lx - 3, ly - 2), (lx + 106, ly + 13)], fill=(15, 23, 42, 235), outline=(244, 63, 94, 235))
        draw.text((lx, ly), cursor_label, fill=(254, 240, 138, 255))

        combined = Image.alpha_composite(annotated, overlay).convert("RGB")
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=78)
        llm_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return ui_b64, llm_b64, delta_pct

    async def refresh_raw_frame(self) -> None:
        """Capture a fresh raw frame right before 4x Zoom-Snap so stale pre-thinking frames are never used."""
        if not self.page or self.page.is_closed():
            return
        try:
            await self.page.evaluate(
                "() => { const r = document.getElementById('__agy_cursor_root'); if (r) r.style.visibility = 'hidden'; }"
            )
            raw_bytes = await self.page.screenshot(type="jpeg", quality=78)
            await self.page.evaluate(
                "() => { const r = document.getElementById('__agy_cursor_root'); if (r) r.style.visibility = 'visible'; }"
            )
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            if img.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
                img = img.resize((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            self._last_raw_image = img
        except Exception:
            pass

    async def resolve_smart_click_coordinates(
        self,
        x: int,
        y: int,
        hint_text: str = "",
        is_gemini: bool = False,
    ) -> Tuple[int, int, Optional[str]]:
        """
        Verify fallback `click_coordinates(x, y)` against the live DOM/SVG before clicking:
        1. If (x, y) already hits a genuine compact interactive element, keep (x, y).
        2. If (x, y) hits dead background / bare <svg> (e.g., Gemini emitting 0..1000 normalized
           coords like (285, 465) for Coalbrookdale at (365, 335)):
           - Snap to any live interactive element or <svg> city slot matching city/card names in `hint_text`.
           - Test 0..1000 -> 1280x720 rescaled coordinates `(round(x*1.28), round(y*0.72))` for live interactive targets!
        """
        if not self.page or self.page.is_closed():
            return x, y, None
        try:
            res = await self._eval_dom_inspection(
                """([rawX, rawY, hint, isGem, canonW, canonH]) => {
                    const hud = document.getElementById('__agy_cursor_root');
                    if (hud) hud.style.display = 'none';

                    const vw = window.innerWidth || canonW;
                    const vh = window.innerHeight || canonH;
                    const toCssX = (vx) => vx * (vw / canonW);
                    const toCssY = (vy) => vy * (vh / canonH);
                    const toCanonX = (cx) => Math.round((cx / Math.max(1, vw)) * canonW);
                    const toCanonY = (cy) => Math.round((cy / Math.max(1, vh)) * canonH);

                    const hasListenersApi = typeof getEventListeners === 'function';

                    function isDisabledEl(el, st) {
                        if (!el) return true;
                        let curr = el;
                        let effOp = 1.0;
                        let d = 0;
                        while (curr && curr !== document.body && curr !== document.documentElement && d < 5) {
                            const cTag = (curr.tagName || '').toLowerCase();
                            if (cTag === 'svg' || cTag === 'canvas' || cTag === 'body' || cTag === 'html') break;
                            const cst = d === 0 && st ? st : window.getComputedStyle(curr);
                            const op = parseFloat(cst.opacity || '1');
                            if (!Number.isNaN(op)) effOp *= op;
                            if (d === 0 || cTag === 'button' || curr.getBoundingClientRect().width <= 380) {
                                if (curr.disabled || (curr.hasAttribute && curr.hasAttribute('disabled'))) return true;
                                if (curr.getAttribute && curr.getAttribute('aria-disabled') === 'true') return true;
                                try {
                                    if (curr.matches && curr.matches(':disabled')) return true;
                                } catch (_) {}
                                if (cst && cst.cursor === 'not-allowed') return true;
                                const cls = typeof curr.className === 'string' ? curr.className : (curr.getAttribute ? (curr.getAttribute('class') || '') : '');
                                if (/(?:^|[-_\\s])(disabled|is-disabled|btn-disabled|cursor-not-allowed|unplayable)(?:$|[-_\\s])/i.test(cls)) return true;
                            }
                            curr = curr.parentElement;
                            d++;
                        }
                        if (effOp < 0.50) return true;
                        return false;
                    }

                    function isCompactInteractive(el) {
                        if (!el) return false;
                        let curr = el;
                        let depth = 0;
                        while (curr && curr !== document.body && curr !== document.documentElement && depth < 5) {
                            const tag = (curr.tagName || '').toLowerCase();
                            if (tag === 'svg' || tag === 'canvas' || tag === 'body' || tag === 'html') break;
                            const r = curr.getBoundingClientRect();
                            if (r.width >= 8 && r.height >= 8 && r.width < vw * 0.40 && r.height < vh * 0.40) {
                                const st = window.getComputedStyle(curr);
                                if (isDisabledEl(curr, st)) return null;
                                if ((tag === 'button' || curr.getAttribute('role') === 'button') && parseFloat(st.opacity || '1') < 0.38) return null;
                                if (tag === 'button' || tag === 'a' || tag === 'input' || tag === 'select' ||
                                    curr.getAttribute('role') === 'button' || st.cursor === 'pointer' ||
                                    typeof curr.onclick === 'function') {
                                    return curr;
                                }
                                if (hasListenersApi) {
                                    try {
                                        const evs = getEventListeners(curr);
                                        if (evs && ((evs.click && evs.click.length) || (evs.pointerdown && evs.pointerdown.length) || (evs.mousedown && evs.mousedown.length))) {
                                            return curr;
                                        }
                                    } catch (_) {}
                                }
                                try {
                                    for (const k of Object.keys(curr)) {
                                        if (k.startsWith('__reactProps$') && curr[k] && (curr[k].onClick || curr[k].onPointerDown || curr[k].onMouseDown)) {
                                            return curr;
                                        }
                                    }
                                } catch (_) {}
                            }
                            curr = curr.parentElement;
                            depth++;
                        }
                        return null;
                    }

                    // 1. Check if raw (x, y) already hits a genuine compact interactive element
                    const elRaw = document.elementFromPoint(toCssX(rawX), toCssY(rawY));
                    const hitRaw = isCompactInteractive(elRaw);
                    if (hitRaw) {
                        if (hud) hud.style.display = 'block';
                        return { x: rawX, y: rawY, note: null };
                    }

                    // Compute 0..1000 -> 1280x720 candidate if coordinates fit 0..1000 range
                    const in1000 = rawX >= 0 && rawX <= 1000 && rawY >= 0 && rawY <= 1000;
                    const normX = in1000 ? Math.min(canonW - 2, Math.round((rawX / 1000.0) * canonW)) : rawX;
                    const normY = in1000 ? Math.min(canonH - 2, Math.round((rawY / 1000.0) * canonH)) : rawY;

                    // 2. Extract meaningful location/card/button keywords from hint_text (e.g. 'Coalbrookdale', 'Wolverhampton', 'brewery')
                    const stopWords = new Set([
                        'click', 'clicking', 'clicked', 'slot', 'highlighted', 'highlight', 'valid', 'target',
                        'card', 'hand', 'board', 'city', 'tile', 'link', 'canal', 'rail', 'build', 'network',
                        'develop', 'loan', 'sell', 'scout', 'action', 'choose', 'select', 'place', 'industry',
                        'near', 'left', 'right', 'bottom', 'center', 'screen', 'area', 'location', 'spot',
                        'with', 'from', 'into', 'this', 'that', 'then', 'your', 'turn', 'game', 'icon', 'button', 'option'
                    ]);
                    const uniqTokens = Array.from(new Set(
                        (hint || '')
                            .replace(/[^a-zA-Z0-9'\\-\\s]/g, ' ')
                            .split(/\\s+/)
                            .map(w => w.trim().toLowerCase())
                            .filter(w => w.length >= 4 && !stopWords.has(w) && !/^target\\d*$/.test(w))
                    ));

                    if (uniqTokens.length > 0) {
                        let bestMatch = null;
                        // Search enabled <button>s, [role="button"], SVG city <text> labels & compact interactive elements!
                        const labelNodes = Array.from(document.querySelectorAll('button, [role="button"], input[type="button"], svg text, div, span'));
                        for (const node of labelNodes) {
                            const nst = window.getComputedStyle(node);
                            if (nst.display === 'none' || nst.visibility === 'hidden' || isDisabledEl(node, nst)) continue;

                            const inSvg = Boolean(node.ownerSVGElement);
                            const nTag = (node.tagName || '').toLowerCase();
                            const isRealButton = !inSvg && (nTag === 'button' || nTag === 'input' || node.getAttribute('role') === 'button');
                            if (isRealButton && parseFloat(nst.opacity || '1') < 0.50) continue;
                            // NEVER snap onto static game-log or header <div>/<span> prose—require genuine interactivity!
                            if (!inSvg && !isRealButton && !isCompactInteractive(node)) continue;
                            if (!inSvg && !isRealButton && node.children && node.children.length > 3) continue;

                            const txt = ((inSvg ? node.textContent : node.innerText) || '').replace(/\\s+/g, ' ').trim();
                            if (!txt || txt.length < 3 || txt.length > 56) continue;
                            const lowTxt = txt.toLowerCase();

                            let matchCount = 0;
                            let exactMatch = false;
                            for (const tok of uniqTokens) {
                                if (lowTxt === tok) {
                                    matchCount++;
                                    exactMatch = true;
                                } else if (lowTxt.includes(tok)) {
                                    matchCount++;
                                }
                            }
                            if (matchCount === 0) continue;

                            const nr = node.getBoundingClientRect();
                            if (nr.width < 4 || nr.height < 4 || nr.width > 340 || nr.height > 140) continue;

                            let targetX = toCanonX(nr.left + nr.width / 2);
                            let targetY = toCanonY(nr.top + nr.height / 2);
                            let slotBoost = 0;

                            if (isRealButton) {
                                // Strongly prefer actual <button> option controls (e.g. 'Wolverhampton: coal L1 — £7')
                                // over re-clicking an already-selected hand card <span> ('Wolverhampton')!
                                slotBoost += 95;
                            } else if (!inSvg) {
                                const pCls = node.parentElement && typeof node.parentElement.className === 'string' ? node.parentElement.className : '';
                                if (/(selected|active)/i.test(pCls)) {
                                    slotBoost -= 55;
                                }
                            } else if (inSvg && node.parentElement) {
                                let cityG = node.parentElement;
                                if (cityG.parentElement && cityG.getBoundingClientRect().width < 32) {
                                    cityG = cityG.parentElement;
                                }
                                const gr = cityG.getBoundingClientRect();
                                if (gr.width >= 16 && gr.width <= 190 && gr.height >= 16 && gr.height <= 150) {
                                    const subShapes = Array.from(cityG.querySelectorAll('rect, circle, polygon, path, g'));
                                    let chosenRect = null;
                                    for (const sh of subShapes) {
                                        const sr = sh.getBoundingClientRect();
                                        if (sr.width < 12 || sr.height < 12 || sr.width > 120 || sr.height > 100) continue;
                                        const sst = window.getComputedStyle(sh);
                                        if (sst.pointerEvents === 'none' || sst.display === 'none') continue;
                                        if (isCompactInteractive(sh) || sst.cursor === 'pointer') {
                                            chosenRect = sr;
                                            slotBoost = 60;
                                            break;
                                        }
                                        if (!chosenRect && sh.tagName.toLowerCase() === 'rect') {
                                            chosenRect = sr;
                                            slotBoost = 25;
                                        }
                                    }
                                    const useR = chosenRect || gr;
                                    targetX = toCanonX(useR.left + useR.width / 2);
                                    targetY = toCanonY(useR.top + useR.height / 2);
                                }
                            }

                            const distRaw = Math.hypot(targetX - rawX, targetY - rawY);
                            const distNorm = Math.hypot(targetX - normX, targetY - normY);
                            const bestDist = Math.min(distRaw, distNorm);
                            if (bestDist <= 260) {
                                const score = (exactMatch ? 95 : 75) + (matchCount - 1) * 45 + slotBoost - bestDist * 0.35;
                                if (!bestMatch || score > bestMatch.score) {
                                    bestMatch = { x: targetX, y: targetY, label: txt, score };
                                }
                            }
                        }

                        if (bestMatch) {
                            if (hud) hud.style.display = 'block';
                            return {
                                x: bestMatch.x,
                                y: bestMatch.y,
                                note: `Smart-snapped (${rawX},${rawY}) -> "${bestMatch.label.slice(0, 34)}" (${bestMatch.x},${bestMatch.y})`
                            };
                        }
                    }

                    // 3. Gemini 0..1000 normalized coordinate probe: if raw (x, y) missed, check if (normX, normY) hits a live interactive element!
                    if (in1000 && (Math.abs(normX - rawX) > 20 || Math.abs(normY - rawY) > 20)) {
                        const offsets = [[0, 0], [0, -14], [0, 14], [-14, 0], [14, 0], [0, -26]];
                        for (const [ox, oy] of offsets) {
                            const px = Math.max(2, Math.min(canonW - 3, normX + ox));
                            const py = Math.max(2, Math.min(canonH - 3, normY + oy));
                            const elN = document.elementFromPoint(toCssX(px), toCssY(py));
                            const hitN = isCompactInteractive(elN);
                            if (hitN) {
                                const hr = hitN.getBoundingClientRect();
                                const sx = toCanonX(hr.left + hr.width / 2);
                                const sy = toCanonY(hr.top + hr.height / 2);
                                if (hud) hud.style.display = 'block';
                                return {
                                    x: sx,
                                    y: sy,
                                    note: `Rescaled 0..1000 (${rawX},${rawY}) -> live target (${sx},${sy})`
                                };
                            }
                        }
                        // If model is Gemini and raw coords hit dead area while 0..1000 rescaled point is on the canvas/svg board, use normX/normY!
                        if (isGem) {
                            const elCenterNorm = document.elementFromPoint(toCssX(normX), toCssY(normY));
                            if (elCenterNorm && (elCenterNorm.tagName.toLowerCase() === 'canvas' || elCenterNorm.tagName.toLowerCase() === 'svg' || elCenterNorm.ownerSVGElement)) {
                                if (hud) hud.style.display = 'block';
                                return {
                                    x: normX,
                                    y: normY,
                                    note: `Rescaled Gemini 0..1000 (${rawX},${rawY}) -> (${normX},${normY})`
                                };
                            }
                        }
                    }

                    if (hud) hud.style.display = 'block';
                    return { x: rawX, y: rawY, note: null };
                }""",
                [int(x), int(y), hint_text or "", bool(is_gemini), VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
            )
            if isinstance(res, dict) and "x" in res and "y" in res:
                cx, cy = self._clamp_coords(res["x"], res["y"])
                return cx, cy, res.get("note")
        except Exception:
            pass
        return x, y, None

    async def should_zoom_refine(self, x: int, y: int, element_id: Optional[int] = None) -> bool:
        """
        Return True when fallback `click_coordinates` lands on a bitmap <canvas> OR on bare/background <svg>
        (meaning it missed a compact DOM/SVG button/slot and needs 4x visual magnification to lock onto the target).
        """
        if element_id is not None or self._last_raw_image is None:
            return False
        if not self.page or self.page.is_closed():
            return False
        try:
            css_x, css_y = self._to_css_coords(x, y)
            needs_zoom = await self.page.evaluate(
                """([x, y]) => {
                    const r = document.getElementById('__agy_cursor_root');
                    if (r) r.style.display = 'none';
                    const el = document.elementFromPoint(x, y);
                    if (r) r.style.display = 'block';
                    if (!el) return false;
                    const tag = el.tagName.toLowerCase();
                    if (tag === 'canvas' || tag === 'svg') return true;
                    if (el.ownerSVGElement) {
                        const br = el.getBoundingClientRect();
                        const st = window.getComputedStyle(el);
                        // Zoom-refine if point landed on a large background SVG map layer rather than a compact clickable node
                        if (st.cursor !== 'pointer' && (br.width > window.innerWidth * 0.38 || br.height > window.innerHeight * 0.38)) {
                            return true;
                        }
                    }
                    return false;
                }""",
                [css_x, css_y],
            )
            return bool(needs_zoom)
        except Exception:
            return False

    def build_zoom_crop_b64(self, coarse_x: int, coarse_y: int) -> Tuple[str, int, int]:
        """
        Crop a 240x180 game region around (coarse_x, coarse_y), magnify 4x to 960x720,
        and overlay a 12x9 Precision Grid ('A1'..'L9', each cell = 20x20 game pixels).
        Returns (zoom_jpeg_b64, crop_x0, crop_y0).
        """
        if self._last_raw_image is None:
            return "", 0, 0
        crop_w, crop_h = 240, 180
        x0 = max(0, min(VIEWPORT_WIDTH - crop_w, int(coarse_x) - crop_w // 2))
        y0 = max(0, min(VIEWPORT_HEIGHT - crop_h, int(coarse_y) - crop_h // 2))

        cropped = self._last_raw_image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        zoomed = cropped.resize((960, 720), Image.Resampling.BICUBIC).convert("RGBA")

        overlay = Image.new("RGBA", zoomed.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cols, rows = 12, 9
        zw, zh = 960 // cols, 720 // rows  # 80x80 zoomed pixels = 20x20 game pixels

        for c in range(1, cols):
            gx = c * zw
            draw.line([(gx, 0), (gx, 720)], fill=(56, 189, 248, 105), width=1)
        for r in range(1, rows):
            gy = r * zh
            draw.line([(0, gy), (960, gy)], fill=(56, 189, 248, 105), width=1)

        for r in range(rows):
            for c in range(cols):
                code = f"{chr(ord('A') + c)}{r + 1}"
                tx = c * zw + 4
                ty = r * zh + 3
                cx = c * zw + zw // 2
                cy = r * zh + zh // 2
                # Small center '+' anchor inside each 80x80 zoomed cell
                draw.line([(cx - 4, cy), (cx + 4, cy)], fill=(255, 255, 255, 110), width=1)
                draw.line([(cx, cy - 4), (cx, cy + 4)], fill=(255, 255, 255, 110), width=1)
                draw.rectangle([(tx - 2, ty - 1), (tx + 17, ty + 11)], fill=(2, 6, 23, 185))
                draw.text((tx, ty), code, fill=(253, 224, 71, 245))

        combined = Image.alpha_composite(zoomed, overlay).convert("RGB")
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), x0, y0

    def resolve_zoom_cell(self, crop_x0: int, crop_y0: int, cell: str, sub_pos: str = "center") -> Tuple[int, int]:
        """Map zoomed 12x9 cell ('A1'..'L9') + 9-point sub_pos back to exact 1280x720 pixel coordinates."""
        cleaned = (cell or "F5").strip().upper()
        if len(cleaned) < 2 or not cleaned[0].isalpha() or not cleaned[1:].isdigit():
            return self._clamp_coords(crop_x0 + 120, crop_y0 + 90)
        col_idx = max(0, min(11, ord(cleaned[0]) - ord("A")))
        row_idx = max(0, min(8, int(cleaned[1:]) - 1))

        offsets = {
            "center": (10, 10),
            "top": (10, 4),
            "bottom": (10, 16),
            "left": (4, 10),
            "right": (16, 10),
            "top_left": (4, 4),
            "top_right": (16, 4),
            "bottom_left": (4, 16),
            "bottom_right": (16, 16),
        }
        ox, oy = offsets.get((sub_pos or "center").lower().strip(), (10, 10))
        fx = crop_x0 + col_idx * 20 + ox
        fy = crop_y0 + row_idx * 20 + oy
        return self._clamp_coords(fx, fy)

    async def get_screenshot_base64(self) -> str:
        ui_b64, _, _ = await self.capture_observation(add_grid_for_llm=False)
        return ui_b64

    async def _describe_element_at(self, x: int, y: int) -> str:
        """Inspect what DOM/Canvas element sits at viewport pixel (x, y)."""
        if not self.page or self.page.is_closed():
            return ""
        try:
            desc = await self.page.evaluate(
                """([x, y]) => {
                    const r = document.getElementById('__agy_cursor_root');
                    if (r) r.style.display = 'none';
                    const el = document.elementFromPoint(x, y);
                    if (r) r.style.display = 'block';
                    if (!el) return 'empty';
                    const tag = el.tagName.toLowerCase();
                    const cls = typeof el.className === 'string'
                        ? el.className
                        : (el.getAttribute ? (el.getAttribute('class') || '') : '');
                    const txt = (el.innerText || (el.getAttribute && el.getAttribute('aria-label')) || el.id || cls || '')
                        .toString().replace(/\\s+/g, ' ').trim().slice(0, 36);
                    return txt ? `<${tag} "${txt}">` : `<${tag}>`;
                }""",
                [x, y],
            )
            return desc or ""
        except Exception:
            return ""

    def _to_css_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Convert canonical 1280x720 coordinates to live CSS viewport coordinates."""
        sx, sy = self._viewport_scale
        return round(x * sx, 1), round(y * sy, 1)

    async def _perform_realistic_click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> str:
        """
        Execute a game-compatible pointer click:
        1. Smooth mouse move onto (x, y) in live CSS viewport space so Canvas/WebGL hover updates.
        2. Brief hover settle (~55ms).
        3. Mouse down -> hold (~75ms) -> mouse up so multi-frame game loops register the press.
        """
        css_x, css_y = self._to_css_coords(x, y)
        mx = int(css_x) if css_x == int(css_x) else css_x
        my = int(css_y) if css_y == int(css_y) else css_y

        await self.page.mouse.move(mx, my, steps=6)
        await self._sync_cursor_hud(int(round(css_x)), int(round(css_y)), clicked=False)
        await asyncio.sleep(0.055)

        target_desc = await self._describe_element_at(int(round(css_x)), int(round(css_y)))

        for i in range(clicks):
            await self.page.mouse.down(button=button)
            await self._sync_cursor_hud(int(round(css_x)), int(round(css_y)), clicked=True)
            await asyncio.sleep(0.075)
            await self.page.mouse.up(button=button)
            if i + 1 < clicks:
                await asyncio.sleep(0.06)

        return target_desc

    async def execute_action(self, action: Dict[str, Any], default_duration_ms: int = 180) -> str:
        """Execute validated structured action safely and return human-readable execution status."""
        if not self.page or self.page.is_closed():
            return "Skipped (browser page closed)"

        action_type = action.get("type")

        try:
            if action_type == "key_press":
                key = action.get("key")
                duration = action.get("duration_ms")
                if key:
                    if duration and duration > 40:
                        clamped = self._clamp_duration(duration, default_duration_ms)
                        await self.page.keyboard.down(key)
                        await asyncio.sleep(clamped / 1000.0)
                        await self.page.keyboard.up(key)
                        return f"Pressed {key} ({clamped}ms)"
                    else:
                        await self.page.keyboard.press(key)
                        return f"Tapped {key}"

            elif action_type == "key_hold":
                key = action.get("key")
                if key:
                    clamped = self._clamp_duration(action.get("duration_ms"), default_duration_ms)
                    self._held_keys.add(key)
                    await self.page.keyboard.down(key)
                    await asyncio.sleep(clamped / 1000.0)
                    await self.page.keyboard.up(key)
                    self._held_keys.discard(key)
                    return f"Held {key} for {clamped}ms"

            elif action_type == "key_release":
                key = action.get("key")
                if key:
                    await self.page.keyboard.up(key)
                    self._held_keys.discard(key)
                    return f"Released {key}"

            elif action_type == "key_combo":
                keys = action.get("keys") or []
                clamped = self._clamp_duration(action.get("duration_ms"), default_duration_ms)
                for k in keys:
                    self._held_keys.add(k)
                    await self.page.keyboard.down(k)
                await asyncio.sleep(clamped / 1000.0)
                for k in reversed(keys):
                    await self.page.keyboard.up(k)
                    self._held_keys.discard(k)
                return f"Combo {' + '.join(keys)} ({clamped}ms)"

            elif action_type in ("mouse_click", "click_target", "click_coordinates"):
                x, y = self._clamp_coords(action.get("x"), action.get("y"))
                button = action.get("button") or "left"
                clicks = int(action.get("clicks") or 1)
                self._last_mouse_pos = (x, y)
                hit_desc = await self._perform_realistic_click(x, y, button=button, clicks=clicks)
                tid = action.get("target_id")
                prefix = f"Clicked {tid} at ({x}, {y})" if tid else f"Clicked {button} ({clicks}x) at ({x}, {y})"
                suffix = f" -> hit {hit_desc}" if hit_desc else ""
                return f"{prefix}{suffix}"

            elif action_type in ("mouse_move", "hover_target"):
                x, y = self._clamp_coords(action.get("x"), action.get("y"))
                self._last_mouse_pos = (x, y)
                css_x, css_y = self._to_css_coords(x, y)
                mx = int(css_x) if css_x == int(css_x) else css_x
                my = int(css_y) if css_y == int(css_y) else css_y
                await self.page.mouse.move(mx, my, steps=6)
                await self._sync_cursor_hud(int(round(css_x)), int(round(css_y)), clicked=False)
                hit_desc = await self._describe_element_at(int(round(css_x)), int(round(css_y)))
                suffix = f" over {hit_desc}" if hit_desc else ""
                return f"Moved cursor to ({x}, {y}){suffix}"

            elif action_type == "mouse_down":
                if action.get("x") is not None and action.get("y") is not None:
                    x, y = self._clamp_coords(action.get("x"), action.get("y"))
                    self._last_mouse_pos = (x, y)
                    await self.page.mouse.move(x, y, steps=4)
                    await self._sync_cursor_hud(x, y, clicked=True)
                button = action.get("button") or "left"
                self._mouse_down_buttons.add(button)
                await self.page.mouse.down(button=button)
                return f"Held {button} mouse button down at {self._last_mouse_pos}"

            elif action_type == "mouse_up":
                if action.get("x") is not None and action.get("y") is not None:
                    x, y = self._clamp_coords(action.get("x"), action.get("y"))
                    self._last_mouse_pos = (x, y)
                    await self.page.mouse.move(x, y, steps=4)
                    await self._sync_cursor_hud(x, y, clicked=False)
                button = action.get("button") or "left"
                await self.page.mouse.up(button=button)
                self._mouse_down_buttons.discard(button)
                return f"Released {button} mouse button at {self._last_mouse_pos}"

            elif action_type == "mouse_drag":
                start_x, start_y = self._clamp_coords(
                    action.get("from_x", self._last_mouse_pos[0]),
                    action.get("from_y", self._last_mouse_pos[1]),
                )
                end_x, end_y = self._clamp_coords(action.get("x"), action.get("y"))
                button = action.get("button") or "left"
                await self.page.mouse.move(start_x, start_y, steps=5)
                await asyncio.sleep(0.05)
                await self.page.mouse.down(button=button)
                await asyncio.sleep(0.05)
                await self.page.mouse.move(end_x, end_y, steps=14)
                await asyncio.sleep(0.05)
                await self.page.mouse.up(button=button)
                self._last_mouse_pos = (end_x, end_y)
                await self._sync_cursor_hud(end_x, end_y, clicked=True)
                return f"Dragged ({start_x}, {start_y}) -> ({end_x}, {end_y})"

            elif action_type == "scroll":
                if action.get("x") is not None and action.get("y") is not None:
                    x, y = self._clamp_coords(action.get("x"), action.get("y"))
                    self._last_mouse_pos = (x, y)
                    await self.page.mouse.move(x, y, steps=4)
                    await self._sync_cursor_hud(x, y, clicked=False)
                sx = int(action.get("scroll_x") or 0)
                sy = int(action.get("scroll_y") or 0)
                await self.page.mouse.wheel(sx, sy)
                return f"Scrolled ({sx}, {sy})"

            elif action_type == "wait":
                clamped = self._clamp_duration(action.get("duration_ms"), 1000)
                await asyncio.sleep(clamped / 1000.0)
                return f"Waited {clamped}ms"

            elif action_type == "request_observation":
                return "Captured fresh observation"

            elif action_type == "declare_success":
                return "Agent declared objective completed"

            elif action_type == "declare_failure":
                return "Agent declared objective failed"

            return f"Executed {action_type}"
        except Exception as e:
            return f"Action error ({action_type}): {e}"

    async def release_all_inputs(self) -> None:
        """Safety release of any held keys or mouse buttons upon pause/stop."""
        if not self.page or self.page.is_closed():
            self._held_keys.clear()
            self._mouse_down_buttons.clear()
            return
        for key in list(self._held_keys):
            try:
                await self.page.keyboard.up(key)
            except Exception:
                pass
        self._held_keys.clear()
        for btn in list(self._mouse_down_buttons):
            try:
                await self.page.mouse.up(button=btn)
            except Exception:
                pass
        self._mouse_down_buttons.clear()

    async def stop(self) -> None:
        await self.release_all_inputs()
        self._cdp_session = None
        if self.page:
            try:
                await self.page.close()
            except Exception:
                pass
            self.page = None
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
