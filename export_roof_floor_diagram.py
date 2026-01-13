from __future__ import annotations

import math
from pathlib import Path

import pygame

import game


class _DummyApp:
    def __init__(self) -> None:
        # Force debug-highrise test world-gen so we can deterministically
        # fetch the single test building from the generated chunk.
        self.config = game.GameConfig(
            scale=2,
            fullscreen=False,
            show_grid=False,
            debug_highrise_test=True,
        )


def _fmt_len_tiles_px(tiles: float, *, tile_px: int) -> str:
    if abs(tiles - round(tiles)) < 1e-6:
        t = f"{int(round(tiles))}t"
    else:
        t = f"{tiles:.1f}t"
    px = int(round(float(tiles) * float(tile_px)))
    return f"{t} / {px}px"


def _draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
    color: pygame.Color,
    *,
    anchor: str = "topleft",
) -> pygame.Rect:
    return game.draw_text(surface, font, text, pos, color, anchor=anchor)


def _draw_dim_h(
    surface: pygame.Surface,
    font: pygame.font.Font,
    x1: int,
    x2: int,
    y: int,
    text: str,
    color: pygame.Color,
) -> None:
    x1 = int(x1)
    x2 = int(x2)
    y = int(y)
    if x2 < x1:
        x1, x2 = x2, x1
    pygame.draw.line(surface, color, (x1, y), (x2, y), 1)
    # Arrow heads.
    pygame.draw.line(surface, color, (x1, y), (x1 + 6, y - 3), 1)
    pygame.draw.line(surface, color, (x1, y), (x1 + 6, y + 3), 1)
    pygame.draw.line(surface, color, (x2, y), (x2 - 6, y - 3), 1)
    pygame.draw.line(surface, color, (x2, y), (x2 - 6, y + 3), 1)
    _draw_text(surface, font, text, (int((x1 + x2) // 2), int(y - 2)), color, anchor="midbottom")


def _draw_dim_v(
    surface: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y1: int,
    y2: int,
    text: str,
    color: pygame.Color,
) -> None:
    x = int(x)
    y1 = int(y1)
    y2 = int(y2)
    if y2 < y1:
        y1, y2 = y2, y1
    pygame.draw.line(surface, color, (x, y1), (x, y2), 1)
    # Arrow heads.
    pygame.draw.line(surface, color, (x, y1), (x - 3, y1 + 6), 1)
    pygame.draw.line(surface, color, (x, y1), (x + 3, y1 + 6), 1)
    pygame.draw.line(surface, color, (x, y2), (x - 3, y2 - 6), 1)
    pygame.draw.line(surface, color, (x, y2), (x + 3, y2 - 6), 1)
    _draw_text(surface, font, text, (int(x - 2), int((y1 + y2) // 2)), color, anchor="midright")


def export_diagram(*, out_path: Path) -> Path:
    pygame.init()
    pygame.font.init()
    font_path = game.pick_ui_font_path()
    font = pygame.font.Font(font_path, 22) if font_path else pygame.font.Font(None, 22)
    font_s = pygame.font.Font(font_path, 18) if font_path else pygame.font.Font(None, 18)

    app = _DummyApp()
    state = game.HardcoreSurvivalState(app, avatar=game.SurvivalAvatar())
    # Fixed seed for reproducible diagram output.
    world = game.HardcoreSurvivalState._World(state, seed=0x13579BDF)
    dchunk = getattr(world, "debug_hr_test_chunk", None)
    if not (isinstance(dchunk, tuple) and len(dchunk) == 2):
        raise RuntimeError("debug_hr_test_chunk missing; set debug_highrise_test=true")

    chunk = world.get_chunk(int(dchunk[0]), int(dchunk[1]))
    tests = list(getattr(world, "debug_hr_tests", []))
    if not tests:
        raise RuntimeError("debug_hr_tests empty")
    spec = tests[0]
    tx0 = int(spec.get("tx0", 0))
    ty0 = int(spec.get("ty0", 0))

    b = None
    for bb in getattr(chunk, "buildings", []):
        try:
            if int(bb[0]) == tx0 and int(bb[1]) == ty0:
                b = bb
                break
        except Exception:
            continue
    if b is None:
        b = getattr(chunk, "buildings", [None])[0]
    if b is None:
        raise RuntimeError("no building generated in debug chunk")

    _tx0, _ty0, w, h, roof_kind, floors = (int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5]))
    style, var = state._building_roof_style_var(int(roof_kind))
    face_h_px = int(state._building_face_height_px(style=int(style), w=int(w), h=int(h), var=int(var), floors=int(floors)))

    # Roof tile footprint.
    if int(style) == 6:
        rw = int(w)
        rh = int(h)
    else:
        rw = max(1, int(w) - 2)
        rh = max(1, int(h) - 2)

    tile_px = int(state.TILE_SIZE)  # 10px in world
    roof_full_px_h = int(rh) * int(tile_px)

    # High-rise: the "true roof" / roof-visible region excludes the facade band
    # (TILE_SIZE + face_h). The per-floor footprint must match that region.
    band_tiles = 0
    inner_h = int(h)
    cut_px = 0
    if int(style) == 6:
        band_tiles = int((int(tile_px) + max(0, int(face_h_px))) // int(tile_px))
        band_tiles = int(max(1, int(band_tiles)))
        inner_h = int(max(0, int(h) - int(band_tiles)))
        cut_px = int(min(int(roof_full_px_h), int(band_tiles) * int(tile_px)))
    elif int(style) == 1 and int(floors) > 1:
        cut_px = int(max(0, int(face_h_px) - 2))
        min_vis = 14
        max_cut = max(0, int(roof_full_px_h) - int(min_vis))
        cut_px = int(min(int(cut_px), int(max_cut)))

    roof_draw_px_h = int(max(0, int(roof_full_px_h) - int(cut_px)))
    roof_draw_tiles_h = float(roof_draw_px_h) / float(max(1, int(tile_px)))

    # Diagram scale (independent from world pixels; labels still use world px). 
    scale = 12  # px per tile in this diagram
    floor_h_tiles = int(inner_h)
    floor_rect = pygame.Rect(0, 0, int(w) * int(scale), int(floor_h_tiles) * int(scale))
    roof_draw_rect = pygame.Rect(
        0,
        0,
        int(rw) * int(scale),
        max(1, int(round(float(roof_draw_tiles_h) * float(scale)))),
    )

    margin = 30
    panel_gap = 60
    title_h = 36
    panel_w = max(int(floor_rect.w), int(roof_draw_rect.w)) + 2 * margin
    panel_h = max(int(floor_rect.h), int(roof_draw_rect.h)) + 2 * margin + 60
    canvas_w = panel_w * 2 + panel_gap
    canvas_h = title_h + panel_h + margin

    surf = pygame.Surface((int(canvas_w), int(canvas_h)), pygame.SRCALPHA)
    surf.fill((18, 18, 22))

    title = "高层测试：1F 范围 vs 屋顶(绘制范围，不含立面)"
    _draw_text(surf, font, title, (int(canvas_w // 2), 12), pygame.Color(240, 240, 240), anchor="midtop")
    sub = f"tile=10px  floors={floors}  roof_style={style} var={var}  face_h={face_h_px}px  band={band_tiles}t  cut={cut_px}px"
    _draw_text(surf, font_s, sub, (int(canvas_w // 2), 32), pygame.Color(180, 180, 190), anchor="midtop")

    def draw_panel(
        x0: int,
        label: str,
        rect: pygame.Rect,
        color: pygame.Color,
        w_tiles: float,
        h_tiles: float,
        extra: str = "",
        *,
        inner: pygame.Rect | None = None,
        inner_color: pygame.Color | None = None,
        inner_label: str = "",
    ) -> None:
        panel = pygame.Rect(int(x0), int(title_h), int(panel_w), int(panel_h))
        pygame.draw.rect(surf, (24, 24, 30), panel, border_radius=10)
        pygame.draw.rect(surf, (70, 70, 86), panel, 2, border_radius=10)

        _draw_text(surf, font, label, (int(panel.centerx), int(panel.top + 10)), pygame.Color(230, 230, 240), anchor="midtop")
        if extra:
            _draw_text(surf, font_s, extra, (int(panel.centerx), int(panel.top + 30)), pygame.Color(170, 170, 180), anchor="midtop")

        r = pygame.Rect(rect)
        r.center = (int(panel.centerx), int(panel.centery + 16))
        fill = pygame.Color(color.r, color.g, color.b, 55)
        pygame.draw.rect(surf, fill, r)
        pygame.draw.rect(surf, color, r, 2)

        if isinstance(inner, pygame.Rect):
            ir = pygame.Rect(inner)
            ir.topleft = (int(r.left), int(r.top))
            fill_col = inner_color if isinstance(inner_color, pygame.Color) else pygame.Color(color.r, color.g, color.b)
            fill = pygame.Color(fill_col.r, fill_col.g, fill_col.b, 55)
            pygame.draw.rect(surf, fill, ir)
            pygame.draw.rect(surf, fill_col, ir, 1)
            if inner_label:
                _draw_text(surf, font_s, inner_label, (int(ir.centerx), int(ir.bottom + 6)), pygame.Color(180, 180, 190), anchor="midtop")

        # Dimension lines.
        w_text = f"宽 {_fmt_len_tiles_px(float(w_tiles), tile_px=tile_px)}"
        h_text = f"高 {_fmt_len_tiles_px(float(h_tiles), tile_px=tile_px)}"
        _draw_dim_h(surf, font_s, int(r.left), int(r.right), int(r.top - 14), w_text, color)
        _draw_dim_v(surf, font_s, int(r.left - 14), int(r.top), int(r.bottom), h_text, color)

    left_x = 0
    right_x = int(panel_w + panel_gap)
    draw_panel(
        left_x,
        "1F（不含立面）",
        floor_rect,
        pygame.Color(255, 175, 70),
        float(w),
        float(floor_h_tiles),
        extra=f"build {w}×{h}  inner {w}×{floor_h_tiles}  ({w*tile_px}×{floor_h_tiles*tile_px}px)",
    )
    draw_panel(
        right_x,
        "屋顶（绘制范围，不含立面）",
        roof_draw_rect,
        pygame.Color(240, 90, 90),
        float(rw),
        float(roof_draw_tiles_h),
        extra=f"roof_tile {rw}×{rh}  cut {cut_px}px  (可见高={roof_draw_px_h}px)",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(out_path))
    return out_path


def main() -> int:
    out = Path(__file__).with_name("__roof_floor_1f_compare.png")
    export_diagram(out_path=out)
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
