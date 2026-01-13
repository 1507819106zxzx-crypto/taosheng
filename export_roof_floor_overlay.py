from __future__ import annotations

from pathlib import Path

import pygame

import game


class _DummyApp:
    def __init__(self) -> None:
        self.config = game.GameConfig(
            scale=2,
            fullscreen=False,
            show_grid=False,
            debug_highrise_test=True,
        )


def _fmt_len_tiles_px(tiles: float, *, tile_px: int) -> str:
    if abs(float(tiles) - round(float(tiles))) < 1e-6:
        t = f"{int(round(float(tiles)))}t"
    else:
        t = f"{float(tiles):.1f}t"
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
    pygame.draw.line(surface, color, (x, y1), (x - 3, y1 + 6), 1)
    pygame.draw.line(surface, color, (x, y1), (x + 3, y1 + 6), 1)
    pygame.draw.line(surface, color, (x, y2), (x - 3, y2 - 6), 1)
    pygame.draw.line(surface, color, (x, y2), (x + 3, y2 - 6), 1)
    _draw_text(surface, font, text, (int(x - 2), int((y1 + y2) // 2)), color, anchor="midright")


def export_overlay_diagram(*, out_path: Path) -> Path:
    pygame.init()
    pygame.font.init()
    font_path = game.pick_ui_font_path()
    font = pygame.font.Font(font_path, 22) if font_path else pygame.font.Font(None, 22)
    font_s = pygame.font.Font(font_path, 18) if font_path else pygame.font.Font(None, 18)

    app = _DummyApp()
    state = game.HardcoreSurvivalState(app, avatar=game.SurvivalAvatar())
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
    tile_px = int(state.TILE_SIZE)

    floor_px_w = int(w) * int(tile_px)
    floor_px_h = int(h) * int(tile_px)

    # Facades overlap the roof in screen space, but they are not part of the roof.
    # High-rise (style=6): per-floor footprint equals the roof-visible region (no facade band).
    facade_band_px_h = int(tile_px)
    if int(style) == 6 or (int(style) == 1 and int(floors) > 1):
        facade_band_px_h = int(tile_px) + int(face_h_px)

    roof_cut_px = 0
    roof_draw_px_h = int(floor_px_h)
    if int(style) == 1 and int(floors) > 1:
        roof_cut_px = int(max(0, int(face_h_px) - 2))
        min_vis = 14
        max_cut = max(0, int(floor_px_h) - int(min_vis))
        roof_cut_px = int(min(int(roof_cut_px), int(max_cut)))
        roof_draw_px_h = int(max(0, int(floor_px_h) - int(roof_cut_px)))        
    elif int(style) == 6:
        roof_draw_px_h = int(max(0, int(floor_px_h) - int(facade_band_px_h)))

    # Diagram scale.
    s = 3  # px per world pixel
    outer = pygame.Rect(0, 0, int(floor_px_w) * int(s), int(floor_px_h) * int(s))
    roof_draw = pygame.Rect(0, 0, int(floor_px_w) * int(s), int(roof_draw_px_h) * int(s))
    facade_h = int(min(int(facade_band_px_h) * int(s), int(outer.h)))
    facade = pygame.Rect(0, int(outer.h - facade_h), int(outer.w), int(facade_h))

    margin = 34
    title_h = 44
    canvas_w = int(outer.w + margin * 2)
    canvas_h = int(title_h + outer.h + margin * 2 + 80)
    surf = pygame.Surface((int(canvas_w), int(canvas_h)), pygame.SRCALPHA)
    surf.fill((18, 18, 22))

    outer.center = (int(canvas_w // 2), int(title_h + margin + outer.h // 2))
    roof_draw.topleft = (int(outer.left), int(outer.top))
    facade.topleft = (int(outer.left), int(outer.bottom - facade.h))

    title = "单图：1F(不含立面) 与 屋顶(不含立面)"
    _draw_text(surf, font, title, (int(canvas_w // 2), 10), pygame.Color(240, 240, 240), anchor="midtop")
    sub = (
        f"tile={tile_px}px  floors={floors}  style={style}  face_h={face_h_px}px  "
        f"facade_band={facade_band_px_h}px  roof_cut={roof_cut_px}px"
    )
    _draw_text(surf, font_s, sub, (int(canvas_w // 2), 30), pygame.Color(180, 180, 190), anchor="midtop")

    if roof_draw.h > 0:
        surf.fill((240, 90, 90, 70), roof_draw)
    if facade.h > 0:
        surf.fill((120, 80, 150, 70), facade)

    pygame.draw.rect(surf, pygame.Color(70, 70, 86), outer, 2)
    pygame.draw.rect(surf, pygame.Color(255, 175, 70), roof_draw, 2)
    if facade.h > 0:
        pygame.draw.line(surf, pygame.Color(120, 80, 150), (outer.left, facade.top), (outer.right, facade.top), 1)

    legend_y = int(outer.bottom + 10)
    _draw_text(surf, font_s, "灰线=建筑全范围(含立面带)", (int(outer.left), int(legend_y)), pygame.Color(160, 160, 175), anchor="topleft")
    _draw_text(surf, font_s, "橙线=1F范围(不含立面带)", (int(outer.left), int(legend_y + 14)), pygame.Color(255, 175, 70), anchor="topleft")
    _draw_text(surf, font_s, "红填充=屋顶(可见/不含立面)", (int(outer.left), int(legend_y + 28)), pygame.Color(240, 120, 120), anchor="topleft")
    _draw_text(surf, font_s, "紫填充=立面占用(不是屋顶)", (int(outer.left), int(legend_y + 42)), pygame.Color(170, 140, 200), anchor="topleft")

    w_text_outer = f"宽 {_fmt_len_tiles_px(float(w), tile_px=tile_px)}"
    inner_tiles_h = float(roof_draw_px_h) / float(max(1, int(tile_px)))
    h_text_outer = f"高 {_fmt_len_tiles_px(inner_tiles_h, tile_px=tile_px)}"
    _draw_dim_h(surf, font_s, int(roof_draw.left), int(roof_draw.right), int(roof_draw.top - 16), w_text_outer, pygame.Color(255, 175, 70))
    _draw_dim_v(surf, font_s, int(roof_draw.left - 16), int(roof_draw.top), int(roof_draw.bottom), h_text_outer, pygame.Color(255, 175, 70))

    hv_text = f"屋顶高 {_fmt_len_tiles_px(inner_tiles_h, tile_px=tile_px)}"
    _draw_dim_v(
        surf,
        font_s,
        int(outer.right + 16),
        int(outer.top),
        int(outer.top + roof_draw.h),
        hv_text,
        pygame.Color(240, 90, 90),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(out_path))
    return out_path


def main() -> int:
    out = Path(__file__).with_name("__roof_floor_1f_overlay.png")
    export_overlay_diagram(out_path=out)
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
