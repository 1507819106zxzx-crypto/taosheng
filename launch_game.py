from __future__ import annotations

import math
import sys
from typing import Callable

import pygame

import game


def _patch_weapon_rotation() -> None:
    # Force "perfect pixel" rotation for held weapons by using the texture-style
    # rotation pipeline (alpha clamp + palette quantize). This avoids the blur /
    # pixel breakup from smooth downsampling at diagonal angles.
    base_rotate_texture = game.rotate_weapon_texture

    def rotate_weapon_sprite(
        sprite: pygame.Surface,
        deg: float,
        *,
        step_deg: float = 2.0,
        outline_diagonal: bool = True,
    ) -> pygame.Surface:
        return base_rotate_texture(sprite, deg, step_deg=step_deg, outline_diagonal=outline_diagonal)

    game.rotate_weapon_sprite = rotate_weapon_sprite


def _patch_grip_point() -> None:
    # Improve the grip heuristic so guns rotate around the handle/mag region
    # (instead of the far-left stock). Cache by sprite id like the original.
    base_cache = game._SPRITE_GRIP_CACHE

    def sprite_grip_point(sprite: pygame.Surface) -> tuple[int, int]:
        cached = base_cache.get(id(sprite))
        if cached is not None:
            return cached

        w, h = sprite.get_size()
        if w <= 0 or h <= 0:
            base_cache[id(sprite)] = (0, 0)
            return (0, 0)

        minx = w
        miny = h
        maxx = -1
        maxy = -1
        for yy in range(h):
            for xx in range(w):
                try:
                    if int(sprite.get_at((xx, yy)).a) <= 0:
                        continue
                except Exception:
                    continue
                if xx < minx:
                    minx = xx
                if xx > maxx:
                    maxx = xx
                if yy < miny:
                    miny = yy
                if yy > maxy:
                    maxy = yy

        if maxy < 0 or maxx < 0:
            pt = (w // 2, h // 2)
            base_cache[id(sprite)] = pt
            return pt

        # Search the lower part of the sprite and pick the column with the most
        # opaque pixels; tie-break toward the right (gun grips are not on stocks).
        lower_y0 = int(miny + (maxy - miny) * 0.55)
        best_x = minx
        best_count = -1
        best_bottom_y = maxy
        for xx in range(minx, maxx + 1):
            count = 0
            bottom_y = -1
            for yy in range(lower_y0, maxy + 1):
                try:
                    if int(sprite.get_at((xx, yy)).a) <= 0:
                        continue
                except Exception:
                    continue
                count += 1
                bottom_y = yy if yy > bottom_y else bottom_y
            if count <= 0:
                continue
            if (count > best_count) or (count == best_count and xx > best_x):
                best_count = count
                best_x = xx
                best_bottom_y = bottom_y if bottom_y >= 0 else maxy

        if best_count <= 0:
            # Fallback to the original heuristic style (rear/bottom of bbox).
            gx = int(game.clamp(int(minx) + 2, 0, w - 1))
            gy = int(game.clamp(int(maxy) - 2, 0, h - 1))
            pt = (gx, gy)
        else:
            gx = int(game.clamp(int(best_x), 0, w - 1))
            gy = int(game.clamp(int(best_bottom_y) - 1, 0, h - 1))
            pt = (gx, gy)

        base_cache[id(sprite)] = pt
        return pt

    game._sprite_grip_point = sprite_grip_point


def _patch_item_visual_sizes() -> None:
    # The in-hand gun draw currently prefers `_ITEM_SPRITES` (inventory size).
    # Normalize gun sprites so `_ITEM_SPRITES[gun_id]` uses the world-sized
    # sprite, keeping the held gun size consistent with pickups.
    cls = game.HardcoreSurvivalState

    # Apply once for any already-created gun sprites (e.g. pistol on import).
    try:
        items = dict(getattr(cls, "_ITEMS", {}) or {})
        bulk = dict(getattr(cls, "_BULK_ITEMS", {}) or {})
        items.update(bulk)
        sprites = getattr(cls, "_ITEM_SPRITES", None)
        sprites_world = getattr(cls, "_ITEM_SPRITES_WORLD", None)
        if isinstance(sprites, dict) and isinstance(sprites_world, dict):
            for iid, idef in items.items():
                if str(getattr(idef, "kind", "") or "") != "gun":
                    continue
                world = sprites_world.get(str(iid))
                if world is not None:
                    sprites[str(iid)] = world
    except Exception:
        pass

    orig: Callable[[game.HardcoreSurvivalState, str], None] = cls._ensure_item_visuals

    def ensure_item_visuals(self: game.HardcoreSurvivalState, item_id: str) -> None:
        orig(self, item_id)
        try:
            iid = str(item_id or "")
            idef = (getattr(self, "_ITEMS", {}) or {}).get(iid) or (getattr(self, "_BULK_ITEMS", {}) or {}).get(iid)
            kind = str(getattr(idef, "kind", "") or "")
            if kind != "gun":
                return
            sprites = getattr(self, "_ITEM_SPRITES", None)
            sprites_world = getattr(self, "_ITEM_SPRITES_WORLD", None)
            if not isinstance(sprites, dict) or not isinstance(sprites_world, dict):
                return
            world = sprites_world.get(iid)
            if world is None:
                return
            sprites[iid] = world
        except Exception:
            return

    cls._ensure_item_visuals = ensure_item_visuals


def _patch_camera_snap() -> None:
    # Override the diagonal-only camera smoothing with a strict pixel-snap
    # follow. This removes the "up/down jitter" seen while sprinting diagonally.
    orig_update: Callable[[game.HardcoreSurvivalState, float], None] = game.HardcoreSurvivalState.update

    def update(self: game.HardcoreSurvivalState, dt: float) -> None:
        orig_update(self, dt)
        try:
            if bool(getattr(self, "house_interior", False)) or bool(getattr(self, "hr_interior", False)) or bool(
                getattr(self, "sch_interior", False)
            ):
                return
            focus = pygame.Vector2(getattr(self, "player").pos)
            target_cam_x = float(focus.x - float(game.INTERNAL_W) / 2.0)
            target_cam_y = float(focus.y - float(game.INTERNAL_H) / 2.0)
            setattr(self, "cam_fx", float(target_cam_x))
            setattr(self, "cam_fy", float(target_cam_y))
            setattr(self, "cam_x", int(game.iround(float(target_cam_x))))
            setattr(self, "cam_y", int(game.iround(float(target_cam_y))))
        except Exception:
            return

    game.HardcoreSurvivalState.update = update


def apply_patches() -> None:
    _patch_weapon_rotation()
    _patch_grip_point()
    _patch_item_visual_sizes()
    _patch_camera_snap()


def main() -> int:
    apply_patches()
    return int(game.main())


if __name__ == "__main__":
    raise SystemExit(main())
