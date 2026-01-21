from __future__ import annotations

import math
import sys
from typing import Callable

import pygame

import game


def _patch_weapon_rotation() -> None:
    # Force "perfect pixel" rotation for held weapons by using the crisp
    # pixel-art pipeline (supersample -> rotate -> nearest downsample -> clamp
    # alpha -> palette quantize -> outline). This avoids blur and reduces
    # diagonal "pixel breakup" / flicker.
    base_rotate_crisp = game.rotate_pixel_sprite_crisp

    def rotate_weapon_sprite(
        sprite: pygame.Surface,
        deg: float,
        *,
        step_deg: float = 2.0,
        outline_diagonal: bool = True,
    ) -> pygame.Surface:
        eff_step = max(3.0, float(step_deg))
        return base_rotate_crisp(sprite, deg, step_deg=float(eff_step), upscale=3)

    def rotate_weapon_texture(
        sprite: pygame.Surface,
        deg: float,
        *,
        step_deg: float = 2.0,
        outline_diagonal: bool = True,
    ) -> pygame.Surface:
        eff_step = max(3.0, float(step_deg))
        return base_rotate_crisp(sprite, deg, step_deg=float(eff_step), upscale=3)

    game.rotate_weapon_sprite = rotate_weapon_sprite
    game.rotate_weapon_texture = rotate_weapon_texture


def _patch_grip_point() -> None:
    # Improve the grip heuristic so weapons rotate around a stable handle/grip
    # point and don't "orbit" around their center. Cache by sprite id like the
    # original.
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
        opaque: list[tuple[int, int]] = []
        for yy in range(h):
            for xx in range(w):
                try:
                    if int(sprite.get_at((xx, yy)).a) <= 0:
                        continue
                except Exception:
                    continue
                opaque.append((xx, yy))
                if xx < minx:
                    minx = xx
                if xx > maxx:
                    maxx = xx
                if yy < miny:
                    miny = yy
                if yy > maxy:
                    maxy = yy

        if not opaque or maxy < 0 or maxx < 0:
            pt = (w // 2, h // 2)
            base_cache[id(sprite)] = pt
            return pt

        span_y = max(1, int(maxy - miny))
        top_y1 = int(miny + span_y * 0.35)
        bottom_y0 = int(miny + span_y * 0.65)

        # Detect the common "melee icon" diagonal (bottom-left -> top-right).
        # If so, prefer leftmost ties (grip on the bottom-left end).
        diag = False
        try:
            top_xs = [xx for (xx, yy) in opaque if yy <= top_y1]
            bottom_xs = [xx for (xx, yy) in opaque if yy >= bottom_y0]
            if top_xs and bottom_xs:
                top_mean = sum(top_xs) / float(len(top_xs))
                bottom_mean = sum(bottom_xs) / float(len(bottom_xs))
                diag = float(bottom_mean) < float(top_mean) - 0.6
        except Exception:
            diag = False

        # Pick the X column with the strongest "lower-half" presence.
        # Guns tend to have a vertical-ish grip/mag column; melee diagonals are
        # thin so ties are common -> use `diag` to break ties left.
        best_x = minx
        best_weight = -1
        best_bottom_y = maxy
        for xx in range(minx, maxx + 1):
            weight = 0
            bottom_y = -1
            for yy in range(miny, maxy + 1):
                try:
                    if int(sprite.get_at((xx, yy)).a) <= 0:
                        continue
                except Exception:
                    continue
                weight += 2 if yy >= bottom_y0 else 1
                bottom_y = yy if yy > bottom_y else bottom_y
            if weight <= 0:
                continue
            if weight > best_weight:
                best_weight = weight
                best_x = xx
                best_bottom_y = bottom_y if bottom_y >= 0 else maxy
            elif weight == best_weight:
                if diag and xx < best_x:
                    best_x = xx
                    best_bottom_y = bottom_y if bottom_y >= 0 else best_bottom_y
                if (not diag) and xx > best_x:
                    best_x = xx
                    best_bottom_y = bottom_y if bottom_y >= 0 else best_bottom_y

        if best_weight <= 0:
            pt = (w // 2, h // 2)
            base_cache[id(sprite)] = pt
            return pt

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
            p = getattr(self, "player", None)
            if p is None:
                return

            # Pixel-perfect lock: derive cam from the rounded player world pos
            # so the player stays perfectly centered (no diagonal shimmer).
            px = int(game.iround(float(p.pos.x)))
            py = int(game.iround(float(p.pos.y)))
            cam_x = int(px - int(game.INTERNAL_W) // 2)
            cam_y = int(py - int(game.INTERNAL_H) // 2)
            setattr(self, "cam_fx", float(cam_x))
            setattr(self, "cam_fy", float(cam_y))
            setattr(self, "cam_x", int(cam_x))
            setattr(self, "cam_y", int(cam_y))

            # Keep aim_dir consistent with the snapped camera (prevents tiny
            # oscillations in aim/dir while moving diagonally).
            gun = getattr(self, "gun", None)
            if gun is not None and float(getattr(gun, "reload_left", 0.0)) > 0.0 and getattr(self, "_reload_lock_dir", None) is not None:
                lock = pygame.Vector2(getattr(self, "_reload_lock_dir", pygame.Vector2(1, 0)))
                if lock.length_squared() > 0.001:
                    self.aim_dir = lock.normalize()
                else:
                    self.aim_dir = pygame.Vector2(1, 0)
            else:
                try:
                    self.aim_dir = pygame.Vector2(self._compute_aim_dir())
                except Exception:
                    self.aim_dir = pygame.Vector2(1, 0)
        except Exception:
            return

    game.HardcoreSurvivalState.update = update


def _patch_melee_aim() -> None:
    # Make melee swings (wood/pipe/etc) follow the mouse aim direction, matching
    # the weapon rotation, instead of using movement direction (which looks like
    # "punching" in the wrong direction).
    cls = game.HardcoreSurvivalState

    def start_punch(self: game.HardcoreSurvivalState) -> None:
        if int(getattr(getattr(self, "player", None), "hp", 0)) <= 0:
            return
        if getattr(self, "mount", None) is not None:
            try:
                self._set_hint("下车后才能出拳", seconds=0.9)
            except Exception:
                pass
            return
        if getattr(self, "gun", None) is not None:
            try:
                self._set_hint("手上拿着枪，不能近战；去背包装备木棍/铁管", seconds=1.0)
            except Exception:
                pass
            return
        if bool(getattr(self, "hr_interior", False)) or bool(getattr(self, "sch_interior", False)) or bool(getattr(self, "house_interior", False)):
            return
        if float(getattr(self, "punch_cooldown_left", 0.0)) > 0.0:
            return

        aim = pygame.Vector2(getattr(self, "aim_dir", pygame.Vector2(0, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(getattr(getattr(self, "player", None), "facing", pygame.Vector2(1, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(1, 0)
        self.punch_dir = aim.normalize()

        self.punch_hand = 1 - int(getattr(self, "punch_hand", 0))
        mid = str(getattr(self, "melee_weapon_id", "") or "")
        mdef = (getattr(self, "_MELEE_DEFS", {}) or {}).get(mid) or (getattr(self, "_MELEE_DEFS", {}) or {}).get("fist")
        if mdef is None:
            return
        self._melee_swing_id = str(getattr(mdef, "id", "fist"))
        self.punch_left = float(getattr(mdef, "total_s", getattr(self, "_PUNCH_TOTAL_S", 0.26)))
        self.punch_hit_done = False
        self.punch_cooldown_left = float(getattr(mdef, "cooldown_s", getattr(self, "_PUNCH_COOLDOWN_S", 0.28)))

        try:
            cost = float(getattr(mdef, "stamina_cost", 5.0))
            p = getattr(self, "player", None)
            if p is not None:
                p.stamina = float(game.clamp(float(getattr(p, "stamina", 0.0)) - float(cost), 0.0, 100.0))
        except Exception:
            pass

        try:
            self.app.play_sfx("swing")
        except Exception:
            pass

    cls._start_punch = start_punch


def apply_patches() -> None:
    _patch_weapon_rotation()
    _patch_grip_point()
    _patch_item_visual_sizes()
    _patch_camera_snap()
    _patch_melee_aim()


def main() -> int:
    apply_patches()
    return int(game.main())


if __name__ == "__main__":
    raise SystemExit(main())
