from __future__ import annotations

import math
import sys
from typing import Callable

import pygame

import game


def _patch_weapon_rotation() -> None:
    # Use the built-in xiaotou-style rotation functions already present in
    # `game.py` (rotate_weapon_texture / rotate_weapon_sprite).
    return


def _patch_item_defs_unified() -> None:
    # Some render paths look up items in `_ITEMS` only. Merge `_BULK_ITEMS`
    # in so melee weapons (铁管/木棍等) are always found and drawn as weapons
    # instead of falling back to fists.
    cls = game.HardcoreSurvivalState
    try:
        items = getattr(cls, "_ITEMS", None)
        bulk = getattr(cls, "_BULK_ITEMS", None)
        if not isinstance(items, dict) or not isinstance(bulk, dict):
            return
        for iid, idef in bulk.items():
            if iid not in items:
                items[iid] = idef
    except Exception:
        return


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


def _patch_disable_internal_gun_draw() -> None:
    # Draw guns on the scaled output surface (like xiaotou) for better looking
    # rotations. To avoid double-drawing, skip the internal gun blit but keep
    # the player's facing based on aim_dir.
    cls = game.HardcoreSurvivalState
    orig = cls._draw_player

    def draw_player(self: game.HardcoreSurvivalState, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        gun = getattr(self, "gun", None)
        if gun is None:
            return orig(self, surface, int(cam_x), int(cam_y))

        p = getattr(self, "player", None)
        if p is None:
            return orig(self, surface, int(cam_x), int(cam_y))

        aim = pygame.Vector2(getattr(self, "aim_dir", pygame.Vector2(0, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(getattr(p, "facing", pygame.Vector2(1, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(1, 0)
        aim = aim.normalize()

        prev_facing = pygame.Vector2(getattr(p, "facing", pygame.Vector2(1, 0)))
        try:
            p.facing = pygame.Vector2(aim)
            self.gun = None
            return orig(self, surface, int(cam_x), int(cam_y))
        finally:
            self.gun = gun
            try:
                p.facing = prev_facing
            except Exception:
                pass

    cls._draw_player = draw_player


def _patch_scaled_weapon_overlay() -> None:
    # Overlay weapon draw onto the already-scaled surface (xiaotou style):
    # scale sprite -> rotate_weapon_texture -> blit. This keeps rotation looking
    # like "贴图" and avoids tiny-sprite jaggies being magnified.

    # Cache scaled sprites.
    _SCALE_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}

    def _scale_sprite(img: pygame.Surface, *, sx: float, sy: float) -> pygame.Surface:
        tw = int(max(1, int(round(float(img.get_width()) * float(sx)))))
        th = int(max(1, int(round(float(img.get_height()) * float(sy)))))
        key = (int(id(img)), int(tw), int(th))
        cached = _SCALE_CACHE.get(key)
        if cached is not None:
            return cached
        if float(sx) >= 1.0 and float(sy) >= 1.0:
            out = pygame.transform.scale(img, (int(tw), int(th)))
        else:
            out = pygame.transform.smoothscale(img, (int(tw), int(th)))
        _SCALE_CACHE[key] = out
        return out

    def _quantize_deg(deg: float, step_deg: float) -> int:
        step_deg = float(step_deg) if step_deg else 2.0
        step_deg = float(max(0.5, min(45.0, float(step_deg))))
        return int(round(round(float(deg) / float(step_deg)) * float(step_deg))) % 360

    cls = game.HardcoreSurvivalState

    def draw_weapon_overlay_scaled(self: game.HardcoreSurvivalState, surface_scaled: pygame.Surface) -> None:
        gun = getattr(self, "gun", None)
        if gun is None:
            return
        if getattr(self, "mount", None) is not None:
            return
        if bool(getattr(self, "world_map_open", False)) or bool(getattr(self, "inv_open", False)) or bool(getattr(self, "pause_open", False)):
            # UI layers already cover the whole view; keep it simple.
            return

        player_rect = getattr(self, "_last_player_screen_rect", None)
        if not isinstance(player_rect, pygame.Rect):
            return

        app = getattr(self, "app", None)
        if app is None:
            return
        sx = float(getattr(app, "view_scale_x", 1.0))
        sy = float(getattr(app, "view_scale_y", 1.0))
        s = float(min(sx, sy))

        # Aim direction.
        aim = pygame.Vector2(getattr(self, "aim_dir", pygame.Vector2(1, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(getattr(getattr(self, "player", None), "facing", pygame.Vector2(1, 0)))
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(1, 0)
        aim = aim.normalize()

        gun_id = str(getattr(gun, "gun_id", "pistol") or "pistol")
        try:
            self._ensure_item_visuals(gun_id)
        except Exception:
            pass
        base = (getattr(self, "_ITEM_SPRITES_WORLD", {}) or {}).get(gun_id) or (getattr(self, "_ITEM_SPRITES", {}) or {}).get(gun_id)
        if base is None:
            return

        flip = float(aim.x) < 0.0
        src = base
        if flip:
            src = game.flip_x_pixel_sprite(src)

        # Compute hand position in INTERNAL space using the same skeleton nodes.
        d = str(getattr(getattr(self, "player", None), "dir", "down") or "down")
        height_delta = 0
        av = getattr(self, "avatar", None)
        if av is not None:
            try:
                hidx = int(getattr(av, "height", 1))
                height_delta = 1 if hidx == 0 else -1 if hidx == 2 else 0
            except Exception:
                height_delta = 0

        pf_walk = getattr(self, "player_frames", getattr(self, "_PLAYER_FRAMES", {}))
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = (pf.get(d) if isinstance(pf, dict) else None) or (pf_walk.get("down") if isinstance(pf_walk, dict) else None) or ()
        moving = bool(getattr(getattr(self, "player", None), "vel", pygame.Vector2(0, 0)).length_squared() > 1.0)
        walk_idx = 0
        idle_anim = True
        if moving and len(frames) > 1:
            idle_anim = False
            walk = frames[1:]
            if walk:
                phase = (float(getattr(getattr(self, "player", None), "walk_phase", 0.0)) % math.tau) / math.tau
                walk_idx = int(phase * len(walk)) % len(walk)

        sk = self._survivor_skeleton_nodes(
            d,
            int(walk_idx),
            idle=bool(idle_anim),
            height_delta=int(height_delta),
            run=bool(is_run),
        )
        hand_node = sk.get("r_hand")
        if hand_node is not None:
            hx, hy = int(hand_node[0]), int(hand_node[1])
            base_hand = pygame.Vector2(int(player_rect.left + hx), int(player_rect.top + hy))
        else:
            base_hand = pygame.Vector2(int(player_rect.centerx), int(player_rect.centery + 3))

        hand = base_hand + aim * 4.0

        # Convert INTERNAL -> scaled surface coords.
        grip_px = pygame.Vector2(float(hand.x) * sx, float(hand.y) * sy)

        deg = -math.degrees(math.atan2(float(aim.y), float(aim.x)))
        if flip:
            deg -= 180.0
        if float(getattr(gun, "reload_left", 0.0)) > 0.0 and float(getattr(gun, "reload_total", 0.0)) > 0.0:
            p = 1.0 - float(getattr(gun, "reload_left", 0.0)) / max(1e-6, float(getattr(gun, "reload_total", 0.0)))
            deg += float(p) * 360.0

        step_deg = 2.0
        qdeg = _quantize_deg(float(deg), float(step_deg))

        # Scale sprite first (xiaotou-style), then rotate as texture.
        src_s = _scale_sprite(src, sx=sx, sy=sy)
        rot = game.rotate_weapon_texture(src_s, float(deg), step_deg=float(step_deg), outline_diagonal=True)

        # Grip point: compute on the scaled sprite so the pivot matches the
        # actual rotated texture in screen pixels.
        gx_s, gy_s = game._sprite_grip_point(src_s)
        center_local = pygame.Vector2(float(src_s.get_width()) * 0.5, float(src_s.get_height()) * 0.5)
        offset_local = pygame.Vector2(float(gx_s), float(gy_s)) - center_local
        offset_rot = game._rotate_vec_screen_ccw(offset_local, float(qdeg))
        draw_center = pygame.Vector2(float(grip_px.x), float(grip_px.y)) - offset_rot
        rr = rot.get_rect(center=(int(round(draw_center.x)), int(round(draw_center.y))))
        surface_scaled.blit(rot, rr)

        # Muzzle flash (scaled).
        if float(getattr(self, "muzzle_flash_left", 0.0)) > 0.0 and float(getattr(gun, "reload_left", 0.0)) <= 0.0:
            muzzle = hand + aim * 13.0
            mx = int(round(float(muzzle.x) * sx))
            my = int(round(float(muzzle.y) * sy))
            r = 3 if float(getattr(self, "muzzle_flash_left", 0.0)) > 0.03 else 2
            r = max(1, int(round(float(r) * s)))
            pygame.draw.circle(surface_scaled, (255, 240, 190), (mx, my), r)
            pygame.draw.circle(surface_scaled, (255, 200, 120), (mx, my), max(1, r - 1))

    cls._draw_weapon_overlay_scaled = draw_weapon_overlay_scaled

    # Patch App.blit_scaled to call the overlay on the scaled surface.
    app_cls = game.App
    orig_blit_scaled = app_cls.blit_scaled

    def blit_scaled(self: game.App) -> None:
        self._update_view_transform()
        pygame.transform.scale(self.render, self._scaled.get_size(), self._scaled)
        if getattr(self, "_grid_overlay", None) is not None:
            self._scaled.blit(self._grid_overlay, (0, 0))
            self._draw_grid_markers(self._scaled)
        try:
            st = getattr(self, "state", None)
            if st is not None and hasattr(st, "_draw_weapon_overlay_scaled"):
                st._draw_weapon_overlay_scaled(self._scaled)
        except Exception:
            pass
        self.screen.fill((0, 0, 0))
        self.screen.blit(self._scaled, getattr(self, "view_offset", (0, 0)))

    app_cls.blit_scaled = blit_scaled


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


def _patch_aim_dir_stable() -> None:
    # Stabilize aim_dir against sub-pixel player movement: use rounded player
    # world coords when computing the aim vector so diagonal movement doesn't
    # cause tiny aim oscillations (which can flip 4-dir sprites).
    cls = game.HardcoreSurvivalState

    def compute_aim_dir(self: game.HardcoreSurvivalState) -> pygame.Vector2:
        m = self.app.screen_to_internal(pygame.mouse.get_pos())
        if m is None:
            d = pygame.Vector2(getattr(getattr(self, "player", None), "facing", pygame.Vector2(1, 0)))
            return d.normalize() if d.length_squared() > 0.001 else pygame.Vector2(1, 0)

        mw = pygame.Vector2(m) + pygame.Vector2(float(getattr(self, "cam_x", 0)), float(getattr(self, "cam_y", 0)))

        p = getattr(self, "player", None)
        if p is None:
            v = mw
        else:
            px = float(game.iround(float(p.pos.x)))
            py = float(game.iround(float(p.pos.y)))
            v = mw - pygame.Vector2(px, py)

        if v.length_squared() <= 0.001:
            d = pygame.Vector2(getattr(getattr(self, "player", None), "facing", pygame.Vector2(1, 0)))
            return d.normalize() if d.length_squared() > 0.001 else pygame.Vector2(1, 0)
        return v.normalize()

    cls._compute_aim_dir = compute_aim_dir


def apply_patches() -> None:
    _patch_weapon_rotation()
    _patch_item_defs_unified()
    _patch_grip_point()
    _patch_item_visual_sizes()
    _patch_camera_snap()
    _patch_disable_internal_gun_draw()
    _patch_scaled_weapon_overlay()
    _patch_melee_aim()
    _patch_aim_dir_stable()


def main() -> int:
    apply_patches()
    return int(game.main())


if __name__ == "__main__":
    raise SystemExit(main())
