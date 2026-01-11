from __future__ import annotations

from array import array
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pygame

INTERNAL_W = 480
INTERNAL_H = 270
FPS = 60
GRID_MARKS = (10, 50, 100, 200, 300, 400)

SETTINGS_PATH = Path(__file__).with_name("settings.json")


def pick_ui_font_path() -> str | None:
    candidates = [
        "Microsoft YaHei",
        "MicrosoftYaHei",
        "msyh",
        "SimHei",
        "simhei",
        "SimSun",
        "simsun",
        "NSimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ]
    for name in candidates:
        try:
            path = pygame.font.match_font(name)
        except Exception:
            path = None
        if path:
            return path

    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        fonts_dir = Path(windir) / "Fonts"
        for filename in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc", "simsun.ttf"):
            p = fonts_dir / filename
            if p.exists():
                return str(p)
    return None


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def iround(x: float) -> int:
    # Python's round() uses bankers rounding (ties to even) which can cause
    # visible 1px jitter when positions sit on .5 boundaries (e.g. odd-sized
    # sprites/colliders). Use a "round half away from zero" instead.
    x = float(x)
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
    color: pygame.Color,
    *,
    anchor: str = "topleft",  # topleft | center | topright
) -> pygame.Rect:
    img = font.render(text, False, color)
    rect = img.get_rect()
    if anchor == "center":
        rect.center = pos
    elif anchor == "topright":
        rect.topright = pos
    else:
        rect.topleft = pos
    surface.blit(img, rect)
    return rect


def draw_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    label: str,
    rect: pygame.Rect,
    *,
    selected: bool = False,
    disabled: bool = False,
) -> pygame.Rect:
    if disabled:
        bg = (18, 18, 22)
        border = (50, 50, 60)
        text = pygame.Color(120, 120, 130)
    else:
        bg = (34, 34, 42) if selected else (24, 24, 30)
        border = (110, 110, 130) if selected else (70, 70, 86)
        text = pygame.Color(240, 240, 240) if selected else pygame.Color(200, 200, 210)

    pygame.draw.rect(surface, bg, rect, border_radius=6)
    pygame.draw.rect(surface, border, rect, 2, border_radius=6)
    draw_text(surface, font, label, rect.center, text, anchor="center")
    return rect


def sprite_from_pixels(pixels: Sequence[str], palette: dict[str, tuple[int, int, int]]) -> pygame.Surface:
    height = len(pixels)
    width = max((len(row) for row in pixels), default=0)
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    for y, row in enumerate(pixels):
        for x, ch in enumerate(row):
            if ch == " ":
                continue
            color = palette.get(ch)
            if color is None:
                continue
            surf.set_at((x, y), color)
    return surf


_BLEED_CACHE: dict[int, pygame.Surface] = {}
_ROTATE_CACHE: dict[tuple[int, int], pygame.Surface] = {}
_FLIPX_CACHE: dict[int, pygame.Surface] = {}


def _alpha_bleed(sprite: pygame.Surface, *, passes: int = 2) -> pygame.Surface:
    cached = _BLEED_CACHE.get(id(sprite))
    if cached is not None and cached.get_size() == sprite.get_size():
        return cached

    surf = sprite.copy()
    w, h = surf.get_size()
    passes = max(0, int(passes))
    if passes <= 0 or w <= 0 or h <= 0:
        _BLEED_CACHE[id(sprite)] = surf
        return surf

    for _ in range(passes):
        base = surf.copy()
        for y in range(h):
            for x in range(w):
                _r, _g, _b, a = base.get_at((x, y))
                if a != 0:
                    continue
                found: tuple[int, int, int] | None = None
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if ny < 0 or ny >= h:
                        continue
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        if nx < 0 or nx >= w:
                            continue
                        nr, ng, nb, na = base.get_at((nx, ny))
                        if na != 0:
                            found = (nr, ng, nb)
                            break
                    if found is not None:
                        break
                if found is not None:
                    surf.set_at((x, y), (*found, 0))

    _BLEED_CACHE[id(sprite)] = surf
    return surf


def _clamp_alpha(sprite: pygame.Surface, *, threshold: int = 150) -> pygame.Surface:
    w, h = sprite.get_size()
    if w <= 0 or h <= 0:
        return sprite
    threshold = int(clamp(int(threshold), 1, 254))
    for y in range(h):
        for x in range(w):
            r, g, b, a = sprite.get_at((x, y))
            if a == 0:
                continue
            if a < threshold:
                sprite.set_at((x, y), (0, 0, 0, 0))
            else:
                sprite.set_at((x, y), (r, g, b, 255))
    return sprite


def flip_x_pixel_sprite(sprite: pygame.Surface) -> pygame.Surface:
    cached = _FLIPX_CACHE.get(id(sprite))
    if cached is not None:
        return cached
    try:
        flipped = pygame.transform.flip(sprite, True, False)
    except Exception:
        flipped = sprite
    _FLIPX_CACHE[id(sprite)] = flipped
    return flipped


def rotate_pixel_sprite(sprite: pygame.Surface, deg: float, *, step_deg: float = 5.0) -> pygame.Surface:
    step_deg = float(step_deg) if step_deg else 5.0
    step_deg = float(clamp(step_deg, 1.0, 45.0))
    qdeg = int(round(float(deg) / step_deg) * step_deg) % 360
    key = (id(sprite), qdeg)
    cached = _ROTATE_CACHE.get(key)
    if cached is not None:
        return cached

    src = _alpha_bleed(sprite, passes=2)
    rot = pygame.transform.rotate(src, qdeg)
    rot = _clamp_alpha(rot, threshold=150)
    _ROTATE_CACHE[key] = rot
    return rot


def _make_sound(samples: array, *, volume: float = 0.35) -> pygame.mixer.Sound | None:
    if pygame.mixer.get_init() is None:
        return None
    try:
        snd = pygame.mixer.Sound(buffer=samples.tobytes())
        snd.set_volume(float(volume))
        return snd
    except Exception:
        return None


def _sfx_shot(rate: int) -> pygame.mixer.Sound | None:
    dur_s = 0.085
    n = max(1, int(rate * dur_s))
    out = array("h")
    max_amp = 32767
    for i in range(n):
        t = i / rate
        env = math.exp(-t * 55.0)
        noise = random.uniform(-1.0, 1.0) * 0.65
        thump = math.sin(math.tau * 120.0 * t) * 0.35
        click = (1.0 if i < 18 else 0.0) * 0.8
        v = (noise + thump + click) * env
        v = clamp(v, -1.0, 1.0)
        out.append(int(v * max_amp))
    return _make_sound(out, volume=0.32)


def _sfx_swing(rate: int) -> pygame.mixer.Sound | None:
    dur_s = 0.14
    n = max(1, int(rate * dur_s))
    out = array("h")
    max_amp = 32767
    for i in range(n):
        t = i / rate
        u = t / max(1e-6, dur_s)
        env = math.exp(-t * 28.0) * (1.0 if u < 1.0 else 0.0)
        sweep = 1400.0 + (260.0 - 1400.0) * u
        tone = math.sin(math.tau * sweep * t) * 0.25
        noise = random.uniform(-1.0, 1.0) * 0.45
        v = (tone + noise) * env
        v = clamp(v, -1.0, 1.0)
        out.append(int(v * max_amp))
    return _make_sound(out, volume=0.26)


class SFX:
    def __init__(self) -> None:
        self.enabled = pygame.mixer.get_init() is not None
        self.shot: pygame.mixer.Sound | None = None
        self.swing: pygame.mixer.Sound | None = None
        if not self.enabled:
            return
        rate = pygame.mixer.get_init()[0] if pygame.mixer.get_init() else 22050
        self.shot = _sfx_shot(int(rate))
        self.swing = _sfx_swing(int(rate))

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        snd = getattr(self, name, None)
        if snd is not None:
            try:
                snd.play()
            except Exception:
                return


@dataclass
class GameConfig:
    scale: int = 2
    fullscreen: bool = True
    show_grid: bool = True

    @classmethod
    def load(cls) -> "GameConfig":
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return cls(
                scale=int(data.get("scale", 2)),
                fullscreen=bool(data.get("fullscreen", True)),
                show_grid=bool(data.get("show_grid", True)),
            )
        except Exception:
            return cls()

    def save(self) -> None:
        SETTINGS_PATH.write_text(
            json.dumps(
                {"scale": int(self.scale), "fullscreen": bool(self.fullscreen), "show_grid": bool(self.show_grid)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class CharacterDef:
    id: str
    name: str
    desc: str
    sprite: pygame.Surface
    base_hp: int
    move_speed: float
    start_weapon: str
    regen: float = 0.0
    damage_mult: float = 1.0
    cooldown_mult: float = 1.0
    frames: tuple[pygame.Surface, ...] | None = None
    skeleton_frames: tuple[dict[str, tuple[int, int]], ...] | None = None


@dataclass(frozen=True)
class MapDef:
    id: str
    name: str
    desc: str
    base_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    enemy_pool: tuple[str, ...]


@dataclass(frozen=True)
class EnemyDef:
    id: str
    name: str
    color: tuple[int, int, int]
    hp: int
    speed: float
    size: int
    contact_damage: int
    xp_value: int
    frames: tuple[pygame.Surface, ...] | None = None


@dataclass(frozen=True)
class WeaponDef:
    id: str
    name: str
    desc: str
    kind: str  # melee | orbit | gun
    color: tuple[int, int, int]
    cooldown_s: float
    damage: int
    proj_speed: float = 0.0
    proj_size: int = 2
    pierce: int = 0
    magazine: int = 0
    reload_s: float = 0.0
    proj_ttl_s: float = 2.5
    spread_rad: float = 0.0
    knockback: float = 0.0
    aoe_radius: float = 0.0
    aoe_damage_mult: float = 1.0
    recoil_per_shot: float = 0.0
    recoil_decay: float = 0.0
    recoil_max: float = 0.0
    shots_per_fire: int = 1
    sprite: pygame.Surface | None = None


@dataclass(frozen=True)
class Upgrade:
    title: str
    desc: str
    apply: Callable[["Player", "GameState"], None]


class Player:
    def __init__(self, char_def: CharacterDef) -> None:
        self.char_def = char_def
        self.frames = char_def.frames or (char_def.sprite,)
        self.skeleton_frames = char_def.skeleton_frames or ()
        self._flip_cache: dict[int, pygame.Surface] = {}
        self.facing = 1
        self.anim_time = 0.0
        self.anim_index = 0
        self.draw_offset = pygame.Vector2(0, 0)
        self.sprite = self.frames[0]
        self.w = self.sprite.get_width()
        self.h = self.sprite.get_height()

        self.skeleton_nodes: dict[str, tuple[int, int]] = self._current_skeleton_nodes()

        self.pos = pygame.Vector2(INTERNAL_W // 2 - self.w // 2, INTERNAL_H // 2 - self.h // 2)
        self.max_hp = int(char_def.base_hp)
        self.hp = float(self.max_hp)
        self.regen = float(char_def.regen)

        self.move_speed = float(char_def.move_speed)
        self.damage_mult = float(char_def.damage_mult)
        self.cooldown_mult = float(char_def.cooldown_mult)
        self.xp_mult = 1.0

        self.level = 1
        self.xp = 0.0
        self.xp_to_next = 12.0

        self.invuln_s = 0.0
        self.hurt_s = 0.0
        self.last_move_dir = pygame.Vector2(1, 0)

        self.weapons: list[WeaponInstance] = []

    def _base_frame(self, index: int) -> pygame.Surface:
        if not self.frames:
            return self.char_def.sprite
        idx = max(0, min(int(index), len(self.frames) - 1))
        return self.frames[idx]

    def _current_skeleton_nodes(self) -> dict[str, tuple[int, int]]:
        if not self.skeleton_frames:
            return {}
        idx = max(0, min(int(self.anim_index), len(self.skeleton_frames) - 1))
        nodes = self.skeleton_frames[idx]
        if self.facing >= 0:
            return dict(nodes)
        w = self._base_frame(idx).get_width()
        return {name: (w - 1 - x, y) for name, (x, y) in nodes.items()}

    def _frame_for_facing(self, index: int) -> pygame.Surface:
        base = self._base_frame(index)
        if self.facing >= 0:
            return base
        cached = self._flip_cache.get(index)
        if cached is None or cached.get_size() != base.get_size():
            cached = pygame.transform.flip(base, True, False)
            self._flip_cache[index] = cached
        return cached

    def update_visual(self, dt: float, move_dir: pygame.Vector2) -> None:
        if move_dir.length_squared() > 0.0001:
            if move_dir.x < -0.1:
                self.facing = -1
            elif move_dir.x > 0.1:
                self.facing = 1

            if len(self.frames) >= 3:
                self.anim_time += dt * (7.0 + self.move_speed * 0.02)
                step = int(self.anim_time) % 2
                self.anim_index = 1 + step
            elif len(self.frames) == 2:
                self.anim_time += dt * (6.0 + self.move_speed * 0.02)
                self.anim_index = int(self.anim_time) % 2
            else:
                self.anim_index = 0
            self.draw_offset.y = -1 if self.anim_index == 1 else 0
        else:
            self.anim_time = 0.0
            self.anim_index = 0
            self.draw_offset.update(0.0, 0.0)

        self.sprite = self._frame_for_facing(self.anim_index)
        self.w = self.sprite.get_width()
        self.h = self.sprite.get_height()
        self.skeleton_nodes = self._current_skeleton_nodes()

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x), int(self.pos.y), self.w, self.h)

    def center(self) -> pygame.Vector2:
        r = self.rect()
        return pygame.Vector2(r.centerx, r.centery)

    def orbit_center(self) -> pygame.Vector2:
        chest = self.skeleton_nodes.get("chest")
        if chest is not None:
            cx, cy = chest
            return pygame.Vector2(int(self.pos.x) + cx, int(self.pos.y) + cy) + self.draw_offset
        return self.center() + self.draw_offset

    def hand_pos(self) -> pygame.Vector2:
        hand = self.skeleton_nodes.get("r_hand")
        if hand is not None:
            hx, hy = hand
            return pygame.Vector2(int(self.pos.x) + hx, int(self.pos.y) + hy) + self.draw_offset
        return self.orbit_center()


class Enemy:
    _next_uid = 1

    def __init__(self, enemy_def: EnemyDef, pos: pygame.Vector2) -> None:
        self.enemy_def = enemy_def
        self.pos = pos
        self.hp = float(enemy_def.hp)
        self.size = int(enemy_def.size)
        self.uid = Enemy._next_uid
        Enemy._next_uid += 1
        self.anim_time = random.random() * 10.0
        self.vel = pygame.Vector2(0, 0)

        self.frames: tuple[pygame.Surface, ...] | None = None
        self.sprite: pygame.Surface | None = None
        if enemy_def.frames:
            scaled: list[pygame.Surface] = []
            for frame in enemy_def.frames:
                if frame.get_size() == (self.size, self.size):
                    scaled.append(frame)
                else:
                    scaled.append(pygame.transform.scale(frame, (self.size, self.size)))
            self.frames = tuple(scaled)
            self.sprite = self.frames[0] if self.frames else None

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x), int(self.pos.y), self.size, self.size)

    def center(self) -> pygame.Vector2:
        r = self.rect()
        return pygame.Vector2(r.centerx, r.centery)

    def update_visual(self, dt: float) -> None:
        if not self.frames or len(self.frames) <= 1:
            return
        self.anim_time += dt
        rate = 4.5 if self.enemy_def.id == "slime" else 6.0
        idx = int(self.anim_time * rate) % len(self.frames)
        self.sprite = self.frames[idx]


class Projectile:
    def __init__(
        self,
        pos: pygame.Vector2,
        vel: pygame.Vector2,
        *,
        size: int,
        color: tuple[int, int, int],
        damage: float,
        pierce: int = 0,
        knockback: float = 0.0,
        aoe_radius: float = 0.0,
        aoe_damage_mult: float = 1.0,
        ttl_s: float = 3.0,
    ) -> None:
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = int(size)
        self.color = color
        self.damage = float(damage)
        self.pierce = int(pierce)
        self.knockback = float(knockback)
        self.aoe_radius = float(aoe_radius)
        self.aoe_damage_mult = float(aoe_damage_mult)
        self.ttl_s = float(ttl_s)
        self.hit_uids: set[int] = set()

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x), int(self.pos.y), self.size, self.size)


class Swing:
    def __init__(self, rect: pygame.Rect, damage: float, ttl_s: float = 0.12) -> None:
        self.rect = rect
        self.damage = float(damage)
        self.ttl_s = float(ttl_s)
        self.hit_uids: set[int] = set()


class WeaponInstance:
    def __init__(self, weapon_def: WeaponDef) -> None:
        self.weapon_def = weapon_def
        self.level = 1
        self.cooldown_left = 0.0
        self.angle = random.random() * math.tau
        self.orbit_radius = 18.0
        self.orbit_speed = 2.6
        self.size = 8
        self.orbit_rect = pygame.Rect(0, 0, self.size, self.size)
        self.aim_angle = 0.0
        self.ammo = max(0, int(weapon_def.magazine))
        self.reload_left = 0.0
        self.reload_total = 0.0
        self.reload_spin = 0.0
        self.recoil = 0.0
        self.muzzle_flash_s = 0.0
        self.swinging = False
        self.swing_t = 0.0
        self.swing_duration_s = 0.16
        self.swing_a0 = 0.0
        self.swing_a1 = 0.0
        self.swing_hit_uids: set[int] = set()
        self.sword_poly: list[tuple[int, int]] = []
        self.sword_base = pygame.Vector2(0, 0)
        self.sword_tip = pygame.Vector2(0, 0)
        self.trail: list[tuple[pygame.Vector2, float]] = []

    def level_bonus(self) -> float:
        return 1.0 + 0.25 * (self.level - 1)

    def cooldown_bonus(self) -> float:
        return 0.92 ** (self.level - 1)

    def damage(self, player: Player) -> float:
        return float(self.weapon_def.damage) * self.level_bonus() * player.damage_mult

    def cooldown_s(self, player: Player) -> float:
        return max(0.08, float(self.weapon_def.cooldown_s) * self.cooldown_bonus() * player.cooldown_mult)

    def update(self, dt: float, player: Player, game: "GameState") -> None:
        if self.weapon_def.kind == "melee":
            self.cooldown_left -= dt

            if self.trail:
                updated: list[tuple[pygame.Vector2, float]] = []
                for pos, age in self.trail:
                    age += dt
                    if age <= 0.18:
                        updated.append((pos, age))
                self.trail = updated

            length = float(12 + max(0, self.level - 1) * 2)
            thickness = float(4 + max(0, self.level - 1) // 2)
            hand = player.hand_pos()

            if not self.swinging and self.cooldown_left <= 0.0:
                self.swinging = True
                self.swing_t = 0.0
                self.swing_hit_uids.clear()
                self.cooldown_left = self.cooldown_s(player)
                game.app.play_sfx("swing")

                base = 0.0 if player.facing >= 0 else math.pi
                arc = 1.08
                self.swing_a0 = base - arc
                self.swing_a1 = base + arc

            if self.swinging:
                self.swing_t += dt
                t = 1.0 if self.swing_duration_s <= 0 else clamp(self.swing_t / self.swing_duration_s, 0.0, 1.0)
                t = t * t * (3.0 - 2.0 * t)
                ang = self.swing_a0 + (self.swing_a1 - self.swing_a0) * t
                if self.swing_t >= self.swing_duration_s:
                    self.swinging = False
            else:
                ang = 0.12 if player.facing >= 0 else math.pi - 0.12

            d = pygame.Vector2(math.cos(ang), math.sin(ang))
            if d.length_squared() <= 0.001:
                d = pygame.Vector2(1, 0 if player.facing >= 0 else -1)
            perp = pygame.Vector2(-d.y, d.x)
            half = thickness * 0.5

            base = pygame.Vector2(hand)
            tip = base + d * length
            p1 = base + perp * half
            p2 = base - perp * half
            p3 = tip - perp * half
            p4 = tip + perp * half

            self.sword_base = base
            self.sword_tip = tip
            self.sword_poly = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y)), (int(p4.x), int(p4.y))]

            min_x = int(min(p1.x, p2.x, p3.x, p4.x))
            min_y = int(min(p1.y, p2.y, p3.y, p4.y))
            max_x = int(max(p1.x, p2.x, p3.x, p4.x))
            max_y = int(max(p1.y, p2.y, p3.y, p4.y))
            self.orbit_rect = pygame.Rect(min_x, min_y, max(1, max_x - min_x + 1), max(1, max_y - min_y + 1))

            if self.swinging:
                if not self.trail or (self.trail[-1][0] - tip).length_squared() > 1.0:
                    self.trail.append((pygame.Vector2(tip), 0.0))

                for e in game.enemies:
                    if e.uid in self.swing_hit_uids:
                        continue
                    if not self.orbit_rect.colliderect(e.rect()):
                        continue
                    self.swing_hit_uids.add(e.uid)
                    dmg = self.damage(player)
                    e.hp -= dmg
                    game.add_damage_text(e.center(), dmg, color=(255, 240, 200))

                    knock = 180.0 if e.enemy_def.id == "slime" else 90.0
                    away = (e.center() - player.center())
                    if away.length_squared() > 0.01:
                        e.vel += away.normalize() * knock
                    else:
                        rand = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                        if rand.length_squared() <= 0.001:
                            rand = pygame.Vector2(1, 0)
                        e.vel += rand.normalize() * knock
            return
        c = player.orbit_center()
        offset = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * self.orbit_radius
        pos = c + offset

        if self.weapon_def.kind == "gun":
            self.muzzle_flash_s = max(0.0, self.muzzle_flash_s - dt)
            size = int(10 + max(0, self.level - 1) // 2)
            self.orbit_rect.size = (size, size)
            self.orbit_rect.center = (int(pos.x), int(pos.y))

            target = game.closest_enemy_to(pos)
            if target is not None:
                aim = (target.center() - pos)
                if aim.length_squared() > 0.01:
                    aim = aim.normalize()
                else:
                    aim = pygame.Vector2(1, 0)
            else:
                aim = pygame.Vector2(player.last_move_dir)
                if aim.length_squared() <= 0.01:
                    aim = pygame.Vector2(1, 0)
                else:
                    aim = aim.normalize()
            self.aim_angle = math.atan2(aim.y, aim.x)

            if self.weapon_def.recoil_decay > 0:
                self.recoil = max(0.0, self.recoil - float(self.weapon_def.recoil_decay) * dt)

            mag = max(1, int(self.weapon_def.magazine + max(0, self.level - 1) // 3))
            reload_s = max(0.35, float(self.weapon_def.reload_s) * (0.96 ** max(0, self.level - 1)))

            if self.reload_left > 0.0:
                self.reload_left = max(0.0, self.reload_left - dt)
                self.reload_spin = (self.reload_spin + dt * 11.0) % math.tau
                if self.reload_left <= 0.0:
                    self.ammo = mag
                    self.reload_total = 0.0
                    self.reload_spin = 0.0
                    self.cooldown_left = max(self.cooldown_left, 0.06)
                return

            self.cooldown_left -= dt
            if self.cooldown_left > 0.0:
                return

            if self.ammo <= 0:
                self.reload_total = reload_s
                self.reload_left = reload_s
                self.reload_spin = 0.0
                return

            spread = float(self.weapon_def.spread_rad) + float(self.recoil)
            base_kb = float(self.weapon_def.knockback)
            if base_kb <= 0.0:
                base_kb = 70.0 + float(self.weapon_def.damage) * 4.0
            knock = base_kb * (1.0 + 0.10 * max(0, self.level - 1))

            bullet_speed = max(10.0, float(self.weapon_def.proj_speed))
            shots = max(1, int(self.weapon_def.shots_per_fire))
            for _ in range(shots):
                a = self.aim_angle
                if spread > 0:
                    a += random.uniform(-spread, spread)
                d = pygame.Vector2(math.cos(a), math.sin(a))
                if d.length_squared() <= 0.001:
                    d = pygame.Vector2(1, 0)

                game.projectiles.append(
                    Projectile(
                        pos,
                        d * bullet_speed,
                        size=max(1, int(self.weapon_def.proj_size)),
                        color=self.weapon_def.color,
                        damage=self.damage(player),
                        pierce=int(self.weapon_def.pierce),
                        knockback=knock,
                        aoe_radius=float(self.weapon_def.aoe_radius),
                        aoe_damage_mult=float(self.weapon_def.aoe_damage_mult),
                        ttl_s=float(self.weapon_def.proj_ttl_s),
                    )
                )
            game.app.play_sfx("shot")
            self.muzzle_flash_s = 0.06 if self.weapon_def.aoe_radius <= 0.0 else 0.10
            if self.weapon_def.recoil_per_shot > 0 and self.weapon_def.recoil_max > 0:
                self.recoil = min(float(self.weapon_def.recoil_max), self.recoil + float(self.weapon_def.recoil_per_shot))

            self.ammo -= 1
            self.cooldown_left = self.cooldown_s(player)
            if self.ammo <= 0:
                self.reload_total = reload_s
                self.reload_left = reload_s
                self.reload_spin = 0.0
            return

        size = int(self.size + max(0, self.level - 1))
        self.orbit_rect.size = (size, size)
        self.orbit_rect.center = (int(pos.x), int(pos.y))

        self.cooldown_left -= dt
        if self.cooldown_left > 0:
            return

        for e in game.enemies:
            if self.orbit_rect.colliderect(e.rect()):
                dmg = self.damage(player)
                e.hp -= dmg
                game.add_damage_text(e.center(), dmg, color=(220, 240, 255))
                self.cooldown_left = self.cooldown_s(player)
                break


@dataclass
class WeaponPickup:
    weapon_id: str
    pos: pygame.Vector2
    bob_t: float = 0.0

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x), int(self.pos.y), 10, 10)

    def level_bonus(self) -> float:
        return 1.0 + 0.25 * (self.level - 1)

    def cooldown_bonus(self) -> float:
        return 0.92 ** (self.level - 1)

    def damage(self, player: Player) -> float:
        return float(self.weapon_def.damage) * self.level_bonus() * player.damage_mult

    def cooldown_s(self, player: Player) -> float:
        return max(0.08, float(self.weapon_def.cooldown_s) * self.cooldown_bonus() * player.cooldown_mult)

    def update(self, dt: float, player: Player, game: "GameState") -> None:     
        if self.weapon_def.kind == "melee":
            self.cooldown_left -= dt

            if self.trail:
                updated: list[tuple[pygame.Vector2, float]] = []
                for pos, age in self.trail:
                    age += dt
                    if age <= 0.18:
                        updated.append((pos, age))
                self.trail = updated

            length = float(12 + max(0, self.level - 1) * 2)
            thickness = float(4 + max(0, self.level - 1) // 2)
            hand = player.hand_pos()

            if not self.swinging and self.cooldown_left <= 0.0:
                self.swinging = True
                self.swing_t = 0.0
                self.swing_hit_uids.clear()
                self.cooldown_left = self.cooldown_s(player)

                base = 0.0 if player.facing >= 0 else math.pi
                arc = 1.08
                self.swing_a0 = base - arc
                self.swing_a1 = base + arc

            if self.swinging:
                self.swing_t += dt
                t = 1.0 if self.swing_duration_s <= 0 else clamp(self.swing_t / self.swing_duration_s, 0.0, 1.0)
                t = t * t * (3.0 - 2.0 * t)
                ang = self.swing_a0 + (self.swing_a1 - self.swing_a0) * t
                if self.swing_t >= self.swing_duration_s:
                    self.swinging = False
            else:
                ang = 0.12 if player.facing >= 0 else math.pi - 0.12

            d = pygame.Vector2(math.cos(ang), math.sin(ang))
            if d.length_squared() <= 0.001:
                d = pygame.Vector2(1, 0 if player.facing >= 0 else -1)
            perp = pygame.Vector2(-d.y, d.x)
            half = thickness * 0.5

            base = pygame.Vector2(hand)
            tip = base + d * length
            p1 = base + perp * half
            p2 = base - perp * half
            p3 = tip - perp * half
            p4 = tip + perp * half

            self.sword_base = base
            self.sword_tip = tip
            self.sword_poly = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y)), (int(p4.x), int(p4.y))]

            min_x = int(min(p1.x, p2.x, p3.x, p4.x))
            min_y = int(min(p1.y, p2.y, p3.y, p4.y))
            max_x = int(max(p1.x, p2.x, p3.x, p4.x))
            max_y = int(max(p1.y, p2.y, p3.y, p4.y))
            self.orbit_rect = pygame.Rect(min_x, min_y, max(1, max_x - min_x + 1), max(1, max_y - min_y + 1))

            if self.swinging:
                if not self.trail or (self.trail[-1][0] - tip).length_squared() > 1.0:
                    self.trail.append((pygame.Vector2(tip), 0.0))

                for e in game.enemies:
                    if e.uid in self.swing_hit_uids:
                        continue
                    if not self.orbit_rect.colliderect(e.rect()):
                        continue
                    self.swing_hit_uids.add(e.uid)
                    e.hp -= self.damage(player)

                    knock = 180.0 if e.enemy_def.id == "slime" else 90.0
                    away = (e.center() - player.center())
                    if away.length_squared() > 0.01:
                        e.vel += away.normalize() * knock
                    else:
                        rand = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                        if rand.length_squared() <= 0.001:
                            rand = pygame.Vector2(1, 0)
                        e.vel += rand.normalize() * knock
            return

        self.angle = (self.angle + dt * self.orbit_speed) % math.tau
        c = player.orbit_center()
        offset = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * self.orbit_radius
        pos = c + offset

        size = int(self.size + max(0, self.level - 1))
        self.orbit_rect.size = (size, size)
        self.orbit_rect.center = (int(pos.x), int(pos.y))

        self.cooldown_left -= dt
        if self.cooldown_left > 0:
            return

        for e in game.enemies:
            if self.orbit_rect.colliderect(e.rect()):
                e.hp -= self.damage(player)
                self.cooldown_left = self.cooldown_s(player)
                break


@dataclass
class WeaponDrop:
    weapon_id: str
    pos: pygame.Vector2
    bob_t: float = 0.0

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x) - 4, int(self.pos.y) - 4, 8, 8)


class DamageText:
    def __init__(self, pos: pygame.Vector2, amount: float, *, color: pygame.Color) -> None:
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(random.uniform(-10.0, 10.0), random.uniform(-28.0, -40.0))
        self.amount = float(amount)
        self.text = str(max(0, int(round(amount))))
        self.color = color
        self.ttl_s = 0.60


class ExplosionFx:
    def __init__(self, pos: pygame.Vector2, *, radius: float, color: tuple[int, int, int]) -> None:
        self.pos = pygame.Vector2(pos)
        self.radius = float(radius)
        self.color = pygame.Color(*color)
        self.ttl_s = 0.22


def make_defs() -> tuple[list[CharacterDef], list[MapDef], dict[str, EnemyDef], dict[str, WeaponDef]]:
    palette = {
        "K": (16, 16, 16),
        "W": (240, 240, 240),
        "G": (96, 192, 96),
        "B": (96, 160, 220),
        "Y": (220, 200, 80),
        "P": (180, 120, 220),
        "S": (220, 180, 140),
        "D": (34, 34, 44),
        "M": (120, 130, 150),
        "L": (170, 180, 205),
        "H": (235, 235, 245),
        "O": (70, 70, 82),
        "R": (240, 120, 80),
    }

    warrior_frames = (
        sprite_from_pixels(
            [
                "    KKK     ",
                "   KWWWWK   ",
                "   KWOOWK   ",
                "   KSSSSK   ",
                "   KSSSSK   ",
                "   KWWWWK   ",
                "  KKWWWWKK  ",
                "  KWWYYWWK  ",
                "   KWWWWK   ",
                "   KWWWWK   ",
                "   KYY YYK  ",
                "   KY   YK  ",
                "   KY   YK  ",
                "  KYY  YYK  ",
            ],
            palette,
        ),
        sprite_from_pixels(
            [
                "    KKK     ",
                "   KWWWWK   ",
                "   KWOOWK   ",
                "   KSSSSK   ",
                "   KSSSSK   ",
                "   KWWWWK   ",
                "  KKWWWWKK  ",
                "  KWWYYWWK  ",
                "   KWWWWK   ",
                "   KWWWWK   ",
                "   KYY YYK  ",
                "   KYY  YK  ",
                "   KY  YYK  ",
                "  KYY   K   ",
            ],
            palette,
        ),
        sprite_from_pixels(
            [
                "    KKK     ",
                "   KWWWWK   ",
                "   KWOOWK   ",
                "   KSSSSK   ",
                "   KSSSSK   ",
                "   KWWWWK   ",
                "  KKWWWWKK  ",
                "  KWWYYWWK  ",
                "   KWWWWK   ",
                "   KWWWWK   ",
                "   KYY YYK  ",
                "   KY  YYK  ",
                "   KYY  YK  ",
                "  K   YYK   ",
            ],
            palette,
        ),
    )
    warrior = warrior_frames[0]
    warrior_skeleton = (
        {
            "head": (5, 3),
            "chest": (5, 7),
            "hip": (5, 10),
            "l_hand": (3, 8),
            "r_hand": (7, 8),
            "l_foot": (4, 13),
            "r_foot": (7, 13),
        },
        {
            "head": (5, 3),
            "chest": (5, 7),
            "hip": (5, 10),
            "l_hand": (3, 8),
            "r_hand": (7, 8),
            "l_foot": (4, 13),
            "r_foot": (8, 12),
        },
        {
            "head": (5, 3),
            "chest": (5, 7),
            "hip": (5, 10),
            "l_hand": (3, 8),
            "r_hand": (7, 8),
            "l_foot": (5, 12),
            "r_foot": (7, 13),
        },
    )
    mage = sprite_from_pixels(
        [
            "    PP      ",
            "   PPPP     ",
            "   PSSP     ",
            "   PWWP     ",
            "  PPPPPP    ",
            "  PWWWWP    ",
            "  PWWWWP    ",
            "   PBBP     ",
            "   PBBP     ",
            "    BB      ",
            "    BB      ",
            "            ",
        ],
        palette,
    )
    dragonkin = sprite_from_pixels(
        [
            "   GGGG     ",
            "  GGGGGG    ",
            "  GSSSSG    ",
            "  GKKKKG    ",
            "  GGGGGG    ",
            "  GGGGGG    ",
            "  GGYYGG    ",
            "   GYYG     ",
            "   G  G     ",
            "  GG  GG    ",
            "  G    G    ",
            "            ",
        ],
        palette,
    )
    undead = sprite_from_pixels(
        [
            "   GGGG     ",
            "  GWWWWG    ",
            "  GWKKWG    ",
            "  GWWWWG    ",
            "  GGGGGG    ",
            "  GWWWWG    ",
            "  GWWWWG    ",
            "   GWWG     ",
            "   G  G     ",
            "  GG  GG    ",
            "  G    G    ",
            "            ",
        ],
        palette,
    )
    elf = sprite_from_pixels(
        [
            "   G  G     ",
            "  GGGGGG    ",
            "  GSSSSG    ",
            "  GWWWWG    ",
            "  GGGGGG    ",
            "  GWWWWG    ",
            "  GWWWWG    ",
            "   GGGG     ",
            "   G  G     ",
            "  GG  GG    ",
            "  G    G    ",
            "            ",
        ],
        palette,
    )

    chars = [
        CharacterDef(
            id="warrior",
            name="战士",
            desc="高生命，近战起手。",
            sprite=warrior,
            base_hp=120,
            move_speed=92.0,
            start_weapon="sword",
            regen=0.25,
            damage_mult=1.05,
            frames=warrior_frames,
            skeleton_frames=warrior_skeleton,
        ),
        CharacterDef(
            id="mage",
            name="法师",
            desc="低生命，远程爆发。",
            sprite=mage,
            base_hp=90,
            move_speed=90.0,
            start_weapon="sword",
            regen=0.15,
            cooldown_mult=0.92,
        ),
        CharacterDef(
            id="dragonkin",
            name="龙族",
            desc="坚韧，火焰压制。",
            sprite=dragonkin,
            base_hp=130,
            move_speed=86.0,
            start_weapon="sword",
            regen=0.22,
            damage_mult=1.08,
        ),
        CharacterDef(
            id="undead",
            name="不死族",
            desc="强回复，拖时间更强。",
            sprite=undead,
            base_hp=110,
            move_speed=88.0,
            start_weapon="sword",
            regen=0.45,
            damage_mult=0.95,
        ),
        CharacterDef(
            id="elf",
            name="精灵族",
            desc="敏捷，射速优秀。",
            sprite=elf,
            base_hp=95,
            move_speed=104.0,
            start_weapon="sword",
            regen=0.18,
            cooldown_mult=0.90,
        ),
    ]

    maps = [
        MapDef(
            id="grass",
            name="草原",
            desc="绿意盎然，史莱姆来袭。",
            base_color=(56, 120, 64),
            accent_color=(84, 180, 96),
            enemy_pool=("slime", "slime", "runner"),
        ),
        MapDef(
            id="ocean",
            name="海洋",
            desc="潮汐拍岸，鱼群缠斗。",
            base_color=(20, 70, 120),
            accent_color=(60, 140, 200),
            enemy_pool=("fish", "fish", "puffer"),
        ),
        MapDef(
            id="grave",
            name="墓地",
            desc="阴森幽暗，亡者苏醒。",
            base_color=(38, 38, 44),
            accent_color=(120, 120, 140),
            enemy_pool=("skeleton", "skeleton", "wraith"),
        ),
    ]

    def squish_frame(base: pygame.Surface, sx: float, sy: float) -> pygame.Surface:
        w, h = base.get_size()
        sw = max(1, int(round(w * sx)))
        sh = max(1, int(round(h * sy)))
        scaled = pygame.transform.scale(base, (sw, sh))
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        out.fill((0, 0, 0, 0))
        out.blit(scaled, scaled.get_rect(center=(w // 2, h // 2)))
        return out

    slime_base = sprite_from_pixels(
        [
            "   KKKK   ",
            "  KGGGGK  ",
            " KGGGGGGK ",
            " KGGW WGGK",
            " KGGGGGGK ",
            "  KGGGGK  ",
            "   KGGK   ",
            "    KK    ",
            "          ",
            "          ",
        ],
        palette,
    )
    slime_frames = (
        slime_base,
        squish_frame(slime_base, 1.10, 0.92),
        squish_frame(slime_base, 0.92, 1.10),
    )
    runner = sprite_from_pixels(
        [
            "  K  K  ",
            "  KYYK  ",
            " KYYYYK ",
            " KYYWYYK",
            "  KYYYYK",
            "   KYYK ",
            "   K  K ",
            "        ",
        ],
        palette,
    )
    fish = sprite_from_pixels(
        [
            "          ",
            "   KKKK   ",
            "  KBBBBKK ",
            " KBBBBBBKK",
            "KBBBBWBBKK",
            " KBBBBBBKK",
            "  KBBBBKK ",
            "   KKKK   ",
            "          ",
            "          ",
        ],
        palette,
    )
    puffer = sprite_from_pixels(
        [
            "     K      ",
            "   KKKKK    ",
            "  KKYYYYKK  ",
            " KKYYYYYYKK ",
            "KKYYYYYYYYKK",
            "KKYYWYYWYYKK",
            "KKYYYYYYYYKK",
            " KKYYYYYYKK ",
            "  KKYYYYKK  ",
            "   KKKKK    ",
            "     K      ",
            "            ",
        ],
        palette,
    )
    skeleton = sprite_from_pixels(
        [
            "   KKKK   ",
            "  KWWWWK  ",
            " KWWKKWWK ",
            " KWWWWWWK ",
            "  KWWWWK  ",
            "   KWWK   ",
            "  KWWWWK  ",
            "  KW  WK  ",
            "   K  K   ",
            "  K    K  ",
        ],
        palette,
    )
    wraith = sprite_from_pixels(
        [
            "   KKK   ",
            "  KPPPK  ",
            " KPPPPPK ",
            " KPW WPK ",
            " KPPPPPK ",
            "  KPPPK  ",
            "  KPPPK  ",
            "  KPKPK  ",
            "   K K   ",
        ],
        palette,
    )

    enemies = {
        "slime": EnemyDef("slime", "史莱姆", (90, 210, 120), hp=18, speed=34.0, size=10, contact_damage=10, xp_value=3, frames=slime_frames),
        "runner": EnemyDef("runner", "野兔", (200, 220, 120), hp=12, speed=58.0, size=8, contact_damage=8, xp_value=3, frames=(runner,)),
        "fish": EnemyDef("fish", "小鱼", (90, 180, 230), hp=16, speed=40.0, size=10, contact_damage=10, xp_value=3, frames=(fish,)),
        "puffer": EnemyDef("puffer", "河豚", (220, 210, 110), hp=28, speed=26.0, size=12, contact_damage=14, xp_value=5, frames=(puffer,)),
        "skeleton": EnemyDef("skeleton", "骷髅", (210, 210, 220), hp=20, speed=36.0, size=10, contact_damage=12, xp_value=4, frames=(skeleton,)),
        "wraith": EnemyDef("wraith", "幽魂", (170, 120, 220), hp=14, speed=62.0, size=9, contact_damage=10, xp_value=4, frames=(wraith,)),
    }

    pistol_sprite = sprite_from_pixels(
        [
            "     DDDDDDD",
            "DDDDDLLLLLDDD",
            "DDLLHHHHHLLDD",
            "DDLLLMMMMMLDD",
            "   DOOOODDD",
            "   DOOOOD",
            "    DDDD",
        ],
        palette,
    )
    uzi_sprite = sprite_from_pixels(
        [
            "      DDDDDDDDDDD",
            "DDDDDDLLLLLLLLLDDD",
            "DDLLHHHHHHHHHLLDD",
            "DDLLLMMMMMMMMMLDD",
            "   DOOOOODDD",
            "   DOOOOOD",
            "    DDDD",
        ],
        palette,
    )
    ak47_sprite = sprite_from_pixels(
        [
            "      DDDDDDDDDDDDDDDD",
            "DDDDDDLLLLLLLLLLLLLLLDDD",
            "DDLLHHHHHHHHHHHHHHLLDD",
            "DDLLLMMMMMMMMMMMMMMLDD",
            "DDDDDOOOOODDDDDDD",
            "    OOOO",
            "    OOO",
            "    DDD",
        ],
        palette,
    )
    scar_sprite = sprite_from_pixels(
        [
            "   DDDDDDDDDDDDDDDDDDD",
            "DDDDDLLLLLLLLLLLLLLLLDDDD",
            "DDLLHHHHHHHHHHHHHHLLDD",
            "DDLLLMMMMMMMMMMMMMMLDD",
            "DDDDDOOOO OOOODDDD",
            "     OOOO",
            "     OOOO",
            "      DDD",
        ],
        palette,
    )
    sniper_sprite = sprite_from_pixels(
        [
            "        HHHH",
            "DDDDDDDLLLLLLLLLLLLLLLLLLDDDD",
            "DDLLHHHHHHHHHHHHHHHHHHLLDD",
            "DDLLLMMMMMMMMMMMMMMMMMMLDD",
            "      OOOO",
            "      OOO",
            "       DDD",
        ],
        palette,
    )
    rocket_sprite = sprite_from_pixels(
        [
            "       RRRR",
            "DDDDDDDLLLLLLLLLLLLDDDDRR",
            "DDLLHHHHHHHHHHHHLLDDRRR",
            "DDLLLMMMMMMMMMMMMLDDDD",
            "    OOOO",
            "    OOO",
            "     DDD",
        ],
        palette,
    )

    weapons = {
        "sword": WeaponDef("sword", "短剑", "近战挥砍。", "melee", (220, 220, 220), cooldown_s=0.45, damage=10),
        "wand": WeaponDef("wand", "法杖", "环绕法杖，触碰造成伤害。", "orbit", (120, 170, 240), cooldown_s=0.35, damage=7, proj_speed=240, proj_size=3),
        "bow": WeaponDef("bow", "弓", "环绕武器，触碰造成伤害。", "orbit", (220, 190, 120), cooldown_s=0.30, damage=6, proj_speed=290, proj_size=2),
        "fireball": WeaponDef("fireball", "火球", "环绕火球，触碰造成伤害。", "orbit", (240, 120, 80), cooldown_s=0.65, damage=12, proj_speed=210, proj_size=4),
        "spear": WeaponDef("spear", "骨矛", "环绕骨矛，触碰造成伤害。", "orbit", (235, 235, 255), cooldown_s=0.75, damage=9, proj_speed=320, proj_size=3, pierce=2),
        "dagger": WeaponDef("dagger", "匕首", "环绕匕首，触碰造成伤害。", "orbit", (200, 200, 200), cooldown_s=0.22, damage=4, proj_speed=330, proj_size=2),
        "pistol": WeaponDef(
            "pistol",
            "手枪",
            "稳健点射；换弹时武器旋转。",
            "gun",
            (230, 230, 200),
            cooldown_s=0.20,
            damage=7,
            proj_speed=380,
            proj_size=2,
            magazine=8,
            reload_s=1.10,
            proj_ttl_s=1.4,
            spread_rad=0.06,
            knockback=160.0,
            recoil_per_shot=0.030,
            recoil_decay=0.20,
            recoil_max=0.18,
            sprite=pistol_sprite,
        ),
        "uzi": WeaponDef(
            "uzi",
            "Uzi",
            "高射速扫射；后坐力更大。",
            "gun",
            (190, 230, 230),
            cooldown_s=0.060,
            damage=3,
            proj_speed=420,
            proj_size=1,
            magazine=28,
            reload_s=1.60,
            proj_ttl_s=1.1,
            spread_rad=0.14,
            knockback=85.0,
            recoil_per_shot=0.050,
            recoil_decay=0.18,
            recoil_max=0.36,
            sprite=uzi_sprite,
        ),
        "ak47": WeaponDef(
            "ak47",
            "AK47",
            "中距离连发，伤害与稳定性均衡。",
            "gun",
            (230, 190, 140),
            cooldown_s=0.11,
            damage=5,
            proj_speed=450,
            proj_size=2,
            magazine=30,
            reload_s=1.85,
            proj_ttl_s=1.4,
            spread_rad=0.10,
            knockback=120.0,
            recoil_per_shot=0.040,
            recoil_decay=0.20,
            recoil_max=0.30,
            sprite=ak47_sprite,
        ),
        "scarl": WeaponDef(
            "scarl",
            "SCAR-L",
            "更稳更准的步枪，单发更强。",
            "gun",
            (200, 230, 170),
            cooldown_s=0.12,
            damage=6,
            proj_speed=470,
            proj_size=2,
            magazine=20,
            reload_s=1.95,
            proj_ttl_s=1.5,
            spread_rad=0.08,
            knockback=135.0,
            recoil_per_shot=0.034,
            recoil_decay=0.22,
            recoil_max=0.26,
            sprite=scar_sprite,
        ),
        "sniper": WeaponDef(
            "sniper",
            "狙击步枪",
            "超高伤害与穿透；射速慢，换弹慢。",
            "gun",
            (230, 230, 240),
            cooldown_s=0.85,
            damage=28,
            proj_speed=700,
            proj_size=2,
            pierce=3,
            magazine=5,
            reload_s=2.60,
            proj_ttl_s=1.8,
            spread_rad=0.01,
            knockback=260.0,
            recoil_per_shot=0.018,
            recoil_decay=0.45,
            recoil_max=0.07,
            sprite=sniper_sprite,
        ),
        "rocket": WeaponDef(
            "rocket",
            "火箭筒",
            "爆炸范围伤害与强击退；弹匣小。",
            "gun",
            (240, 120, 80),
            cooldown_s=1.25,
            damage=18,
            proj_speed=220,
            proj_size=4,
            magazine=1,
            reload_s=2.35,
            proj_ttl_s=2.4,
            spread_rad=0.04,
            knockback=320.0,
            aoe_radius=28.0,
            aoe_damage_mult=1.35,
            recoil_per_shot=0.020,
            recoil_decay=0.30,
            recoil_max=0.10,
            sprite=rocket_sprite,
        ),
    }

    return chars, maps, enemies, weapons


CHARS, MAPS, ENEMY_DEFS, WEAPON_DEFS = make_defs()


class State:
    def __init__(self, app: "App") -> None:
        self.app = app

    def on_enter(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pass


class App:
    def __init__(self) -> None:
        try:
            pygame.mixer.pre_init(22050, -16, 1, 256)
        except Exception:
            pass
        pygame.init()
        pygame.font.init()
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(22050, -16, 1, 256)
            pygame.mixer.set_num_channels(16)
        except Exception:
            pass
        self.sfx = SFX()

        self.config = GameConfig.load()
        self.config.scale = int(clamp(self.config.scale, 1, 6))

        self.clock = pygame.time.Clock()
        self.render = pygame.Surface((INTERNAL_W, INTERNAL_H))
        self._scaled: pygame.Surface | None = None
        self._grid_overlay: pygame.Surface | None = None
        self._grid_overlay_key: tuple[int, int] | None = None

        pygame.display.set_caption("幸存者 Demo (pygame)")
        self._apply_display()

        font_path = pick_ui_font_path()
        self.font_s = pygame.font.Font(font_path, 12) if font_path else pygame.font.Font(None, 12)
        self.font_m = pygame.font.Font(font_path, 16) if font_path else pygame.font.Font(None, 16)
        self.font_l = pygame.font.Font(font_path, 28) if font_path else pygame.font.Font(None, 28)

        self.state: State = StartScreenState(self)
        self.state.on_enter()

    def play_sfx(self, name: str) -> None:
        self.sfx.play(name)

    def _apply_display(self) -> None:
        if self.config.fullscreen:
            # 桌面全屏：窗口最大化（保留桌面环境/任务栏/边框），不使用独占全屏
            self.screen = pygame.display.set_mode(
                (INTERNAL_W * self.config.scale, INTERNAL_H * self.config.scale),
                pygame.RESIZABLE,
            )
            self._maximize_window()
        else:
            self.screen = pygame.display.set_mode((INTERNAL_W * self.config.scale, INTERNAL_H * self.config.scale))
        self._scaled = None
        self._update_view_transform()

    def _maximize_window(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            pygame.event.pump()
            hwnd = pygame.display.get_wm_info().get("window")
            if not hwnd:
                return
            SW_MAXIMIZE = 3
            ctypes.windll.user32.ShowWindow(int(hwnd), SW_MAXIMIZE)
        except Exception:
            return

    def _update_view_transform(self) -> None:
        sw, sh = self.screen.get_size()
        sw = max(1, int(sw))
        sh = max(1, int(sh))
        # Keep a consistent INTERNAL_W x INTERNAL_H aspect ratio.
        # Prefer integer scaling to keep pixel sizes consistent; fall back to
        # fractional scaling only when the window is smaller than the internal
        # resolution.
        fit_scale = min(sw / max(1, INTERNAL_W), sh / max(1, INTERNAL_H))
        if fit_scale >= 1.0:
            scale = float(max(1, int(fit_scale)))
        else:
            scale = float(fit_scale)

        dest_w = max(1, int(round(INTERNAL_W * scale)))
        dest_h = max(1, int(round(INTERNAL_H * scale)))
        offset_x = max(0, (sw - dest_w) // 2)
        offset_y = max(0, (sh - dest_h) // 2)
        dest_size = (dest_w, dest_h)

        self.view_scale_x = dest_w / max(1, INTERNAL_W)
        self.view_scale_y = dest_h / max(1, INTERNAL_H)
        self.view_offset = (offset_x, offset_y)
        self.view_rect = pygame.Rect(offset_x, offset_y, dest_w, dest_h)
        if self._scaled is None or self._scaled.get_size() != dest_size:
            self._scaled = pygame.Surface(dest_size)

        if not self.config.show_grid:
            self._grid_overlay = None
            self._grid_overlay_key = None
            return

        if self._grid_overlay is not None and self._grid_overlay_key == (dest_size[0], dest_size[1]):
            return

        overlay = pygame.Surface(dest_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        w, h = dest_size
        cell_px = min(self.view_scale_x, self.view_scale_y)
        alpha = 18 if cell_px <= 2.5 else 26 if cell_px <= 4.5 else 34
        minor = (0, 0, 0, alpha)

        if w > 1 and h > 1:
            # Draw the full INTERNAL_W x INTERNAL_H grid.
            for x_cell in range(INTERNAL_W + 1):
                x = (x_cell * (w - 1)) // INTERNAL_W
                pygame.draw.line(overlay, minor, (x, 0), (x, h - 1))
            for y_cell in range(INTERNAL_H + 1):
                y = (y_cell * (h - 1)) // INTERNAL_H
                pygame.draw.line(overlay, minor, (0, y), (w - 1, y))

        # Major marker lines for quick counting (10/50/100/200/300).
        major_alpha = min(120, alpha * 4)
        major = (255, 220, 100, major_alpha)
        major_w = 1 if cell_px <= 3.5 else 2
        for x_cell in GRID_MARKS:
            if not (0 <= x_cell <= INTERNAL_W):
                continue
            x = (x_cell * (w - 1)) // INTERNAL_W
            if 0 <= x < w:
                pygame.draw.line(overlay, major, (x, 0), (x, h - 1), major_w)
        for y_cell in GRID_MARKS:
            if not (0 <= y_cell <= INTERNAL_H):
                continue
            y = (y_cell * (h - 1)) // INTERNAL_H
            if 0 <= y < h:
                pygame.draw.line(overlay, major, (0, y), (w - 1, y), major_w)

        self._grid_overlay = overlay
        self._grid_overlay_key = (dest_size[0], dest_size[1])

    def screen_to_internal(self, screen_pos: tuple[int, int]) -> tuple[int, int] | None:
        if not hasattr(self, "view_rect"):
            self._update_view_transform()
        if not self.view_rect.collidepoint(screen_pos):
            return None
        w = max(1, int(self.view_rect.w))
        h = max(1, int(self.view_rect.h))
        x = int((screen_pos[0] - self.view_rect.x) * INTERNAL_W / w)
        y = int((screen_pos[1] - self.view_rect.y) * INTERNAL_H / h)
        if 0 <= x < INTERNAL_W and 0 <= y < INTERNAL_H:
            return x, y
        return None

    def set_state(self, state: State) -> None:
        self.state = state
        self.state.on_enter()

    def set_config(self, *, scale: int | None = None, fullscreen: bool | None = None, show_grid: bool | None = None) -> None:
        if scale is not None:
            self.config.scale = int(clamp(scale, 1, 6))
        if fullscreen is not None:
            self.config.fullscreen = bool(fullscreen)
        if show_grid is not None:
            self.config.show_grid = bool(show_grid)
        self.config.save()
        self._apply_display()

    def blit_scaled(self) -> None:
        self._update_view_transform()
        pygame.transform.scale(self.render, self._scaled.get_size(), self._scaled)
        if self._grid_overlay is not None:
            self._scaled.blit(self._grid_overlay, (0, 0))
            self._draw_grid_markers(self._scaled)
        self.screen.fill((0, 0, 0))
        self.screen.blit(self._scaled, getattr(self, "view_offset", (0, 0)))

    def _draw_grid_markers(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        cell_px = float(min(getattr(self, "view_scale_x", 1.0), getattr(self, "view_scale_y", 1.0)))
        ui_scale = max(1, int(round(cell_px)))
        tick_len = max(6, int(cell_px * 4))
        col = pygame.Color(255, 220, 100)
        text_col = pygame.Color(255, 240, 180)

        marks = GRID_MARKS

        def scale_text(img: pygame.Surface) -> pygame.Surface:
            if ui_scale == 1:
                return img
            return pygame.transform.scale(img, (img.get_width() * ui_scale, img.get_height() * ui_scale))

        info = f"GRID {INTERNAL_W}x{INTERNAL_H}"
        info_img = scale_text(self.font_s.render(info, False, pygame.Color(230, 230, 240)))
        info_rect = info_img.get_rect(topright=(w - 2, 2))
        surface.blit(info_img, info_rect)

        # X-axis markers (top/bottom)
        for x_cell in marks:
            if not (0 <= x_cell <= INTERNAL_W):
                continue
            x = (x_cell * (w - 1)) // INTERNAL_W
            if not (0 <= x < w):
                continue
            pygame.draw.line(surface, col, (x, 0), (x, min(h - 1, tick_len)))
            pygame.draw.line(surface, col, (x, h - 1), (x, max(0, h - 1 - tick_len)))

            label = str(x_cell)
            img = scale_text(self.font_s.render(label, False, text_col))
            lx = x + 2
            ly = 2
            if lx + img.get_width() > w - 1:
                lx = max(0, w - 1 - img.get_width())
            surface.blit(img, (lx, ly))

        # Y-axis markers (left/right)
        for y_cell in marks:
            if not (0 <= y_cell <= INTERNAL_H):
                continue
            y = (y_cell * (h - 1)) // INTERNAL_H
            if not (0 <= y < h):
                continue
            pygame.draw.line(surface, col, (0, y), (min(w - 1, tick_len), y))
            pygame.draw.line(surface, col, (w - 1, y), (max(0, w - 1 - tick_len), y))

            label = str(y_cell)
            img = scale_text(self.font_s.render(label, False, text_col))
            lx = 2
            ly = y + 2
            if ly + img.get_height() > h - 1:
                ly = max(0, y - 2 - img.get_height())
            surface.blit(img, (lx, ly))

    def run(self) -> None:
        running = True
        while running:
            # Fixed-step timing: keeps pixel motion consistent and avoids the
            # subtle "lurch" you get from variable dt + pixel snapping.
            self.clock.tick_busy_loop(FPS)
            dt = 1.0 / float(FPS)
            self._update_view_transform()
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    # Keep the display surface in sync with the resized window.
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self._scaled = None
                    self._update_view_transform()
                    continue
                if event.type == pygame.QUIT:
                    running = False
                    break
                self.state.handle_event(event)

            if not running:
                break

            self.state.update(dt)
            self.state.draw(self.render)
            self.blit_scaled()
            pygame.display.flip()

        pygame.quit()


SURVIVAL_GENDER_OPTIONS = ("男", "女")
SURVIVAL_HEIGHT_OPTIONS = ("矮", "标准", "高")
SURVIVAL_FACE_OPTIONS = ("圆", "方", "尖")
SURVIVAL_EYE_OPTIONS = ("正常", "大眼", "凶", "眯眼")
SURVIVAL_HAIR_OPTIONS = ("短发", "刘海", "长发", "帽子")
SURVIVAL_NOSE_OPTIONS = ("无", "点", "线")
SURVIVAL_OUTFIT_OPTIONS = ("夹克蓝", "工装绿", "战术灰", "囚服橙", "医护白")


@dataclass
class SurvivalAvatar:
    gender: int = 0
    height: int = 1
    face: int = 0
    eyes: int = 0
    hair: int = 0
    nose: int = 0
    outfit: int = 0

    def clamp_all(self) -> None:
        self.gender = int(self.gender) % len(SURVIVAL_GENDER_OPTIONS)
        self.height = int(self.height) % len(SURVIVAL_HEIGHT_OPTIONS)
        self.face = int(self.face) % len(SURVIVAL_FACE_OPTIONS)
        self.eyes = int(self.eyes) % len(SURVIVAL_EYE_OPTIONS)
        self.hair = int(self.hair) % len(SURVIVAL_HAIR_OPTIONS)
        self.nose = int(self.nose) % len(SURVIVAL_NOSE_OPTIONS)
        self.outfit = int(self.outfit) % len(SURVIVAL_OUTFIT_OPTIONS)


class StartScreenState(State):
    def on_enter(self) -> None:
        self.t = 0.0

    def _start(self) -> None:
        self.app.set_state(MainMenuState(self.app))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE,):
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            self._start()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            self._start()
            return

        if event.type == pygame.JOYBUTTONDOWN:
            self._start()
            return

    def update(self, dt: float) -> None:
        self.t += dt

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((12, 12, 16))

        draw_text(
            surface,
            self.app.font_l,
            "幸存者 DEMO",
            (INTERNAL_W // 2, 92),
            pygame.Color(240, 240, 240),
            anchor="center",
        )
        draw_text(
            surface,
            self.app.font_s,
            f"像素画布 {INTERNAL_W}x{INTERNAL_H}",
            (INTERNAL_W // 2, 122),
            pygame.Color(170, 170, 180),
            anchor="center",
        )

        blink = int(self.t * 2.0) % 2 == 0
        prompt_col = pygame.Color(230, 230, 240) if blink else pygame.Color(120, 120, 135)
        draw_text(
            surface,
            self.app.font_m,
            "按任意键开始",
            (INTERNAL_W // 2, 168),
            prompt_col,
            anchor="center",
        )
        draw_text(
            surface,
            self.app.font_s,
            "Esc 退出",
            (INTERNAL_W // 2, INTERNAL_H - 18),
            pygame.Color(120, 120, 135),
            anchor="center",
        )


class MainMenuState(State):
    def on_enter(self) -> None:
        self.items = ["硬核生存(原型)", "幸存者Demo", "设置", "退出"]
        self.index = 0
        self.item_rects: list[pygame.Rect] = []

    def _activate(self) -> None:
        if self.index == 0:
            self.app.set_state(SurvivalCreateState(self.app))
        elif self.index == 1:
            self.app.set_state(CharacterSelectState(self.app))
        elif self.index == 2:
            self.app.set_state(SettingsState(self.app))
        else:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.index = (self.index - 1) % len(self.items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.index = (self.index + 1) % len(self.items)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate()
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        for i, rect in enumerate(self.item_rects):
            if rect.collidepoint(internal):
                self.index = i
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    self._activate()
                break

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((18, 18, 22))
        draw_text(
            surface,
            self.app.font_l,
            "幸存者 DEMO",
            (INTERNAL_W // 2, 30),
            pygame.Color(240, 240, 240),
            anchor="center",
        )
        draw_text(
            surface,
            self.app.font_s,
            "方向键/鼠标选择 | Enter/左键确认",
            (INTERNAL_W // 2, 50),
            pygame.Color(180, 180, 190),
            anchor="center",
        )

        self.item_rects = []
        y0 = 88
        for i, label in enumerate(self.items):
            rect = pygame.Rect(0, 0, 180, 22)
            rect.center = (INTERNAL_W // 2, y0 + i * 28)
            self.item_rects.append(draw_button(surface, self.app.font_m, label, rect, selected=(i == self.index)))


class SurvivalCreateState(State):
    def on_enter(self) -> None:
        self.avatar = SurvivalAvatar()
        self.avatar.clamp_all()
        self.fields: list[tuple[str, str, tuple[str, ...]]] = [
            ("性别", "gender", SURVIVAL_GENDER_OPTIONS),
            ("身高", "height", SURVIVAL_HEIGHT_OPTIONS),
            ("脸型", "face", SURVIVAL_FACE_OPTIONS),
            ("眼睛", "eyes", SURVIVAL_EYE_OPTIONS),
            ("头发", "hair", SURVIVAL_HAIR_OPTIONS),
            ("鼻子", "nose", SURVIVAL_NOSE_OPTIONS),
            ("衣服", "outfit", SURVIVAL_OUTFIT_OPTIONS),
        ]
        self.index = 0
        self._dirty = True
        self._preview_frames: dict[str, list[pygame.Surface]] | None = None
        self.row_rects: list[pygame.Rect] = []
        self.left_rects: list[pygame.Rect] = []
        self.right_rects: list[pygame.Rect] = []

    def _rebuild_preview(self) -> None:
        self.avatar.clamp_all()
        self._preview_frames = HardcoreSurvivalState.build_avatar_player_frames(self.avatar, run=False)
        self._dirty = False

    def _change_value(self, delta: int) -> None:
        label, attr, options = self.fields[int(self.index)]
        cur = int(getattr(self.avatar, attr, 0))
        nxt = (cur + int(delta)) % max(1, len(options))
        setattr(self.avatar, attr, int(nxt))
        self._dirty = True

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            key = int(event.key)
            if key in (pygame.K_ESCAPE,):
                self.app.set_state(MainMenuState(self.app))
                return
            if key in (pygame.K_UP, pygame.K_w):
                self.index = (int(self.index) - 1) % len(self.fields)
                return
            if key in (pygame.K_DOWN, pygame.K_s):
                self.index = (int(self.index) + 1) % len(self.fields)
                return
            if key in (pygame.K_LEFT, pygame.K_a):
                self._change_value(-1)
                return
            if key in (pygame.K_RIGHT, pygame.K_d):
                self._change_value(1)
                return
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.avatar.clamp_all()
                self.app.set_state(HardcoreSurvivalState(self.app, avatar=self.avatar))
                return
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return
        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        # Arrow buttons first.
        for i, r in enumerate(self.left_rects):
            if r.collidepoint(internal):
                self.index = int(i)
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    self._change_value(-1)
                return
        for i, r in enumerate(self.right_rects):
            if r.collidepoint(internal):
                self.index = int(i)
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    self._change_value(1)
                return

        for i, r in enumerate(self.row_rects):
            if r.collidepoint(internal):
                self.index = int(i)
                return

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        if self._dirty or self._preview_frames is None:
            self._rebuild_preview()

        surface.fill((14, 14, 18))
        draw_text(surface, self.app.font_l, "捏脸 / 角色创建", (INTERNAL_W // 2, 26), pygame.Color(240, 240, 240), anchor="center")
        draw_text(surface, self.app.font_s, "↑↓选择  ←→调整  Enter开始  Esc返回", (INTERNAL_W // 2, 48), pygame.Color(170, 170, 180), anchor="center")

        # Options (left)
        x0 = 22
        y0 = 74
        row_w = 220
        row_h = 20
        gap = 4
        btn_w = 20
        btn_gap = 2

        self.row_rects = []
        self.left_rects = []
        self.right_rects = []
        for i, (label, attr, options) in enumerate(self.fields):
            v = int(getattr(self.avatar, attr, 0)) % max(1, len(options))
            selected = i == int(self.index)
            text = f"{label}: {options[v]}"
            r = pygame.Rect(x0, y0 + i * (row_h + gap), row_w, row_h)

            bg = (34, 34, 42) if selected else (24, 24, 30)
            border = (110, 110, 130) if selected else (70, 70, 86)
            text_col = pygame.Color(240, 240, 240) if selected else pygame.Color(200, 200, 210)
            pygame.draw.rect(surface, bg, r, border_radius=6)
            pygame.draw.rect(surface, border, r, 2, border_radius=6)

            left_btn = pygame.Rect(r.right - btn_w * 2 - btn_gap, r.y, btn_w, r.h)
            right_btn = pygame.Rect(r.right - btn_w, r.y, btn_w, r.h)
            draw_button(surface, self.app.font_m, "<", left_btn, selected=selected)
            draw_button(surface, self.app.font_m, ">", right_btn, selected=selected)
            draw_text(surface, self.app.font_m, text, (r.x + 8, r.y + 2), text_col, anchor="topleft")

            self.row_rects.append(r)
            self.left_rects.append(left_btn)
            self.right_rects.append(right_btn)

        # Preview (right)
        panel = pygame.Rect(250, 70, 208, 170)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (70, 70, 86), panel, 2, border_radius=12)
        frames = self._preview_frames or {}
        spr_down = frames.get("down", [None])[0]
        spr_right = frames.get("right", [None])[0]
        scale = 7
        if spr_down is not None:
            img = pygame.transform.scale(spr_down, (spr_down.get_width() * scale, spr_down.get_height() * scale))
            surface.blit(img, img.get_rect(center=(panel.centerx - 40, panel.centery + 10)))
        if spr_right is not None:
            img = pygame.transform.scale(spr_right, (spr_right.get_width() * scale, spr_right.get_height() * scale))
            surface.blit(img, img.get_rect(center=(panel.centerx + 48, panel.centery + 10)))
        draw_text(surface, self.app.font_s, "预览: 下 / 右", (panel.centerx, panel.top + 10), pygame.Color(200, 200, 210), anchor="center")


class HardcoreSurvivalState(State):
    TILE_SIZE = 10
    CHUNK_SIZE = 32
    ROAD_PERIOD = CHUNK_SIZE * 4
    ROAD_HALF = ROAD_PERIOD // 2
    ROAD_W = 2

    T_GRASS = 1
    T_FOREST = 2
    T_WATER = 3
    T_ROAD = 4
    T_FLOOR = 5
    T_WALL = 6
    T_DOOR = 7
    T_TABLE = 8
    T_SHELF = 9
    T_BED = 10
    T_PAVEMENT = 11
    T_PARKING = 12
    T_COURT = 13
    T_SIDEWALK = 14
    T_BRICK = 15
    T_CONCRETE = 16
    T_SAND = 17
    T_BOARDWALK = 18
    T_MARSH = 19
    T_HIGHWAY = 20

    DAY_LENGTH_S = 8 * 60.0
    SEASON_LENGTH_DAYS = 7
    SEASONS = ("春", "夏", "秋", "冬")
    START_DAY_FRACTION = 0.30  # ~07:12

    WEATHER_TRANSITION_S = 18.0
    WEATHER_KINDS = ("clear", "cloudy", "rain", "storm", "snow")
    WEATHER_NAMES: dict[str, str] = {
        "clear": "晴",
        "cloudy": "阴",
        "rain": "雨",
        "storm": "暴雨",
        "snow": "雪",
    }

    def __init__(self, app: "App", *, avatar: SurvivalAvatar | None = None) -> None:
        super().__init__(app)
        self.avatar = avatar if avatar is not None else SurvivalAvatar()
        self.avatar.clamp_all()

    @dataclass(frozen=True)
    class _TileDef:
        name: str
        color: tuple[int, int, int]
        solid: bool
        slow: float = 1.0

    _TILES: dict[int, _TileDef] = {
        T_GRASS: _TileDef("grass", (32, 120, 56), solid=False),
        T_FOREST: _TileDef("forest", (22, 86, 42), solid=False, slow=0.82),
        T_WATER: _TileDef("water", (22, 56, 98), solid=True),
        T_ROAD: _TileDef("road", (72, 72, 74), solid=False),
        T_HIGHWAY: _TileDef("highway", (58, 58, 62), solid=False),
        T_FLOOR: _TileDef("floor", (58, 56, 52), solid=False),
        T_WALL: _TileDef("wall", (28, 28, 30), solid=True),
        T_DOOR: _TileDef("door", (120, 96, 52), solid=False),
        T_TABLE: _TileDef("table", (78, 62, 44), solid=True),
        T_SHELF: _TileDef("shelf", (50, 50, 54), solid=True),
        T_BED: _TileDef("bed", (84, 84, 110), solid=True),
        T_PAVEMENT: _TileDef("pavement", (92, 92, 96), solid=False),
        T_PARKING: _TileDef("parking", (66, 66, 78), solid=False),
        T_COURT: _TileDef("court", (116, 86, 66), solid=False),
        T_SIDEWALK: _TileDef("sidewalk", (156, 132, 86), solid=False),
        T_BRICK: _TileDef("brick", (140, 94, 80), solid=False),
        T_CONCRETE: _TileDef("concrete", (112, 112, 118), solid=False),
        T_SAND: _TileDef("sand", (192, 176, 112), solid=False, slow=0.92),
        T_BOARDWALK: _TileDef("boardwalk", (156, 116, 68), solid=False),
        T_MARSH: _TileDef("marsh", (56, 92, 62), solid=False, slow=0.78),
    }

    _PLAYER_PAL = {
        "H": (26, 26, 30),  # hair
        "S": (220, 190, 160),  # skin
        "C": (72, 92, 160),  # coat
        "P": (52, 52, 62),  # pants
        "B": (22, 22, 26),  # boots
        "E": (6, 6, 8),  # eye
    }

    # Side-walk gait tables (6 steps). "Near" is the leg closer to the camera
    # when facing right; "far" is the back leg. Offsets are in pixels from the
    # hip X, lifts drive knee bend (and a small 1px foot raise).
    #
    # Default gait is WALK; sprint (Shift) switches to RUN.
    _SIDE_WALK_WALK_NEAR_OFFSETS = (1, 1, 0, -1, 0, 2)
    _SIDE_WALK_WALK_FAR_OFFSETS = (-1, 0, 2, 1, 1, 0)
    _SIDE_WALK_WALK_NEAR_LIFTS = (0, 0, 0, 0, 1, 2)
    _SIDE_WALK_WALK_FAR_LIFTS = (0, 1, 2, 0, 0, 0)

    _SIDE_WALK_RUN_NEAR_OFFSETS = (2, 1, -1, -1, 1, 3)
    _SIDE_WALK_RUN_FAR_OFFSETS = (-2, 1, 3, 2, 1, -1)
    _SIDE_WALK_RUN_NEAR_LIFTS = (0, 0, 0, 1, 3, 4)
    _SIDE_WALK_RUN_FAR_LIFTS = (1, 3, 4, 0, 0, 0)

    @classmethod
    def _side_walk_tables(
        cls,
        *,
        run: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if bool(run):
            return (
                cls._SIDE_WALK_RUN_NEAR_OFFSETS,
                cls._SIDE_WALK_RUN_FAR_OFFSETS,
                cls._SIDE_WALK_RUN_NEAR_LIFTS,
                cls._SIDE_WALK_RUN_FAR_LIFTS,
            )
        return (
            cls._SIDE_WALK_WALK_NEAR_OFFSETS,
            cls._SIDE_WALK_WALK_FAR_OFFSETS,
            cls._SIDE_WALK_WALK_NEAR_LIFTS,
            cls._SIDE_WALK_WALK_FAR_LIFTS,
        )

    _SURVIVOR_SKELETON_BONES: tuple[tuple[str, str], ...] = (
        ("head", "chest"),
        ("chest", "hip"),
        ("hip", "l_knee"),
        ("l_knee", "l_foot"),
        ("hip", "r_knee"),
        ("r_knee", "r_foot"),
        ("chest", "l_hand"),
        ("chest", "r_hand"),
    )

    @staticmethod
    def _solve_two_bone_knee(
        hip_x: float,
        hip_y: float,
        foot_x: float,
        foot_y: float,
        upper_len: float,
        lower_len: float,
        *,
        prefer_forward: bool = True,
    ) -> tuple[int, int]:
        dx = float(foot_x) - float(hip_x)
        dy = float(foot_y) - float(hip_y)
        d = float(math.hypot(dx, dy))
        if d <= 1e-4:
            return int(round(float(hip_x))), int(round(float(hip_y) + float(upper_len)))

        max_d = float(upper_len + lower_len) - 1e-3
        min_d = float(abs(float(upper_len) - float(lower_len))) + 1e-3
        d = float(clamp(float(d), float(min_d), float(max_d)))

        ux = dx / d
        uy = dy / d

        a = (float(upper_len) * float(upper_len) - float(lower_len) * float(lower_len) + d * d) / (2.0 * d)
        h2 = float(upper_len) * float(upper_len) - a * a
        h = math.sqrt(max(0.0, float(h2)))

        px = float(hip_x) + a * ux
        py = float(hip_y) + a * uy

        vx = -uy
        vy = ux
        k1x = px + h * vx
        k1y = py + h * vy
        k2x = px - h * vx
        k2y = py - h * vy

        if prefer_forward:
            kx, ky = (k1x, k1y) if k1x >= k2x else (k2x, k2y)
        else:
            kx, ky = (k1x, k1y) if k1x <= k2x else (k2x, k2y)
        return int(round(float(kx))), int(round(float(ky)))

    @classmethod
    def _survivor_skeleton_nodes(
        cls,
        direction: str,
        step: int,
        *,
        idle: bool,
        height_delta: int = 0,
        run: bool = False,
    ) -> dict[str, tuple[int, int]]:
        direction = str(direction)
        w, h = 12, 16

        step = int(step) % 6
        if idle:
            step = 2

        height_delta = int(height_delta)
        nodes: dict[str, tuple[int, int]] = {}

        if direction in ("right", "left"):
            leg_top = int(clamp(10 + height_delta, 8, 12))
            hip_y = int(leg_top)
            far_hip_x = 5
            near_hip_x = 6

            if idle:
                step_idx = 0
                near_off = 0
                far_off = 0
                near_lift = 0
                far_lift = 0
            else:
                step_idx = int(step) % 6
                near_offsets, far_offsets, near_lifts, far_lifts = cls._side_walk_tables(run=bool(run))
                near_off = int(near_offsets[step_idx])
                far_off = int(far_offsets[step_idx])
                near_lift = int(near_lifts[step_idx])
                far_lift = int(far_lifts[step_idx])

            def leg_joints(hip_x: int, foot_x: int, lift: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
                hip_x = int(clamp(int(hip_x), 0, w - 1))
                foot_x = int(clamp(int(foot_x), 0, w - 1))
                lift_px = int(clamp(int(min(int(lift), 4)), 0, 4))

                raise_px = 1 if int(lift_px) > 0 else 0
                foot_y = int(max(int(hip_y) + 1, 15 - int(raise_px)))
                foot_y = int(clamp(int(foot_y), 0, 15))

                mid_x = float(hip_x + foot_x) * 0.5
                mid_y = float(hip_y + foot_y) * 0.5

                if lift_px <= 0:
                    knee_x = int(round((float(hip_x) + float(foot_x)) * 0.5))
                    knee_y = int(round((float(hip_y) + float(foot_y)) * 0.5))
                else:
                    dist = float(15 - int(hip_y))
                    total = float(max(4.0, float(dist) + 1.0))
                    upper = float(total * 0.5)
                    lower = float(total * 0.5)
                    ik_x, ik_y = cls._solve_two_bone_knee(
                        float(hip_x),
                        float(hip_y),
                        float(foot_x),
                        float(foot_y),
                        float(upper),
                        float(lower),
                        prefer_forward=True,
                    )
                    # Blend toward IK as the lift increases to avoid a too-sharp knee on small lifts.
                    t = float(clamp((float(lift_px) / 4.0) ** 0.85, 0.0, 1.0))
                    knee_x = int(round(mid_x + (float(ik_x) - mid_x) * t))
                    knee_y = int(round(mid_y + (float(ik_y) - mid_y) * t))

                kmin = min(int(hip_x), int(foot_x)) - 1
                kmax = max(int(hip_x), int(foot_x)) + 2
                knee_x = int(clamp(int(knee_x), int(kmin), int(kmax)))
                knee_x = int(clamp(int(knee_x), 0, w - 1))
                knee_y = int(clamp(int(knee_y), int(hip_y), int(foot_y)))
                return (int(hip_x), int(hip_y)), (int(knee_x), int(knee_y)), (int(foot_x), int(foot_y))

            far_foot_x = int(far_hip_x + int(far_off))
            near_foot_x = int(near_hip_x + int(near_off))

            l_hip, l_knee, l_foot = leg_joints(int(far_hip_x), int(far_foot_x), int(far_lift))
            r_hip, r_knee, r_foot = leg_joints(int(near_hip_x), int(near_foot_x), int(near_lift))

            nodes.update(
                {
                    "head": (7, 3),
                    "chest": (6, 8),
                    "hip": (int(round((float(far_hip_x) + float(near_hip_x)) * 0.5)), int(hip_y)),
                    "l_hip": l_hip,
                    "l_knee": l_knee,
                    "l_foot": l_foot,
                    "r_hip": r_hip,
                    "r_knee": r_knee,
                    "r_foot": r_foot,
                    "l_hand": (4, 9),
                    "r_hand": (8, 9),
                }
            )

            if direction == "left":
                nodes = {name: (int(w - 1 - x), int(y)) for name, (x, y) in nodes.items()}
            return nodes

        if direction in ("down", "up"):
            leg_top = int(clamp((11 if direction == "up" else 10) + height_delta, 8, 12))
            if direction == "down":
                left_hip = 4
                right_hip = 7
                head = (6, 3)
                l_hand = (2, 9)
                r_hand = (9, 9)
            else:
                left_hip = 5
                right_hip = 6
                head = (6, 2)
                l_hand = (2, 8)
                r_hand = (9, 8)

            left_lift = 2 if step == 1 else 1 if step == 0 else 0
            right_lift = 2 if step == 4 else 1 if step == 3 else 0

            left_x = int(left_hip - (1 if left_lift else 0))
            right_x = int(right_hip + (1 if right_lift else 0))
            left_knee_y = int(13 - left_lift)
            right_knee_y = int(13 - right_lift)
            left_boot_y = int(15 - left_lift)
            right_boot_y = int(15 - right_lift)

            nodes.update(
                {
                    "head": (int(head[0]), int(head[1])),
                    "chest": (6, 8),
                    "hip": (6, int(leg_top)),
                    "l_hip": (int(left_hip), int(leg_top)),
                    "r_hip": (int(right_hip), int(leg_top)),
                    "l_knee": (int(left_x), int(left_knee_y)),
                    "r_knee": (int(right_x), int(right_knee_y)),
                    "l_foot": (int(left_x), int(left_boot_y)),
                    "r_foot": (int(right_x), int(right_boot_y)),
                    "l_hand": (int(l_hand[0]), int(l_hand[1])),
                    "r_hand": (int(r_hand[0]), int(r_hand[1])),
                }
            )
            return nodes

        return {}

    def _make_player_sprite(
        direction: str,
        step: int,
        *,
        idle: bool,
        run: bool = False,
        pal=_PLAYER_PAL,
    ) -> pygame.Surface:
        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        hair = pal["H"]
        skin = pal["S"]
        coat = pal["C"]
        pants = pal["P"]
        boots = pal["B"]
        eye = pal["E"]
        coat_dark = (56, 72, 140)
        pants_dark = (40, 40, 50)

        step = int(step) % 6
        if idle:
            step = 2

        def draw_line(
            col: tuple[int, int, int],
            x0: int,
            y0: int,
            x1: int,
            y1: int,
        ) -> None:
            x0 = int(clamp(int(x0), 0, w - 1))
            x1 = int(clamp(int(x1), 0, w - 1))
            y0 = int(clamp(int(y0), 0, h - 1))
            y1 = int(clamp(int(y1), 0, h - 1))

            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x, y = x0, y0
            while True:
                surf.fill(col, pygame.Rect(int(x), int(y), 1, 1))
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        def draw_leg(
            hip_x: int,
            knee_x: int,
            boot_x: int,
            boot_y: int,
            *,
            leg_top: int,
            knee_y: int,
            leg_col: tuple[int, int, int],
        ) -> None:
            hip_x = int(clamp(int(hip_x), 0, w - 1))
            knee_x = int(clamp(int(knee_x), 0, w - 1))
            boot_x = int(clamp(int(boot_x), 0, w - 1))
            leg_top = int(clamp(int(leg_top), 0, h - 1))
            knee_y = int(clamp(int(knee_y), 0, h - 1))
            boot_y = int(clamp(int(boot_y), 0, h - 1))

            if knee_y < leg_top:
                knee_y = leg_top
            if boot_y < knee_y:
                boot_y = knee_y

            draw_line(leg_col, int(hip_x), int(leg_top), int(knee_x), int(knee_y))
            draw_line(leg_col, int(knee_x), int(knee_y), int(boot_x), int(boot_y))
            surf.fill(boots, pygame.Rect(boot_x, boot_y, 1, 1))

        def draw_straight_leg(
            hip_x: int,
            foot_x: int,
            foot_y: int,
            *,
            leg_top: int,
            leg_col: tuple[int, int, int],
        ) -> None:
            hip_x = int(clamp(int(hip_x), 0, w - 1))
            foot_x = int(clamp(int(foot_x), 0, w - 1))
            leg_top = int(clamp(int(leg_top), 0, h - 1))
            foot_y = int(clamp(int(foot_y), 0, h - 1))

            draw_line(leg_col, int(hip_x), int(leg_top), int(foot_x), int(foot_y))
            surf.fill(boots, pygame.Rect(int(foot_x), int(foot_y), 1, 1))

        if direction in ("down", "up"):
            leg_top = 11 if direction == "up" else 10
            if direction == "down":
                left_hip = 4
                right_hip = 7
            else:
                left_hip = 5
                right_hip = 6

            # 6-step walk: left swings early, right swings late.
            left_lift = 2 if step == 1 else 1 if step == 0 else 0
            right_lift = 2 if step == 4 else 1 if step == 3 else 0

            left_x = left_hip - (1 if left_lift else 0)
            right_x = right_hip + (1 if right_lift else 0)
            left_knee_y = 13 - left_lift
            right_knee_y = 13 - right_lift
            left_boot_y = 15 - left_lift
            right_boot_y = 15 - right_lift

            # Legs first (behind torso) so the player doesn't look "spidery".
            draw_leg(
                int(right_hip),
                int(right_x),
                int(right_x),
                int(right_boot_y),
                leg_top=int(leg_top),
                knee_y=int(right_knee_y),
                leg_col=pants_dark,
            )
            draw_leg(
                int(left_hip),
                int(left_x),
                int(left_x),
                int(left_boot_y),
                leg_top=int(leg_top),
                knee_y=int(left_knee_y),
                leg_col=pants,
            )

            if direction == "down":
                # Torso + arms.
                surf.fill(coat, pygame.Rect(3, 5, 6, 6))
                surf.fill(coat_dark, pygame.Rect(3, 10, 6, 1))
                surf.fill(coat, pygame.Rect(2, 6, 1, 3))
                surf.fill(coat, pygame.Rect(9, 6, 1, 3))
                surf.fill(skin, pygame.Rect(2, 9, 1, 1))
                surf.fill(skin, pygame.Rect(9, 9, 1, 1))

                # Head.
                surf.fill(hair, pygame.Rect(4, 0, 4, 2))
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
                surf.set_at((5, 3), eye)
                surf.set_at((7, 3), eye)
            else:
                # Torso + arms.
                surf.fill(coat, pygame.Rect(3, 4, 6, 7))
                surf.fill(coat_dark, pygame.Rect(4, 6, 4, 4))
                surf.fill(coat, pygame.Rect(2, 5, 1, 4))
                surf.fill(coat, pygame.Rect(9, 5, 1, 4))

                # Back head.
                surf.fill(hair, pygame.Rect(4, 0, 4, 3))
                surf.fill(skin, pygame.Rect(5, 3, 2, 1))

            return surf

        if direction == "right":
            leg_top = 10
            far_hip = 5
            near_hip = 6

            # Side-walk: alternating stride (one leg swings while the other supports).
            # 4-frame side gait (迈腿→向前→交叉→换腿向前):
            # - Swing leg bends (knee) on the first frame of the step.
            # - Then it extends straight while still forward.
            # - Swap draw order when legs cross so it doesn't look like only one
            #   leg is moving.
            if idle:
                step = 0
                near_offsets = (0, 0, 0, 0, 0, 0)
                far_offsets = (0, 0, 0, 0, 0, 0)
                near_lifts = (0, 0, 0, 0, 0, 0)
                far_lifts = (0, 0, 0, 0, 0, 0)
            else:
                step = int(step) % 6
                if bool(run):
                    near_offsets = (2, 1, -1, -1, 1, 3)
                    far_offsets = (-2, 1, 3, 2, 1, -1)
                    near_lifts = (0, 0, 0, 1, 3, 4)
                    far_lifts = (1, 3, 4, 0, 0, 0)
                else:
                    near_offsets = (1, 1, 0, -1, 0, 2)
                    far_offsets = (-1, 0, 2, 1, 1, 0)
                    near_lifts = (0, 0, 0, 0, 1, 2)
                    far_lifts = (0, 1, 2, 0, 0, 0)

            near_foot_x = int(near_hip + int(near_offsets[int(step)]))
            far_foot_x = int(far_hip + int(far_offsets[int(step)]))
            near_lift = int(near_lifts[int(step)])
            far_lift = int(far_lifts[int(step)])

            def solve_knee_ik(
                hip_x: float,
                hip_y: float,
                foot_x: float,
                foot_y: float,
                upper_len: float,
                lower_len: float,
            ) -> tuple[int, int]:
                dx = float(foot_x) - float(hip_x)
                dy = float(foot_y) - float(hip_y)
                d = float(math.hypot(dx, dy))
                if d <= 1e-4:
                    return int(round(float(hip_x))), int(round(float(hip_y) + float(upper_len)))

                max_d = float(upper_len + lower_len) - 1e-3
                min_d = float(abs(float(upper_len) - float(lower_len))) + 1e-3
                d = float(clamp(float(d), float(min_d), float(max_d)))

                ux = dx / d
                uy = dy / d

                a = (float(upper_len) * float(upper_len) - float(lower_len) * float(lower_len) + d * d) / (2.0 * d)
                h2 = float(upper_len) * float(upper_len) - a * a
                h = math.sqrt(max(0.0, float(h2)))

                px = float(hip_x) + a * ux
                py = float(hip_y) + a * uy

                # Prefer the knee to face forward (right) for the unflipped sprite.
                vx = -uy
                vy = ux
                k1x = px + h * vx
                k1y = py + h * vy
                k2x = px - h * vx
                k2y = py - h * vy
                if k1x >= k2x:
                    kx, ky = k1x, k1y
                else:
                    kx, ky = k2x, k2y
                return int(round(float(kx))), int(round(float(ky)))

            def draw_pose(hip_x: int, foot_x: int, lift: int, *, col: tuple[int, int, int]) -> None:
                hip_x = int(hip_x)
                foot_x = int(clamp(int(foot_x), 0, w - 1))
                lift_px = int(clamp(int(min(int(lift), 4)), 0, 4))

                hip_y = int(leg_top)
                raise_px = 1 if int(lift_px) > 0 else 0
                foot_y = int(max(int(leg_top) + 1, 15 - int(raise_px)))
                foot_y = int(clamp(int(foot_y), 0, 15))

                if lift_px <= 0:
                    draw_straight_leg(int(hip_x), int(foot_x), int(foot_y), leg_top=int(hip_y), leg_col=col)
                    return

                else:
                    dist = float(15 - int(hip_y))
                    total = float(max(4.0, float(dist) + 1.0))
                    upper = float(total * 0.5)
                    lower = float(total * 0.5)
                    knee_x, knee_y = solve_knee_ik(float(hip_x), float(hip_y), float(foot_x), float(foot_y), float(upper), float(lower))

                    mid_x = float(hip_x + foot_x) * 0.5
                    mid_y = float(hip_y + foot_y) * 0.5
                    t = float(clamp((float(lift_px) / 4.0) ** 0.85, 0.0, 1.0))
                    knee_x = int(round(mid_x + (float(knee_x) - mid_x) * t))
                    knee_y = int(round(mid_y + (float(knee_y) - mid_y) * t))
                    kmin = min(int(hip_x), int(foot_x)) - 1
                    kmax = max(int(hip_x), int(foot_x)) + 2
                    knee_x = int(clamp(int(knee_x), int(kmin), int(kmax)))

                knee_x = int(clamp(int(knee_x), 0, w - 1))
                knee_y = int(clamp(int(knee_y), int(hip_y), int(foot_y)))

                draw_leg(
                    int(hip_x),
                    int(knee_x),
                    int(foot_x),
                    int(foot_y),
                    leg_top=int(hip_y),
                    knee_y=int(knee_y),
                    leg_col=col,
                )

            # Depth + shading: the front leg uses the lighter pants color so
            # the crossing reads clearly (otherwise it can look like only the
            # near leg is moving).
            if int(near_foot_x) >= int(far_foot_x):
                # Near leg is in front.
                draw_pose(int(far_hip), int(far_foot_x), int(far_lift), col=pants_dark)
                draw_pose(int(near_hip), int(near_foot_x), int(near_lift), col=pants)
            else:
                # Far leg crosses in front.
                draw_pose(int(near_hip), int(near_foot_x), int(near_lift), col=pants_dark)
                draw_pose(int(far_hip), int(far_foot_x), int(far_lift), col=pants)

            # Torso + near arm.
            surf.fill(coat, pygame.Rect(4, 5, 5, 6))
            surf.fill(coat_dark, pygame.Rect(4, 10, 5, 1))
            surf.fill(coat, pygame.Rect(8, 6, 1, 3))
            surf.fill(skin, pygame.Rect(8, 9, 1, 1))

            # Head.
            surf.fill(hair, pygame.Rect(4, 0, 4, 2))
            surf.fill(skin, pygame.Rect(6, 2, 3, 3))
            surf.set_at((8, 3), eye)

            return surf

        return surf

    _P_DOWN_FRAMES: list[pygame.Surface] = []
    _P_DOWN_FRAMES.append(_make_player_sprite("down", 0, idle=True))
    for _i in range(6):
        _P_DOWN_FRAMES.append(_make_player_sprite("down", _i, idle=False))

    _P_UP_FRAMES: list[pygame.Surface] = []
    _P_UP_FRAMES.append(_make_player_sprite("up", 0, idle=True))
    for _i in range(6):
        _P_UP_FRAMES.append(_make_player_sprite("up", _i, idle=False))

    _P_RIGHT_FRAMES: list[pygame.Surface] = []
    _P_RIGHT_FRAMES.append(_make_player_sprite("right", 0, idle=True))
    for _i in range(6):
        _P_RIGHT_FRAMES.append(_make_player_sprite("right", _i, idle=False))    

    _PLAYER_FRAMES: dict[str, list[pygame.Surface]] = {
        "down": _P_DOWN_FRAMES,
        "up": _P_UP_FRAMES,
        "right": _P_RIGHT_FRAMES,
        "left": [],
    }
    for _s in _P_RIGHT_FRAMES:
        _PLAYER_FRAMES["left"].append(pygame.transform.flip(_s, True, False))

    @classmethod
    def _avatar_colors(cls, avatar: SurvivalAvatar) -> dict[str, tuple[int, int, int]]:
        avatar.clamp_all()
        gender = int(avatar.gender)
        outfit = int(avatar.outfit)

        hair = (26, 26, 30) if gender == 0 else (34, 28, 30)
        skin = (220, 190, 160) if gender == 0 else (230, 198, 168)
        eye = (6, 6, 8)
        boots = (22, 22, 26)

        if outfit == 0:  # jacket blue
            coat = (72, 92, 160)
            coat_dark = (56, 72, 140)
            pants = (52, 52, 62)
            pants_dark = (40, 40, 50)
        elif outfit == 1:  # work green
            coat = (70, 120, 90)
            coat_dark = (48, 86, 64)
            pants = (66, 54, 40)
            pants_dark = (46, 38, 28)
        elif outfit == 2:  # tactical gray
            coat = (92, 92, 104)
            coat_dark = (70, 70, 82)
            pants = (44, 44, 56)
            pants_dark = (32, 32, 40)
        elif outfit == 3:  # prisoner orange
            coat = (210, 130, 60)
            coat_dark = (170, 90, 44)
            pants = (70, 46, 34)
            pants_dark = (50, 32, 24)
        else:  # medic white
            coat = (220, 220, 230)
            coat_dark = (180, 180, 190)
            pants = (92, 92, 104)
            pants_dark = (70, 70, 82)

        return {
            "hair": hair,
            "skin": skin,
            "eye": eye,
            "coat": coat,
            "coat_dark": coat_dark,
            "pants": pants,
            "pants_dark": pants_dark,
            "boots": boots,
        }

    @classmethod
    def _make_avatar_player_sprite(
        cls,
        direction: str,
        step: int,
        *,
        idle: bool,
        avatar: SurvivalAvatar,
        run: bool = False,
    ) -> pygame.Surface:
        avatar.clamp_all()
        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        cols = cls._avatar_colors(avatar)
        hair = cols["hair"]
        skin = cols["skin"]
        coat = cols["coat"]
        coat_dark = cols["coat_dark"]
        pants = cols["pants"]
        pants_dark = cols["pants_dark"]
        boots = cols["boots"]
        eye = cols["eye"]
        outline = (10, 10, 12)
        skin_shade = (max(0, skin[0] - 45), max(0, skin[1] - 45), max(0, skin[2] - 45))

        step = int(step) % 6
        if idle:
            step = 2

        height_delta = 1 if int(avatar.height) == 0 else -1 if int(avatar.height) == 2 else 0
        face_kind = int(avatar.face)
        eye_kind = int(avatar.eyes)
        hair_kind = int(avatar.hair)
        nose_kind = int(avatar.nose)
        outfit_kind = int(avatar.outfit)

        def draw_hair_down() -> None:
            if hair_kind == 3:  # hat
                hat = coat_dark
                pygame.draw.rect(surf, hat, pygame.Rect(3, 0, 6, 2), border_radius=1)
                pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 6, 2), 1, border_radius=1)
                pygame.draw.rect(surf, hat, pygame.Rect(3, 2, 5, 1), border_radius=1)
                return

            surf.fill(hair, pygame.Rect(4, 0, 4, 2))
            if hair_kind == 1:  # bangs
                surf.set_at((5, 2), hair)
                surf.set_at((6, 2), hair)
            elif hair_kind == 2:  # long
                for (x, y) in ((3, 2), (8, 2), (3, 3), (8, 3), (3, 4), (8, 4)):
                    surf.set_at((x, y), hair)

        def draw_face_down() -> None:
            if face_kind == 0:  # round
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
            elif face_kind == 1:  # square
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
                surf.fill(skin, pygame.Rect(3, 3, 1, 2))
                surf.fill(skin, pygame.Rect(8, 3, 1, 2))
            else:  # sharp
                surf.fill(skin, pygame.Rect(4, 2, 4, 2))
                surf.fill(skin, pygame.Rect(5, 4, 2, 1))

            if eye_kind == 0:  # normal
                surf.set_at((5, 3), eye)
                surf.set_at((7, 3), eye)
            elif eye_kind == 1:  # big
                for px in (5, 7):
                    surf.set_at((px, 3), eye)
                    surf.set_at((px, 4), eye)
            elif eye_kind == 2:  # angry
                surf.set_at((5, 3), eye)
                surf.set_at((7, 3), eye)
                surf.set_at((5, 2), outline)
                surf.set_at((7, 2), outline)
            else:  # sleepy
                surf.set_at((5, 3), outline)
                surf.set_at((7, 3), outline)

            if nose_kind == 1:  # dot
                surf.set_at((5, 4), skin_shade)
            elif nose_kind == 2:  # line
                surf.set_at((5, 4), skin_shade)
                surf.set_at((6, 4), skin_shade)

        def draw_hair_right() -> None:
            if hair_kind == 3:  # hat
                hat = coat_dark
                pygame.draw.rect(surf, hat, pygame.Rect(3, 0, 7, 2), border_radius=1)
                pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 7, 2), 1, border_radius=1)
                pygame.draw.rect(surf, hat, pygame.Rect(4, 2, 5, 1), border_radius=1)
                return

            surf.fill(hair, pygame.Rect(4, 0, 4, 2))
            if hair_kind == 1:  # bangs
                surf.set_at((7, 2), hair)
            elif hair_kind == 2:  # long
                for (x, y) in ((4, 2), (4, 3), (4, 4)):
                    surf.set_at((x, y), hair)

        def draw_face_right() -> None:
            if face_kind == 0:
                surf.fill(skin, pygame.Rect(6, 2, 3, 3))
            elif face_kind == 1:
                surf.fill(skin, pygame.Rect(6, 2, 3, 3))
                surf.fill(skin, pygame.Rect(5, 3, 1, 2))
            else:
                surf.fill(skin, pygame.Rect(6, 2, 3, 2))
                surf.fill(skin, pygame.Rect(7, 4, 1, 1))

            if eye_kind == 1:  # big
                surf.set_at((8, 3), eye)
                surf.set_at((8, 4), eye)
            elif eye_kind == 2:  # angry
                surf.set_at((8, 3), eye)
                surf.set_at((8, 2), outline)
            elif eye_kind == 3:  # sleepy
                surf.set_at((8, 3), outline)
            else:
                surf.set_at((8, 3), eye)

            if nose_kind == 1:
                surf.set_at((7, 4), skin_shade)
            elif nose_kind == 2:
                surf.set_at((7, 4), skin_shade)
                surf.set_at((8, 4), skin_shade)

        def draw_line(
            col: tuple[int, int, int],
            x0: int,
            y0: int,
            x1: int,
            y1: int,
        ) -> None:
            x0 = int(clamp(int(x0), 0, w - 1))
            x1 = int(clamp(int(x1), 0, w - 1))
            y0 = int(clamp(int(y0), 0, h - 1))
            y1 = int(clamp(int(y1), 0, h - 1))

            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x, y = x0, y0
            while True:
                surf.fill(col, pygame.Rect(int(x), int(y), 1, 1))
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        def draw_leg(
            hip_x: int,
            knee_x: int,
            boot_x: int,
            boot_y: int,
            *,
            leg_top: int,
            knee_y: int,
            leg_col: tuple[int, int, int],
        ) -> None:
            hip_x = int(clamp(int(hip_x), 0, w - 1))
            knee_x = int(clamp(int(knee_x), 0, w - 1))
            boot_x = int(clamp(int(boot_x), 0, w - 1))
            leg_top = int(clamp(int(leg_top), 0, h - 1))
            knee_y = int(clamp(int(knee_y), 0, h - 1))
            boot_y = int(clamp(int(boot_y), 0, h - 1))

            if knee_y < leg_top:
                knee_y = leg_top
            if boot_y < knee_y:
                boot_y = knee_y

            draw_line(leg_col, int(hip_x), int(leg_top), int(knee_x), int(knee_y))
            draw_line(leg_col, int(knee_x), int(knee_y), int(boot_x), int(boot_y))
            surf.fill(boots, pygame.Rect(boot_x, boot_y, 1, 1))

        def draw_straight_leg(
            hip_x: int,
            foot_x: int,
            foot_y: int,
            *,
            leg_top: int,
            leg_col: tuple[int, int, int],
        ) -> None:
            hip_x = int(clamp(int(hip_x), 0, w - 1))
            foot_x = int(clamp(int(foot_x), 0, w - 1))
            leg_top = int(clamp(int(leg_top), 0, h - 1))
            foot_y = int(clamp(int(foot_y), 0, h - 1))

            draw_line(leg_col, int(hip_x), int(leg_top), int(foot_x), int(foot_y))
            surf.fill(boots, pygame.Rect(int(foot_x), int(foot_y), 1, 1))

        if direction in ("down", "up"):
            leg_top = (11 if direction == "up" else 10) + height_delta
            leg_top = int(clamp(int(leg_top), 8, 12))

            if direction == "down":
                left_hip = 4
                right_hip = 7
            else:
                left_hip = 5
                right_hip = 6

            # 6-step walk: left swings early, right swings late.
            left_lift = 2 if step == 1 else 1 if step == 0 else 0
            right_lift = 2 if step == 4 else 1 if step == 3 else 0

            left_x = left_hip - (1 if left_lift else 0)
            right_x = right_hip + (1 if right_lift else 0)
            left_knee_y = 13 - left_lift
            right_knee_y = 13 - right_lift
            left_boot_y = 15 - left_lift
            right_boot_y = 15 - right_lift

            draw_leg(
                int(right_hip),
                int(right_x),
                int(right_x),
                int(right_boot_y),
                leg_top=int(leg_top),
                knee_y=int(right_knee_y),
                leg_col=pants_dark,
            )
            draw_leg(
                int(left_hip),
                int(left_x),
                int(left_x),
                int(left_boot_y),
                leg_top=int(leg_top),
                knee_y=int(left_knee_y),
                leg_col=pants,
            )

            if direction == "down":
                surf.fill(coat, pygame.Rect(3, 5, 6, 6))
                surf.fill(coat_dark, pygame.Rect(3, 10, 6, 1))
                surf.fill(coat, pygame.Rect(2, 6, 1, 3))
                surf.fill(coat, pygame.Rect(9, 6, 1, 3))
                surf.fill(skin, pygame.Rect(2, 9, 1, 1))
                surf.fill(skin, pygame.Rect(9, 9, 1, 1))

                if outfit_kind == 2:  # tactical pouches
                    surf.fill(coat_dark, pygame.Rect(4, 7, 2, 2))
                    surf.fill(coat_dark, pygame.Rect(6, 7, 2, 2))
                elif outfit_kind == 3:  # stripes
                    for y in (6, 8, 10):
                        surf.fill(pants_dark, pygame.Rect(3, y, 6, 1))
                elif outfit_kind == 4:  # medic cross
                    surf.fill((210, 70, 80), pygame.Rect(5, 7, 1, 3))
                    surf.fill((210, 70, 80), pygame.Rect(4, 8, 3, 1))
                elif outfit_kind == 1:  # belt
                    surf.fill(pants_dark, pygame.Rect(3, 9, 6, 1))

                draw_hair_down()
                draw_face_down()
            else:
                surf.fill(coat, pygame.Rect(3, 4, 6, 7))
                surf.fill(coat_dark, pygame.Rect(4, 6, 4, 4))
                surf.fill(coat, pygame.Rect(2, 5, 1, 4))
                surf.fill(coat, pygame.Rect(9, 5, 1, 4))
                surf.fill(hair, pygame.Rect(4, 0, 4, 3))
                surf.fill(skin, pygame.Rect(5, 3, 2, 1))
                if hair_kind == 2:  # long hair back
                    surf.set_at((3, 3), hair)
                    surf.set_at((8, 3), hair)

            return surf

        # right (left built by flipping)
        leg_top = 10 + height_delta
        leg_top = int(clamp(int(leg_top), 8, 12))
        sk = cls._survivor_skeleton_nodes(
            "right",
            int(step),
            idle=bool(idle),
            height_delta=int(height_delta),
            run=bool(run),
        )
        l_hip = sk.get("l_hip", (5, int(leg_top)))
        l_knee = sk.get("l_knee", (5, 13))
        l_foot = sk.get("l_foot", (5, 15))
        r_hip = sk.get("r_hip", (6, int(leg_top)))
        r_knee = sk.get("r_knee", (6, 13))
        r_foot = sk.get("r_foot", (6, 15))

        if idle:
            step_idx = 0
            near_lift = 0
            far_lift = 0
        else:
            step_idx = int(step) % 6
            _, _, near_lifts, far_lifts = cls._side_walk_tables(run=bool(run))
            near_lift = int(near_lifts[step_idx])
            far_lift = int(far_lifts[step_idx])

        def draw_pose(
            hip: tuple[int, int],
            knee: tuple[int, int],
            foot: tuple[int, int],
            lift: int,
            *,
            col: tuple[int, int, int],
        ) -> None:
            if int(lift) <= 0:
                draw_straight_leg(int(hip[0]), int(foot[0]), int(foot[1]), leg_top=int(hip[1]), leg_col=col)
                return
            draw_leg(
                int(hip[0]),
                int(knee[0]),
                int(foot[0]),
                int(foot[1]),
                leg_top=int(hip[1]),
                knee_y=int(knee[1]),
                leg_col=col,
            )

        # Depth + shading: whichever leg is in front gets the lighter pants
        # color so the "cross" reads clearly.
        if int(r_foot[0]) >= int(l_foot[0]):
            # Near (right) leg in front.
            draw_pose(l_hip, l_knee, l_foot, int(far_lift), col=pants_dark)
            draw_pose(r_hip, r_knee, r_foot, int(near_lift), col=pants)
        else:
            # Far (left) leg crosses in front.
            draw_pose(r_hip, r_knee, r_foot, int(near_lift), col=pants_dark)
            draw_pose(l_hip, l_knee, l_foot, int(far_lift), col=pants)

        surf.fill(coat, pygame.Rect(4, 5, 5, 6))
        surf.fill(coat_dark, pygame.Rect(4, 10, 5, 1))
        surf.fill(coat, pygame.Rect(8, 6, 1, 3))
        surf.fill(skin, pygame.Rect(8, 9, 1, 1))

        if outfit_kind == 2:
            surf.fill(coat_dark, pygame.Rect(5, 7, 2, 2))
        elif outfit_kind == 3:
            for y in (6, 8, 10):
                surf.fill(pants_dark, pygame.Rect(4, y, 5, 1))
        elif outfit_kind == 4:
            surf.fill((210, 70, 80), pygame.Rect(6, 7, 1, 3))
            surf.fill((210, 70, 80), pygame.Rect(5, 8, 3, 1))

        draw_hair_right()
        draw_face_right()
        return surf

    @classmethod
    def build_avatar_player_frames(cls, avatar: SurvivalAvatar, *, run: bool = False) -> dict[str, list[pygame.Surface]]:
        avatar = avatar or SurvivalAvatar()
        avatar.clamp_all()

        down = [cls._make_avatar_player_sprite("down", 0, idle=True, avatar=avatar, run=bool(run))]
        for i in range(6):
            down.append(cls._make_avatar_player_sprite("down", i, idle=False, avatar=avatar, run=bool(run)))

        up = [cls._make_avatar_player_sprite("up", 0, idle=True, avatar=avatar, run=bool(run))]
        for i in range(6):
            up.append(cls._make_avatar_player_sprite("up", i, idle=False, avatar=avatar, run=bool(run)))

        right = [cls._make_avatar_player_sprite("right", 0, idle=True, avatar=avatar, run=bool(run))]
        for i in range(6):
            right.append(cls._make_avatar_player_sprite("right", i, idle=False, avatar=avatar, run=bool(run)))

        left: list[pygame.Surface] = []
        for s in right:
            left.append(pygame.transform.flip(s, True, False))

        return {"down": down, "up": up, "right": right, "left": left}

    @classmethod
    def _make_avatar_pose_sit_sprite(cls, direction: str, *, avatar: SurvivalAvatar) -> pygame.Surface:
        avatar = avatar or SurvivalAvatar()
        avatar.clamp_all()
        direction = str(direction)
        if direction == "left":
            return pygame.transform.flip(cls._make_avatar_pose_sit_sprite("right", avatar=avatar), True, False)

        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cols = cls._avatar_colors(avatar)
        hair = cols["hair"]
        skin = cols["skin"]
        coat = cols["coat"]
        coat_dark = cols["coat_dark"]
        pants = cols["pants"]
        pants_dark = cols["pants_dark"]
        boots = cols["boots"]
        eye = cols["eye"]
        outline = (10, 10, 12)

        def px(x: int, y: int, c: tuple[int, int, int]) -> None:
            if 0 <= x < w and 0 <= y < h:
                surf.set_at((int(x), int(y)), c)

        def rect(x: int, y: int, rw: int, rh: int, c: tuple[int, int, int]) -> None:
            if rw <= 0 or rh <= 0:
                return
            surf.fill(c, pygame.Rect(int(x), int(y), int(rw), int(rh)))

        if direction in ("down", "up"):
            # Head + hair.
            rect(4, 0, 4, 2, hair)
            rect(4, 2, 4, 3, skin)
            if direction == "down":
                px(5, 3, eye)
                px(6, 3, eye)
            else:
                px(5, 2, hair)
                px(6, 2, hair)

            # Torso.
            rect(4, 5, 4, 4, coat)
            rect(4, 9, 4, 1, coat_dark)
            rect(3, 6, 1, 3, coat_dark)
            rect(8, 6, 1, 3, coat_dark)

            # Sitting legs (bent).
            rect(4, 10, 4, 1, pants)
            rect(4, 11, 4, 1, pants_dark)
            rect(5, 12, 2, 1, pants)
            rect(4, 13, 2, 1, boots)
            rect(6, 13, 2, 1, boots)

            pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 6, 14), 1)
        else:
            # Right-facing sit pose.
            rect(4, 0, 4, 2, hair)
            rect(6, 2, 3, 3, skin)
            px(8, 3, eye)
            rect(4, 5, 5, 4, coat)
            rect(4, 9, 5, 1, coat_dark)
            rect(4, 6, 1, 3, coat_dark)
            rect(8, 6, 1, 3, coat_dark)

            rect(5, 10, 4, 1, pants)
            rect(6, 11, 3, 1, pants_dark)
            rect(7, 12, 2, 1, pants)
            rect(8, 13, 2, 1, boots)
            pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 7, 14), 1)

        return surf

    @classmethod
    def _make_avatar_pose_sleep_sprite(cls, frame: int, *, avatar: SurvivalAvatar) -> pygame.Surface:
        avatar = avatar or SurvivalAvatar()
        avatar.clamp_all()
        w, h = 16, 12
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cols = cls._avatar_colors(avatar)
        hair = cols["hair"]
        skin = cols["skin"]
        coat = cols["coat"]
        coat_dark = cols["coat_dark"]
        pants = cols["pants"]
        boots = cols["boots"]
        eye = cols["eye"]
        outline = (10, 10, 12)

        def rect(x: int, y: int, rw: int, rh: int, c: tuple[int, int, int]) -> None:
            if rw <= 0 or rh <= 0:
                return
            surf.fill(c, pygame.Rect(int(x), int(y), int(rw), int(rh)))

        # Head on the left.
        rect(1, 1, 4, 2, hair)
        rect(1, 3, 4, 2, skin)
        surf.set_at((3, 4), eye)

        # Body.
        rect(5, 2, 6, 4, coat)
        rect(5, 6, 6, 1, coat_dark)
        rect(11, 3, 3, 3, pants)
        rect(14, 4, 1, 2, boots)

        # Tiny "breath" highlight.
        if int(frame) % 2 == 0:
            surf.set_at((7, 3), (min(255, coat[0] + 20), min(255, coat[1] + 20), min(255, coat[2] + 20)))
        else:
            surf.set_at((7, 4), (min(255, coat[0] + 20), min(255, coat[1] + 20), min(255, coat[2] + 20)))

        pygame.draw.rect(surf, outline, pygame.Rect(0, 1, 15, 7), 1)
        return surf

    @classmethod
    def _make_avatar_cyclist_sprite(cls, direction: str, frame: int, *, avatar: SurvivalAvatar) -> pygame.Surface:
        avatar.clamp_all()
        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        cols = cls._avatar_colors(avatar)
        hair = cols["hair"]
        skin = cols["skin"]
        coat = cols["coat"]
        coat_dark = cols["coat_dark"]
        pants = cols["pants"]
        pants_dark = cols["pants_dark"]
        boots = cols["boots"]
        eye = cols["eye"]
        outline = (10, 10, 12)
        skin_shade = (max(0, skin[0] - 45), max(0, skin[1] - 45), max(0, skin[2] - 45))

        phase = int(frame) % 2
        face_kind = int(avatar.face)
        eye_kind = int(avatar.eyes)
        hair_kind = int(avatar.hair)
        nose_kind = int(avatar.nose)

        def draw_hair_down() -> None:
            if hair_kind == 3:
                hat = coat_dark
                pygame.draw.rect(surf, hat, pygame.Rect(3, 0, 6, 2), border_radius=1)
                pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 6, 2), 1, border_radius=1)
                pygame.draw.rect(surf, hat, pygame.Rect(3, 2, 5, 1), border_radius=1)
                return
            surf.fill(hair, pygame.Rect(4, 0, 4, 2))
            if hair_kind == 1:
                surf.set_at((5, 2), hair)
                surf.set_at((6, 2), hair)
            elif hair_kind == 2:
                for (x, y) in ((3, 2), (8, 2), (3, 3), (8, 3), (3, 4), (8, 4)):
                    surf.set_at((x, y), hair)

        def draw_face_down() -> None:
            if face_kind == 0:
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
            elif face_kind == 1:
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
                surf.fill(skin, pygame.Rect(3, 3, 1, 2))
                surf.fill(skin, pygame.Rect(8, 3, 1, 2))
            else:
                surf.fill(skin, pygame.Rect(4, 2, 4, 2))
                surf.fill(skin, pygame.Rect(5, 4, 2, 1))

            if eye_kind == 1:
                surf.set_at((5, 3), eye)
                surf.set_at((5, 4), eye)
                surf.set_at((6, 3), eye)
                surf.set_at((6, 4), eye)
            elif eye_kind == 2:
                surf.set_at((5, 3), eye)
                surf.set_at((6, 3), eye)
                surf.set_at((5, 2), outline)
                surf.set_at((6, 2), outline)
            elif eye_kind == 3:
                surf.set_at((5, 3), outline)
                surf.set_at((6, 3), outline)
            else:
                surf.set_at((5, 3), eye)
                surf.set_at((6, 3), eye)

            if nose_kind == 1:
                surf.set_at((5, 4), skin_shade)
            elif nose_kind == 2:
                surf.set_at((5, 4), skin_shade)
                surf.set_at((6, 4), skin_shade)

        def draw_hair_right() -> None:
            if hair_kind == 3:
                hat = coat_dark
                pygame.draw.rect(surf, hat, pygame.Rect(3, 0, 7, 2), border_radius=1)
                pygame.draw.rect(surf, outline, pygame.Rect(3, 0, 7, 2), 1, border_radius=1)
                pygame.draw.rect(surf, hat, pygame.Rect(4, 2, 5, 1), border_radius=1)
                return
            surf.fill(hair, pygame.Rect(4, 0, 4, 2))
            if hair_kind == 1:
                surf.set_at((7, 2), hair)
            elif hair_kind == 2:
                for (x, y) in ((4, 2), (4, 3), (4, 4)):
                    surf.set_at((x, y), hair)

        def draw_face_right() -> None:
            if face_kind == 0:
                surf.fill(skin, pygame.Rect(6, 2, 3, 3))
            elif face_kind == 1:
                surf.fill(skin, pygame.Rect(6, 2, 3, 3))
                surf.fill(skin, pygame.Rect(5, 3, 1, 2))
            else:
                surf.fill(skin, pygame.Rect(6, 2, 3, 2))
                surf.fill(skin, pygame.Rect(7, 4, 1, 1))

            if eye_kind == 1:
                surf.set_at((8, 3), eye)
                surf.set_at((8, 4), eye)
            elif eye_kind == 2:
                surf.set_at((8, 3), eye)
                surf.set_at((8, 2), outline)
            elif eye_kind == 3:
                surf.set_at((8, 3), outline)
            else:
                surf.set_at((8, 3), eye)

            if nose_kind == 1:
                surf.set_at((7, 4), skin_shade)
            elif nose_kind == 2:
                surf.set_at((7, 4), skin_shade)
                surf.set_at((8, 4), skin_shade)

        def draw_leg(x: int, y0: int, y1: int, *, col: tuple[int, int, int]) -> None:
            x = int(clamp(int(x), 0, w - 1))
            y0 = int(clamp(int(y0), 0, h - 1))
            y1 = int(clamp(int(y1), 0, h - 1))
            if y1 < y0:
                y0, y1 = y1, y0
            for y in range(y0, y1):
                surf.fill(col, pygame.Rect(x, y, 1, 1))
            surf.fill(boots, pygame.Rect(x, y1, 1, 1))

        if direction in ("down", "up"):
            if direction == "down":
                draw_hair_down()
                draw_face_down()
            else:
                surf.fill(hair, pygame.Rect(4, 0, 4, 3))
                surf.fill(skin, pygame.Rect(5, 3, 2, 1))
                if hair_kind == 2:
                    surf.set_at((3, 3), hair)
                    surf.set_at((8, 3), hair)

            surf.fill(coat, pygame.Rect(3, 5, 6, 5))
            surf.fill(coat_dark, pygame.Rect(4, 7, 4, 3))
            surf.fill(coat, pygame.Rect(2, 6, 1, 2))
            surf.fill(coat, pygame.Rect(9, 6, 1, 2))
            surf.fill(skin, pygame.Rect(2, 8, 1, 1))
            surf.fill(skin, pygame.Rect(9, 8, 1, 1))

            lx = 5
            rx = 6
            if direction == "down":
                l_y = 13 + (0 if phase == 0 else 1)
                r_y = 13 + (1 if phase == 0 else 0)
            else:
                l_y = 14 - (0 if phase == 0 else 1)
                r_y = 14 - (1 if phase == 0 else 0)
            draw_leg(lx, 11, l_y, col=pants)
            draw_leg(rx, 11, r_y, col=pants_dark)
            surf.fill((30, 30, 34), pygame.Rect(4, 10, 4, 1))  # seat hint
            return surf

        # right
        draw_hair_right()
        draw_face_right()
        surf.fill(coat, pygame.Rect(4, 5, 5, 5))
        surf.fill(coat_dark, pygame.Rect(4, 9, 5, 1))
        surf.fill(coat, pygame.Rect(8, 6, 2, 1))
        surf.fill(skin, pygame.Rect(9, 7, 1, 1))
        near_shift = 1 if phase == 0 else 0
        far_shift = 0 if phase == 0 else 1
        draw_leg(5 + far_shift, 10, 14, col=pants_dark)
        draw_leg(6 + near_shift, 10, 13, col=pants)
        return surf

    @classmethod
    def build_avatar_cyclist_frames(cls, avatar: SurvivalAvatar) -> dict[str, list[pygame.Surface]]:
        avatar = avatar or SurvivalAvatar()
        avatar.clamp_all()
        down = [cls._make_avatar_cyclist_sprite("down", 0, avatar=avatar), cls._make_avatar_cyclist_sprite("down", 1, avatar=avatar)]
        up = [cls._make_avatar_cyclist_sprite("up", 0, avatar=avatar), cls._make_avatar_cyclist_sprite("up", 1, avatar=avatar)]
        right = [cls._make_avatar_cyclist_sprite("right", 0, avatar=avatar), cls._make_avatar_cyclist_sprite("right", 1, avatar=avatar)]
        left: list[pygame.Surface] = []
        for s in right:
            left.append(pygame.transform.flip(s, True, False))
        return {"down": down, "up": up, "right": right, "left": left}

    _MONSTER_PALS: dict[str, dict[str, tuple[int, int, int]]] = {
        "walker": {
            "skin": (92, 140, 96),
            "cloth": (76, 82, 74),
            "cloth2": (56, 60, 54),
            "pants": (44, 44, 56),
            "boots": (18, 18, 22),
            "eye": (18, 18, 22),
            "blood": (150, 70, 70),
        },
        "runner": {
            "skin": (100, 150, 104),
            "cloth": (140, 70, 70),
            "cloth2": (108, 52, 54),
            "pants": (40, 40, 50),
            "boots": (18, 18, 22),
            "eye": (18, 18, 22),
            "blood": (160, 76, 76),
        },
        "screamer": {
            "skin": (170, 170, 182),
            "cloth": (110, 80, 140),
            "cloth2": (78, 56, 104),
            "pants": (40, 40, 50),
            "boots": (18, 18, 22),
            "eye": (18, 18, 22),
            "blood": (170, 80, 90),
        },
    }

    def _make_monster_sprite(
        kind: str,
        direction: str,
        step: int,
        *,
        idle: bool,
        pals: dict[str, dict[str, tuple[int, int, int]]] = _MONSTER_PALS,
    ) -> pygame.Surface:
        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        pal = pals.get(str(kind), pals["walker"])
        skin = pal["skin"]
        cloth = pal["cloth"]
        cloth2 = pal["cloth2"]
        pants = pal["pants"]
        boots = pal["boots"]
        eye = pal["eye"]
        blood = pal["blood"]
        outline = (10, 10, 12)

        def shade(col: tuple[int, int, int], k: float) -> tuple[int, int, int]:
            return (
                int(clamp(float(col[0]) * float(k), 0.0, 255.0)),
                int(clamp(float(col[1]) * float(k), 0.0, 255.0)),
                int(clamp(float(col[2]) * float(k), 0.0, 255.0)),
            )

        pants_far = shade(pants, 0.72)
        boots_far = shade(boots, 0.72)

        step = int(step) % 2
        if idle:
            step = 0

        stride = 1 if step == 0 else -1

        def draw_leg(
            hip_x: int,
            knee_x: int,
            boot_x: int,
            boot_y: int,
            *,
            col: tuple[int, int, int],
            boot_col: tuple[int, int, int] | None = None,
        ) -> None:
            hip_x = int(clamp(int(hip_x), 0, w - 1))
            knee_x = int(clamp(int(knee_x), 0, w - 1))
            boot_x = int(clamp(int(boot_x), 0, w - 1))
            boot_y = int(clamp(int(boot_y), 0, h - 1))
            leg_top = 10
            knee_y = 13
            for y in range(leg_top, knee_y):
                surf.fill(col, pygame.Rect(hip_x, y, 1, 1))
            x0 = min(hip_x, knee_x)
            x1 = max(hip_x, knee_x)
            for x in range(x0, x1 + 1):
                surf.fill(col, pygame.Rect(x, knee_y, 1, 1))
            for y in range(knee_y, boot_y):
                surf.fill(col, pygame.Rect(knee_x, y, 1, 1))
            surf.fill((boots if boot_col is None else boot_col), pygame.Rect(boot_x, boot_y, 1, 1))

        if direction in ("down", "up"):
            left_hip = 4
            right_hip = 7
            left_boot_x = left_hip
            right_boot_x = right_hip
            if kind in ("walker", "screamer"):
                # Zombie shuffle: one leg "scuffs" forward with a bent knee, the other drags.
                lift = (1 if kind == "walker" else 2) if not idle else 0
                bend = (1 if kind == "walker" else 2) if not idle else 0
                if step == 0:
                    left_boot_y = 15 - lift
                    right_boot_y = 15
                    left_knee_x = left_hip + bend
                    right_knee_x = right_hip
                else:
                    left_boot_y = 15
                    right_boot_y = 15 - lift
                    left_knee_x = left_hip
                    right_knee_x = right_hip - bend
                left_boot_x = left_knee_x
                right_boot_x = right_knee_x
            else:
                left_boot_y = 15 - (1 if stride > 0 else 0)
                right_boot_y = 15 - (1 if stride < 0 else 0)
                left_knee_x = left_hip + stride
                right_knee_x = right_hip - stride
                left_boot_x = left_knee_x
                right_boot_x = right_knee_x

            # Legs behind torso.
            draw_leg(
                right_hip,
                right_knee_x,
                right_boot_x,
                right_boot_y,
                col=pants_far,
                boot_col=boots_far,
            )
            draw_leg(left_hip, left_knee_x, left_boot_x, left_boot_y, col=pants, boot_col=boots)

            # Torso + arms.
            torso_y = 6 if kind in ("walker", "screamer") else 5
            surf.fill(cloth, pygame.Rect(3, torso_y, 6, 6))
            surf.fill(cloth2, pygame.Rect(3, torso_y + 5, 6, 1))
            if kind == "walker":
                arm_y = torso_y
                reach = 4 if not idle else 3
                if direction == "down":
                    y0 = arm_y + (0 if stride > 0 else 1)
                else:
                    y0 = max(0, arm_y - reach - (0 if stride > 0 else 1))
                for ax in (3, 8):
                    surf.fill(skin, pygame.Rect(ax, y0, 1, reach))
                    surf.fill(skin, pygame.Rect(ax, y0 + reach, 1, 1))
                # Sleeves.
                surf.fill(cloth, pygame.Rect(3, arm_y, 1, 1))
                surf.fill(cloth, pygame.Rect(8, arm_y, 1, 1))
            else:
                arm_y = 6 if kind != "runner" else 5
                surf.fill(cloth, pygame.Rect(2, arm_y, 1, 3))
                surf.fill(cloth, pygame.Rect(9, arm_y, 1, 3))
                if kind == "screamer":
                    surf.fill(cloth, pygame.Rect(1, arm_y - 1, 1, 3))
                    surf.fill(cloth, pygame.Rect(10, arm_y - 1, 1, 3))

            # Head / face.
            head_y = 2 if kind in ("walker", "screamer") else 1
            if direction == "down":
                surf.fill(skin, pygame.Rect(4, head_y, 4, 4))
                surf.fill(outline, pygame.Rect(4, head_y, 4, 4), 1)
                surf.set_at((5, head_y + 2), eye)
                surf.set_at((6, head_y + 2), eye)
                if kind == "screamer":
                    surf.fill(blood, pygame.Rect(5, head_y + 3, 2, 1))
                else:
                    surf.set_at((6, head_y + 3), blood if kind == "runner" else outline)
            else:
                surf.fill(skin, pygame.Rect(4, head_y, 4, 4))
                surf.fill(outline, pygame.Rect(4, head_y, 4, 4), 1)
                surf.fill(cloth2, pygame.Rect(4, head_y, 4, 2))

            return surf

        # right / left (build right, flip elsewhere)
        # Legs.
        far_hip = 4
        near_hip = 7
        if kind in ("walker", "screamer"):
            lift = (1 if kind == "walker" else 2) if not idle else 0
            bend = (1 if kind == "walker" else 2) if not idle else 0
            far_knee_x = far_hip + (bend if step == 1 else 0)
            near_knee_x = near_hip - (bend if step == 0 else 0)
            near_boot_y = 15 - (lift if step == 0 else 0)
            far_boot_y = 15 - (lift if step == 1 else 0)
            draw_leg(far_hip, far_knee_x, far_knee_x, far_boot_y, col=pants_far, boot_col=boots_far)
            draw_leg(near_hip, near_knee_x, near_knee_x, near_boot_y, col=pants, boot_col=boots)
        else:
            near_dx = 1 if stride > 0 else -1
            far_dx = -near_dx
            near_boot_y = 15 - (1 if stride > 0 else 0)
            far_boot_y = 15 - (1 if stride < 0 else 0)
            draw_leg(far_hip, far_hip + far_dx, far_hip + far_dx, far_boot_y, col=pants_far, boot_col=boots_far)
            draw_leg(near_hip, near_hip + near_dx, near_hip + near_dx, near_boot_y, col=pants, boot_col=boots)

        # Body.
        body_y = 4 if kind == "runner" else 6 if kind in ("walker", "screamer") else 5
        surf.fill(cloth, pygame.Rect(4, body_y, 5, 7))
        surf.fill(cloth2, pygame.Rect(4, body_y + 5, 5, 1))
        # Arm / reach.
        if kind == "runner":
            surf.fill(cloth, pygame.Rect(8, body_y + 1, 3, 2))
            surf.fill(blood, pygame.Rect(10, body_y + 2, 1, 1))
        elif kind == "screamer":
            surf.fill(cloth, pygame.Rect(8, body_y, 3, 3))
            surf.fill(cloth, pygame.Rect(8, body_y + 3, 2, 2))
        else:
            # Walker: two forward-reaching arms (zombie style).
            arm_y0 = body_y + 2 + (0 if stride > 0 else 1)
            arm_y1 = body_y + 4 + (1 if stride > 0 else 0)
            ext0 = 4 if not idle else 3
            ext1 = 3 if not idle else 2
            surf.fill(skin, pygame.Rect(8, arm_y0, ext0, 1))
            surf.fill(skin, pygame.Rect(9, arm_y0 + 1, max(1, ext0 - 1), 1))
            surf.fill(skin, pygame.Rect(8, arm_y1, ext1, 1))
            surf.fill(skin, pygame.Rect(9, arm_y1 + 1, max(1, ext1 - 1), 1))
            # Sleeve pixel near torso.
            surf.fill(cloth, pygame.Rect(8, body_y + 2, 1, 1))

        # Head.
        head_y = 2 if kind in ("walker", "screamer") else 1
        surf.fill(skin, pygame.Rect(6, head_y, 4, 4))
        surf.fill(outline, pygame.Rect(6, head_y, 4, 4), 1)
        surf.set_at((9, head_y + 2), eye)
        if kind == "screamer":
            surf.fill(blood, pygame.Rect(8, head_y + 3, 2, 1))
        else:
            surf.set_at((9, head_y + 3), blood if kind == "runner" else outline)
        return surf

    _MONSTER_FRAMES: dict[str, dict[str, list[pygame.Surface]]] = {}
    for _kind in ("walker", "runner", "screamer"):
        _down = [
            _make_monster_sprite(_kind, "down", 0, idle=True),
            _make_monster_sprite(_kind, "down", 0, idle=False),
            _make_monster_sprite(_kind, "down", 1, idle=False),
        ]
        _up = [
            _make_monster_sprite(_kind, "up", 0, idle=True),
            _make_monster_sprite(_kind, "up", 0, idle=False),
            _make_monster_sprite(_kind, "up", 1, idle=False),
        ]
        _right = [
            _make_monster_sprite(_kind, "right", 0, idle=True),
            _make_monster_sprite(_kind, "right", 0, idle=False),
            _make_monster_sprite(_kind, "right", 1, idle=False),
        ]
        _left: list[pygame.Surface] = []
        for _spr in _right:
            _left.append(pygame.transform.flip(_spr, True, False))
        _MONSTER_FRAMES[_kind] = {"down": _down, "up": _up, "right": _right, "left": _left}

    @dataclass(frozen=True)
    class _CarModel:
        id: str
        name: str
        sprite_size: tuple[int, int]
        collider: tuple[int, int]
        wheelbase: float
        max_fwd: float
        max_rev: float
        accel: float
        brake: float
        fuel_per_px: float
        steer_max: float = 0.62

    _CAR_MODELS: dict[str, _CarModel] = {
        "rv": _CarModel(
            id="rv",
            name="房车",
            sprite_size=(52, 22),
            collider=(36, 14),
            wheelbase=32.0,
            max_fwd=105.0,
            max_rev=38.0,
            accel=165.0,
            brake=320.0,
            fuel_per_px=0.022,
            steer_max=0.52,
        ),
        "beetle": _CarModel(
            id="beetle",
            name="甲壳虫(风格)",
            sprite_size=(32, 22),
            collider=(22, 14),
            wheelbase=18.0,
            max_fwd=115.0,
            max_rev=48.0,
            accel=190.0,
            brake=320.0,
            fuel_per_px=0.012,
        ),
        "porsche": _CarModel(
            id="porsche",
            name="保时捷(风格)",
            sprite_size=(36, 18),
            collider=(26, 12),
            wheelbase=20.0,
            max_fwd=150.0,
            max_rev=55.0,
            accel=240.0,
            brake=380.0,
            fuel_per_px=0.014,
        ),
        "truck": _CarModel(
            id="truck",
            name="卡车",
            sprite_size=(44, 20),
            collider=(30, 14),
            wheelbase=26.0,
            max_fwd=110.0,
            max_rev=44.0,
            accel=175.0,
            brake=320.0,
            fuel_per_px=0.018,
        ),
        "schoolbus": _CarModel(
            id="schoolbus",
            name="校车",
            sprite_size=(50, 20),
            collider=(34, 14),
            wheelbase=30.0,
            max_fwd=105.0,
            max_rev=40.0,
            accel=165.0,
            brake=310.0,
            fuel_per_px=0.020,
        ),
        "lamborghini": _CarModel(
            id="lamborghini",
            name="兰博基尼(风格)",
            sprite_size=(36, 18),
            collider=(26, 12),
            wheelbase=20.0,
            max_fwd=165.0,
            max_rev=58.0,
            accel=260.0,
            brake=400.0,
            fuel_per_px=0.016,
        ),
        "mercedes": _CarModel(
            id="mercedes",
            name="奔驰(风格)",
            sprite_size=(38, 18),
            collider=(28, 12),
            wheelbase=22.0,
            max_fwd=135.0,
            max_rev=52.0,
            accel=220.0,
            brake=360.0,
            fuel_per_px=0.015,
        ),
    }

    def _draw_wheel(
        surf: pygame.Surface,
        cx: int,
        cy: int,
        *,
        r: int,
        frame: int,
        outline: tuple[int, int, int],
        steer_state: int = 0,
        is_front: bool = False,
    ) -> None:
        # Top-down wheel: compact contact patch with a steer hint (MiniDayZ-like feel,
        # but original art).
        r = max(1, int(r))
        cx = int(cx)
        cy = int(cy)
        steer_state = int(clamp(int(steer_state), -1, 1))

        tire = (14, 14, 18)
        rim = (64, 64, 76)
        hi = (44, 44, 54)

        length = max(5, int(r * 3))
        thickness = max(2, int(r))
        half_l = int(length // 2)
        half_t = int(thickness // 2)
        br = max(1, int(min(length, thickness) // 2))

        def draw_patch(rect: pygame.Rect) -> None:
            pygame.draw.rect(surf, tire, rect, border_radius=br)
            inner = rect.inflate(-2, -2)
            if inner.w > 0 and inner.h > 0:
                pygame.draw.rect(surf, rim, inner, border_radius=max(1, int(min(inner.w, inner.h) // 2)))
            # Spinning hint.
            if int(frame) % 2 == 0:
                pygame.draw.line(surf, hi, (rect.left + 1, rect.centery), (rect.right - 2, rect.centery), 1)
            else:
                pygame.draw.line(surf, hi, (rect.centerx, rect.top + 1), (rect.centerx, rect.bottom - 2), 1)
            pygame.draw.rect(surf, outline, rect, 1, border_radius=br)

        if is_front and steer_state != 0:
            # A simple slanted patch to imply steering.
            sl = 2 if r >= 3 else 1
            sl *= 1 if steer_state > 0 else -1
            pts = [
                (cx - half_l + sl, cy - half_t),
                (cx + half_l + sl, cy - half_t),
                (cx + half_l - sl, cy + half_t),
                (cx - half_l - sl, cy + half_t),
            ]
            pygame.draw.polygon(surf, tire, pts)
            ipts = [
                (cx - max(1, half_l - 2) + sl, cy - max(0, half_t - 1)),
                (cx + max(1, half_l - 2) + sl, cy - max(0, half_t - 1)),
                (cx + max(1, half_l - 2) - sl, cy + max(0, half_t - 1)),
                (cx - max(1, half_l - 2) - sl, cy + max(0, half_t - 1)),
            ]
            pygame.draw.polygon(surf, rim, ipts)
            pygame.draw.polygon(surf, outline, pts, 1)
            # Steer direction accent.
            if steer_state > 0:
                pygame.draw.line(surf, hi, (cx + half_l - 1, cy - half_t), (cx + half_l - 3, cy + half_t), 1)
            else:
                pygame.draw.line(surf, hi, (cx - half_l + 1, cy - half_t), (cx - half_l + 3, cy + half_t), 1)
            return

        rect = pygame.Rect(int(cx - half_l), int(cy - half_t), int(length), int(thickness))
        draw_patch(rect)

    def _make_car_base(
        model_id: str,
        frame: int,
        steer_state: int,
        _car_models: dict[str, "HardcoreSurvivalState._CarModel"] = _CAR_MODELS,
        _draw_wheel_fn: Callable[..., None] = _draw_wheel,
    ) -> pygame.Surface:
        model = _car_models.get(str(model_id))
        if model is None:
            model = _car_models["beetle"]
        w, h = int(model.sprite_size[0]), int(model.sprite_size[1])
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        outline = (12, 12, 16)
        glass = (120, 140, 160)
        glass2 = (160, 180, 200)
        light = (255, 230, 150)
        tail = (255, 110, 110)

        steer_state = int(clamp(int(steer_state), -1, 1))

        def tint(c: tuple[int, int, int], add: tuple[int, int, int]) -> tuple[int, int, int]:
            return (
                int(clamp(int(c[0]) + int(add[0]), 0, 255)),
                int(clamp(int(c[1]) + int(add[1]), 0, 255)),
                int(clamp(int(c[2]) + int(add[2]), 0, 255)),
            )

        wheel_centers: list[tuple[int, int]] = []
        wheel_r = 3

        if model.id == "rv":
            body = (236, 236, 242)
            body2 = tint(body, (-30, -30, -32))
            body3 = tint(body, (10, 10, 12))
            stripe = (210, 120, 80)

            shell = pygame.Rect(2, 3, w - 4, h - 6)
            coach = pygame.Rect(shell.x, shell.y, shell.w - 18, shell.h)
            cab_left = int(coach.right - 2)
            cab_top = int(shell.y + 2)
            cab_bottom = int(shell.bottom - 3)
            front = int(shell.right - 1)
            midy = int((cab_top + cab_bottom) // 2)

            pygame.draw.rect(surf, body, coach, border_radius=4)
            pygame.draw.rect(surf, outline, coach, 2, border_radius=4)
            # Rear bumper + ladder hint.
            pygame.draw.rect(surf, body2, pygame.Rect(coach.x + 1, coach.y + 1, 3, coach.h - 2), border_radius=2)
            pygame.draw.line(surf, outline, (coach.x + 3, coach.y + 3), (coach.x + 3, coach.bottom - 4), 1)
            pygame.draw.line(surf, outline, (coach.x + 2, coach.y + 5), (coach.x + 4, coach.y + 5), 1)
            pygame.draw.line(surf, outline, (coach.x + 2, coach.y + 8), (coach.x + 4, coach.y + 8), 1)

            # Roof on the coach.
            roof = pygame.Rect(coach.x + 6, coach.y + 2, coach.w - 14, coach.h - 4)
            pygame.draw.rect(surf, body3, roof, border_radius=3)
            pygame.draw.rect(surf, outline, roof, 1, border_radius=3)
            pygame.draw.rect(surf, tint(body, (-10, -10, -10)), pygame.Rect(roof.x + 8, roof.y + 2, 6, 3), border_radius=1)  # vent
            pygame.draw.rect(surf, tint(body, (-10, -10, -10)), pygame.Rect(roof.right - 12, roof.y + 2, 8, 3), border_radius=1)  # AC

            # Side windows (coach).
            wx0 = coach.x + 10
            for i in range(4):
                rwin = pygame.Rect(wx0 + i * 8, coach.y + 3, 6, 4)
                pygame.draw.rect(surf, glass, rwin, border_radius=2)
                pygame.draw.rect(surf, outline, rwin, 1, border_radius=2)

            # Cab (tapered nose).
            cab_poly = [
                (cab_left, cab_top),
                (front - 3, cab_top + 1),
                (front, midy),
                (front - 3, cab_bottom - 1),
                (cab_left, cab_bottom),
            ]
            pygame.draw.polygon(surf, body2, cab_poly)
            pygame.draw.polygon(surf, outline, cab_poly, 1)
            # Windshield.
            wind = [
                (front - 8, cab_top + 2),
                (front - 2, cab_top + 3),
                (front - 2, cab_bottom - 3),
                (front - 8, cab_bottom - 2),
            ]
            pygame.draw.polygon(surf, glass, wind)
            pygame.draw.polygon(surf, outline, wind, 1)
            pygame.draw.line(surf, glass2, (front - 7, cab_top + 3), (front - 3, cab_top + 3), 1)

            # Side stripe + door hint.
            pygame.draw.rect(surf, stripe, pygame.Rect(coach.x + 6, coach.bottom - 5, coach.w + 6, 2), border_radius=1)
            door_x = int(coach.right - 10)
            pygame.draw.line(surf, outline, (door_x, coach.y + 4), (door_x, coach.bottom - 5), 1)
            pygame.draw.circle(surf, outline, (door_x + 2, midy + 2), 1)

            # Lights.
            pygame.draw.circle(surf, light, (front, cab_top + 2), 1)
            pygame.draw.circle(surf, light, (front, cab_bottom - 2), 1)
            pygame.draw.circle(surf, tail, (coach.x + 1, coach.y + 2), 1)
            pygame.draw.circle(surf, tail, (coach.x + 1, coach.bottom - 3), 1)

            y_top = int(shell.y + 1)
            y_bottom = int(shell.bottom - 2)
            wheel_centers = [(12, y_top), (12, y_bottom), (w - 10, y_top), (w - 10, y_bottom)]
            wheel_r = 3
        elif model.id == "beetle":
            body = (84, 170, 190)
            body2 = tint(body, (-22, -22, -18))
            shell = pygame.Rect(3, 4, w - 6, h - 8)
            pygame.draw.rect(surf, body, shell, border_radius=10)
            pygame.draw.rect(surf, outline, shell, 2, border_radius=10)
            roof = pygame.Rect(shell.x + 9, shell.y + 2, shell.w - 18, shell.h - 4)
            pygame.draw.rect(surf, body2, roof, border_radius=8)
            win = roof.inflate(-2, -2)
            pygame.draw.rect(surf, glass, win, border_radius=7)
            pygame.draw.rect(surf, outline, win, 1, border_radius=7)
            pygame.draw.line(surf, outline, (shell.x + 7, shell.y + 2), (shell.x + 7, shell.bottom - 3), 1)
            pygame.draw.line(surf, outline, (shell.right - 8, shell.y + 2), (shell.right - 8, shell.bottom - 3), 1)
            pygame.draw.circle(surf, light, (w - 4, h // 2 - 3), 1)
            pygame.draw.circle(surf, light, (w - 4, h // 2 + 3), 1)
            pygame.draw.circle(surf, tail, (3, h // 2 - 3), 1)
            pygame.draw.circle(surf, tail, (3, h // 2 + 3), 1)
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(7, y_top), (7, y_bottom), (w - 7, y_top), (w - 7, y_bottom)]
            wheel_r = 3
        elif model.id == "porsche":
            body = (220, 80, 80)
            body2 = tint(body, (-40, -32, -32))
            shell = pygame.Rect(3, 4, w - 6, h - 8)
            pygame.draw.rect(surf, body, shell, border_radius=6)
            pygame.draw.rect(surf, outline, shell, 2, border_radius=6)
            hood = pygame.Rect(shell.right - 14, shell.y + 1, 10, shell.h - 2)
            pygame.draw.rect(surf, body2, hood, border_radius=5)
            roof = pygame.Rect(shell.x + 12, shell.y + 2, shell.w - 26, shell.h - 4)
            pygame.draw.rect(surf, tint(body, (-12, -12, -12)), roof, border_radius=4)
            win = roof.inflate(-2, -2)
            pygame.draw.rect(surf, glass, win, border_radius=3)
            pygame.draw.rect(surf, outline, win, 1, border_radius=3)
            pygame.draw.line(surf, glass2, (win.x + 2, win.y + 2), (win.right - 3, win.y + 2), 1)
            pygame.draw.circle(surf, light, (w - 4, 6), 1)
            pygame.draw.circle(surf, light, (w - 4, h - 7), 1)
            pygame.draw.circle(surf, tail, (3, 6), 1)
            pygame.draw.circle(surf, tail, (3, h - 7), 1)
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(9, y_top), (9, y_bottom), (w - 7, y_top), (w - 7, y_bottom)]
            wheel_r = 2
        elif model.id == "lamborghini":
            body = (86, 190, 96)
            body2 = tint(body, (-28, -38, -28))
            shell = pygame.Rect(3, 4, w - 6, h - 8)
            pygame.draw.rect(surf, body, shell, border_radius=4)
            pygame.draw.rect(surf, outline, shell, 2, border_radius=4)
            # Wedge nose.
            pygame.draw.polygon(surf, body2, [(w - 4, h // 2), (w - 14, shell.y + 1), (w - 14, shell.bottom - 2)])
            # Engine cover vents.
            eng = pygame.Rect(shell.x + 4, shell.y + 2, 10, shell.h - 4)
            pygame.draw.rect(surf, body2, eng, border_radius=2)
            for y in range(eng.y + 1, eng.bottom - 1, 2):
                pygame.draw.line(surf, outline, (eng.x + 1, y), (eng.right - 2, y), 1)
            roof = pygame.Rect(shell.x + 14, shell.y + 2, shell.w - 28, shell.h - 4)
            pygame.draw.polygon(
                surf,
                glass,
                [(roof.x, roof.y + 1), (roof.right, roof.y + 2), (roof.right - 2, roof.bottom - 2), (roof.x + 2, roof.bottom - 1)],
            )
            pygame.draw.polygon(
                surf,
                outline,
                [(roof.x, roof.y + 1), (roof.right, roof.y + 2), (roof.right - 2, roof.bottom - 2), (roof.x + 2, roof.bottom - 1)],
                1,
            )
            pygame.draw.circle(surf, light, (w - 4, 6), 1)
            pygame.draw.circle(surf, light, (w - 4, h - 7), 1)
            pygame.draw.circle(surf, tail, (3, 6), 1)
            pygame.draw.circle(surf, tail, (3, h - 7), 1)
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(9, y_top), (9, y_bottom), (w - 7, y_top), (w - 7, y_bottom)]
            wheel_r = 2
        elif model.id == "mercedes":
            body = (150, 150, 160)
            body2 = tint(body, (-34, -34, -34))
            shell = pygame.Rect(3, 4, w - 6, h - 8)
            pygame.draw.rect(surf, body, shell, border_radius=5)
            pygame.draw.rect(surf, outline, shell, 2, border_radius=5)
            hood = pygame.Rect(shell.right - 15, shell.y + 1, 10, shell.h - 2)
            pygame.draw.rect(surf, body2, hood, border_radius=4)
            roof = pygame.Rect(shell.x + 11, shell.y + 2, shell.w - 24, shell.h - 4)
            pygame.draw.rect(surf, tint(body, (8, 8, 10)), roof, border_radius=4)
            win = roof.inflate(-2, -2)
            pygame.draw.rect(surf, glass, win, border_radius=3)
            pygame.draw.rect(surf, outline, win, 1, border_radius=3)
            pygame.draw.circle(surf, light, (w - 4, 6), 1)
            pygame.draw.circle(surf, light, (w - 4, h - 7), 1)
            pygame.draw.circle(surf, tail, (3, 6), 1)
            pygame.draw.circle(surf, tail, (3, h - 7), 1)
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(10, y_top), (10, y_bottom), (w - 6, y_top), (w - 6, y_bottom)]
            wheel_r = 2
        elif model.id == "truck":
            cab = (220, 220, 230)
            cab2 = tint(cab, (-26, -26, -26))
            bed = (92, 92, 104)
            bed2 = tint(bed, (-22, -22, -22))
            cab_w = 16
            bed_r = pygame.Rect(3, 4, w - cab_w - 5, h - 8)
            pygame.draw.rect(surf, bed, bed_r, border_radius=4)
            pygame.draw.rect(surf, outline, bed_r, 2, border_radius=4)
            pygame.draw.rect(surf, bed2, pygame.Rect(bed_r.x + 1, bed_r.y + bed_r.h - 4, bed_r.w - 2, 2), border_radius=2)
            for x in range(bed_r.x + 3, bed_r.right - 2, 4):
                pygame.draw.line(surf, outline, (x, bed_r.y + 2), (x, bed_r.bottom - 3), 1)
            cab_r = pygame.Rect(w - cab_w - 2, 3, cab_w, h - 6)
            pygame.draw.rect(surf, cab, cab_r, border_radius=4)
            pygame.draw.rect(surf, outline, cab_r, 2, border_radius=4)
            wind = pygame.Rect(cab_r.x + 3, cab_r.y + 3, cab_r.w - 6, 6)
            pygame.draw.rect(surf, glass, wind, border_radius=2)
            pygame.draw.rect(surf, outline, wind, 1, border_radius=2)
            pygame.draw.rect(surf, cab2, pygame.Rect(cab_r.x + 2, cab_r.bottom - 6, cab_r.w - 4, 3), border_radius=2)
            pygame.draw.circle(surf, light, (w - 3, 6), 1)
            pygame.draw.circle(surf, light, (w - 3, h - 7), 1)
            pygame.draw.circle(surf, tail, (3, 6), 1)
            pygame.draw.circle(surf, tail, (3, h - 7), 1)
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(11, y_top), (11, y_bottom), (w - 7, y_top), (w - 7, y_bottom)]
            wheel_r = 3
        else:
            # schoolbus
            body = (230, 200, 90)
            body2 = tint(body, (-28, -28, -18))
            stripe = (80, 80, 86)
            shell = pygame.Rect(2, 4, w - 4, h - 8)
            pygame.draw.rect(surf, body, shell, border_radius=3)
            pygame.draw.rect(surf, outline, shell, 2, border_radius=3)
            pygame.draw.rect(surf, body2, pygame.Rect(shell.x + 1, shell.y + shell.h - 4, shell.w - 2, 2), border_radius=2)
            pygame.draw.rect(surf, stripe, pygame.Rect(shell.x + 6, shell.bottom - 6, shell.w - 12, 2), border_radius=1)
            wx0 = shell.x + 10
            for i in range(6):
                r = pygame.Rect(wx0 + i * 6, shell.y + 2, 4, 4)
                pygame.draw.rect(surf, glass, r, border_radius=1)
                pygame.draw.rect(surf, outline, r, 1, border_radius=1)
            pygame.draw.line(surf, outline, (w - 16, shell.y + 2), (w - 16, shell.bottom - 3), 1)  # door hint
            pygame.draw.circle(surf, light, (w - 3, shell.y + 2), 1)
            pygame.draw.circle(surf, light, (w - 3, shell.bottom - 3), 1)       
            pygame.draw.circle(surf, tail, (shell.x + 1, shell.y + 2), 1)       
            pygame.draw.circle(surf, tail, (shell.x + 1, shell.bottom - 3), 1)  
            y_top = 4
            y_bottom = h - 5
            wheel_centers = [(12, y_top), (12, y_bottom), (w - 8, y_top), (w - 8, y_bottom)]
            wheel_r = 3

        front_x = max(int(cx) for cx, _ in wheel_centers) - 1 if wheel_centers else (w - 6)
        for cx, cy in wheel_centers:
            is_front = int(cx) >= int(front_x)
            _draw_wheel_fn(
                surf,
                int(cx),
                int(cy),
                r=int(wheel_r),
                frame=int(frame),
                outline=outline,
                steer_state=int(steer_state),
                is_front=is_front,
            )

        return surf

    def _make_shadow(sprite: pygame.Surface, *, alpha: int = 110) -> pygame.Surface:
        w, h = sprite.get_size()
        shadow = pygame.Surface((w, h), pygame.SRCALPHA)
        alpha = int(clamp(int(alpha), 0, 255))
        for y in range(int(h)):
            for x in range(int(w)):
                r, g, b, a = sprite.get_at((x, y))
                if a == 0:
                    continue
                shadow.set_at((x, y), (0, 0, 0, alpha))
        return shadow

    _CAR_BASE: dict[tuple[str, int, int], pygame.Surface] = {}
    _CAR_SHADOW: dict[tuple[str, int, int], pygame.Surface] = {}
    for _mid in _CAR_MODELS:
        for _steer in (-1, 0, 1):
            for _frm in (0, 1):
                _base = _make_car_base(_mid, _frm, _steer)
                _CAR_BASE[(_mid, _steer, _frm)] = _base
                _CAR_SHADOW[(_mid, _steer, _frm)] = _make_shadow(_base, alpha=110)

    def _make_bike_sprite(direction: str, frame: int) -> pygame.Surface:        
        w, h = 18, 18
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        outline = (10, 10, 12)
        tire = (16, 16, 20)
        rim = (62, 62, 72)
        spoke = (130, 130, 142)
        frame_col = (96, 96, 110)
        frame_dark = (72, 72, 86)
        seat = (44, 44, 54)
        light = (255, 230, 150)
        moving = int(frame) % 2 == 1

        def draw_wheel(cx: int, cy: int, *, orient: str) -> None:
            cx = int(cx)
            cy = int(cy)
            # Round wheels (top-down look).
            outer = 3
            inner = 2
            pygame.draw.circle(surf, tire, (cx, cy), int(outer))
            pygame.draw.circle(surf, rim, (cx, cy), int(inner))
            pygame.draw.circle(surf, outline, (cx, cy), int(outer), 1)
            pygame.draw.circle(surf, (90, 90, 104), (cx, cy), 1)  # hub

            if moving:
                pygame.draw.line(surf, spoke, (cx - 2, cy), (cx + 2, cy), 1)
                pygame.draw.line(surf, spoke, (cx, cy - 2), (cx, cy + 2), 1)
            else:
                pygame.draw.line(surf, spoke, (cx - 2, cy - 2), (cx + 2, cy + 2), 1)
                pygame.draw.line(surf, spoke, (cx - 2, cy + 2), (cx + 2, cy - 2), 1)

        if direction in ("left", "right"):
            y = 10
            front_x = 14 if direction == "right" else 4
            rear_x = 4 if direction == "right" else 14

            draw_wheel(rear_x, y, orient="h")
            draw_wheel(front_x, y, orient="h")

            crank_x, crank_y = 9, 10
            seat_x, seat_y = 8, 6
            handle_y = 6

            pygame.draw.line(surf, frame_col, (rear_x, y), (crank_x, crank_y), 1)
            pygame.draw.line(surf, frame_col, (crank_x, crank_y), (front_x, y), 1)
            pygame.draw.line(surf, frame_col, (crank_x, crank_y), (seat_x, seat_y), 1)
            pygame.draw.line(surf, frame_col, (seat_x, seat_y), (rear_x + (2 if direction == "right" else -2), y - 1), 1)

            pygame.draw.rect(surf, seat, pygame.Rect(seat_x - 2, seat_y - 1, 5, 2), border_radius=1)
            pygame.draw.line(surf, outline, (seat_x - 2, seat_y), (seat_x + 2, seat_y), 1)

            # Fork + handlebar.
            pygame.draw.line(surf, frame_dark, (front_x, y), (front_x + (0 if direction == "right" else 0), handle_y + 2), 1)
            pygame.draw.line(surf, frame_dark, (front_x - 2, handle_y), (front_x + 2, handle_y), 1)
            pygame.draw.line(surf, frame_dark, (front_x, handle_y), (front_x, handle_y + 2), 1)

            # Pedal hint.
            if moving:
                surf.set_at((crank_x + (1 if direction == "right" else -1), crank_y), spoke)
                surf.set_at((crank_x, crank_y + 1), spoke)
            else:
                surf.set_at((crank_x, crank_y + 1), spoke)
                surf.set_at((crank_x + (1 if direction == "right" else -1), crank_y + 1), spoke)

            pygame.draw.circle(surf, light, (front_x + (2 if direction == "right" else -2), y - 2), 1)
        else:
            x = 9
            front_y = 4 if direction == "up" else 14
            rear_y = 14 if direction == "up" else 4

            draw_wheel(x, rear_y, orient="v")
            draw_wheel(x, front_y, orient="v")

            crank_x, crank_y = 8, 9
            seat_x, seat_y = 6, 8
            handle_x = x

            pygame.draw.line(surf, frame_col, (x, rear_y), (crank_x, crank_y), 1)
            pygame.draw.line(surf, frame_col, (crank_x, crank_y), (x, front_y), 1)
            pygame.draw.line(surf, frame_col, (crank_x, crank_y), (seat_x, seat_y), 1)
            pygame.draw.line(surf, frame_col, (seat_x, seat_y), (x - 1, rear_y + (-2 if direction == "up" else 2)), 1)

            pygame.draw.rect(surf, seat, pygame.Rect(seat_x - 2, seat_y - 1, 5, 2), border_radius=1)
            pygame.draw.line(surf, outline, (seat_x - 2, seat_y), (seat_x + 2, seat_y), 1)

            pygame.draw.line(surf, frame_dark, (x, front_y), (x + 2, front_y + (2 if direction == "down" else -2)), 1)
            pygame.draw.line(surf, frame_dark, (handle_x - 2, front_y), (handle_x + 2, front_y), 1)
            pygame.draw.line(surf, frame_dark, (handle_x, front_y - 1), (handle_x, front_y + 1), 1)

            if moving:
                surf.set_at((crank_x + 1, crank_y), spoke)
                surf.set_at((crank_x, crank_y + (1 if direction == "down" else -1)), spoke)
            else:
                surf.set_at((crank_x, crank_y + 1), spoke)
                surf.set_at((crank_x + 1, crank_y + 1), spoke)

            pygame.draw.circle(surf, light, (x + 3, front_y + (3 if direction == "down" else -3)), 1)

        return surf

    _BIKE_FRAMES: dict[str, list[pygame.Surface]] = {
        "up": [_make_bike_sprite("up", 0), _make_bike_sprite("up", 1)],
        "down": [_make_bike_sprite("down", 0), _make_bike_sprite("down", 1)],   
        "left": [_make_bike_sprite("left", 0), _make_bike_sprite("left", 1)],   
        "right": [_make_bike_sprite("right", 0), _make_bike_sprite("right", 1)],
    }

    def _recolor_sprite(
        src: pygame.Surface,
        mapping: dict[tuple[int, int, int, int], tuple[int, int, int, int]],
    ) -> pygame.Surface:
        out = src.copy()
        w, h = out.get_size()
        for y in range(int(h)):
            for x in range(int(w)):
                rgba = out.get_at((x, y))
                repl = mapping.get((int(rgba.r), int(rgba.g), int(rgba.b), int(rgba.a)))
                if repl is not None:
                    out.set_at((x, y), repl)
        return out

    def _bike_variant_from_base(
        base: dict[str, list[pygame.Surface]],
        mapping: dict[tuple[int, int, int, int], tuple[int, int, int, int]],
        *,
        basket: bool = False,
        battery: bool = False,
    ) -> dict[str, list[pygame.Surface]]:
        out: dict[str, list[pygame.Surface]] = {}
        basket_col = (130, 92, 62, 255)
        basket_line = (10, 10, 12, 255)
        battery_col = (70, 90, 140, 255)
        for d, frames in base.items():
            nf: list[pygame.Surface] = []
            for s in frames:
                ns = s.copy()
                w, h = ns.get_size()
                for yy in range(int(h)):
                    for xx in range(int(w)):
                        rgba = ns.get_at((xx, yy))
                        repl = mapping.get((int(rgba.r), int(rgba.g), int(rgba.b), int(rgba.a)))
                        if repl is not None:
                            ns.set_at((xx, yy), repl)
                if basket:
                    if d == "right":
                        pygame.draw.rect(ns, basket_col, pygame.Rect(14, 6, 3, 3), border_radius=1)
                        pygame.draw.rect(ns, basket_line, pygame.Rect(14, 6, 3, 3), 1, border_radius=1)
                    elif d == "left":
                        pygame.draw.rect(ns, basket_col, pygame.Rect(1, 6, 3, 3), border_radius=1)
                        pygame.draw.rect(ns, basket_line, pygame.Rect(1, 6, 3, 3), 1, border_radius=1)
                    elif d == "up":
                        pygame.draw.rect(ns, basket_col, pygame.Rect(12, 1, 3, 3), border_radius=1)
                        pygame.draw.rect(ns, basket_line, pygame.Rect(12, 1, 3, 3), 1, border_radius=1)
                    else:
                        pygame.draw.rect(ns, basket_col, pygame.Rect(12, 14, 3, 3), border_radius=1)
                        pygame.draw.rect(ns, basket_line, pygame.Rect(12, 14, 3, 3), 1, border_radius=1)
                if battery:
                    pygame.draw.rect(ns, battery_col, pygame.Rect(8, 9, 4, 3), border_radius=1)
                    pygame.draw.rect(ns, basket_line, pygame.Rect(8, 9, 4, 3), 1, border_radius=1)
                nf.append(ns)
            out[str(d)] = nf
        return out

    def _make_moto_sprite(
        direction: str,
        frame: int,
        *,
        body: tuple[int, int, int] = (110, 80, 80),
        body2: tuple[int, int, int] = (72, 50, 50),
        accent: tuple[int, int, int] = (240, 210, 140),
        long: bool = False,
    ) -> pygame.Surface:
        w, h = 22, 22
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        outline = (10, 10, 12)
        tire = (16, 16, 20)
        rim = (62, 62, 72)
        spoke = (130, 130, 142)
        moving = int(frame) % 2 == 1

        def draw_wheel(cx: int, cy: int) -> None:
            pygame.draw.circle(surf, tire, (int(cx), int(cy)), 4)
            pygame.draw.circle(surf, rim, (int(cx), int(cy)), 3)
            pygame.draw.circle(surf, outline, (int(cx), int(cy)), 4, 1)
            if moving:
                pygame.draw.line(surf, spoke, (cx - 2, cy), (cx + 2, cy), 1)
                pygame.draw.line(surf, spoke, (cx, cy - 2), (cx, cy + 2), 1)
            else:
                pygame.draw.line(surf, spoke, (cx - 2, cy - 2), (cx + 2, cy + 2), 1)
                pygame.draw.line(surf, spoke, (cx - 2, cy + 2), (cx + 2, cy - 2), 1)

        if direction in ("left", "right"):
            y = 12
            front_x = 17 if direction == "right" else 4
            rear_x = 5 if direction == "right" else 18
            draw_wheel(rear_x, y)
            draw_wheel(front_x, y)
            body_r = pygame.Rect(min(front_x, rear_x) + 2, y - 4, abs(front_x - rear_x) - 3, 7)
            if long:
                body_r.w += 2
                body_r.x -= 1
            pygame.draw.rect(surf, body, body_r, border_radius=3)
            pygame.draw.rect(surf, outline, body_r, 1, border_radius=3)
            pygame.draw.rect(surf, body2, pygame.Rect(body_r.x + 2, body_r.y + 1, max(1, body_r.w - 4), 2), border_radius=2)
            pygame.draw.rect(surf, accent, pygame.Rect(body_r.centerx - 2, body_r.y + 3, 4, 2), border_radius=1)
        else:
            x = 11
            front_y = 4 if direction == "up" else 17
            rear_y = 18 if direction == "up" else 5
            draw_wheel(x, rear_y)
            draw_wheel(x, front_y)
            body_r = pygame.Rect(x - 4, min(front_y, rear_y) + 2, 7, abs(front_y - rear_y) - 3)
            if long:
                body_r.h += 2
                body_r.y -= 1
            pygame.draw.rect(surf, body, body_r, border_radius=3)
            pygame.draw.rect(surf, outline, body_r, 1, border_radius=3)
            pygame.draw.rect(surf, body2, pygame.Rect(body_r.x + 1, body_r.y + 2, 2, max(1, body_r.h - 4)), border_radius=2)
            pygame.draw.rect(surf, accent, pygame.Rect(body_r.x + 3, body_r.centery - 2, 2, 4), border_radius=1)
        return surf

    _TWO_WHEEL_FRAMES: dict[str, dict[str, list[pygame.Surface]]] = {
        "bike": _BIKE_FRAMES,
        "bike_lady": _bike_variant_from_base(
            _BIKE_FRAMES,
            {
                (96, 96, 110, 255): (220, 110, 180, 255),
                (72, 72, 86, 255): (170, 70, 140, 255),
                (44, 44, 54, 255): (60, 40, 50, 255),
            },
            basket=True,
        ),
        "bike_mountain": _bike_variant_from_base(
            _BIKE_FRAMES,
            {
                (96, 96, 110, 255): (90, 170, 110, 255),
                (72, 72, 86, 255): (60, 130, 80, 255),
                (44, 44, 54, 255): (30, 40, 34, 255),
            },
        ),
        "bike_auto": _bike_variant_from_base(
            _BIKE_FRAMES,
            {
                (96, 96, 110, 255): (90, 140, 210, 255),
                (72, 72, 86, 255): (60, 100, 170, 255),
                (44, 44, 54, 255): (30, 40, 60, 255),
            },
            battery=True,
        ),
        "moto": {
            "up": [_make_moto_sprite("up", 0), _make_moto_sprite("up", 1)],
            "down": [_make_moto_sprite("down", 0), _make_moto_sprite("down", 1)],
            "right": [_make_moto_sprite("right", 0), _make_moto_sprite("right", 1)],
            "left": [pygame.transform.flip(_make_moto_sprite("right", 0), True, False), pygame.transform.flip(_make_moto_sprite("right", 1), True, False)],
        },
        "moto_lux": {
            "up": [_make_moto_sprite("up", 0, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90)), _make_moto_sprite("up", 1, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90))],
            "down": [_make_moto_sprite("down", 0, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90)), _make_moto_sprite("down", 1, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90))],
            "right": [_make_moto_sprite("right", 0, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90)), _make_moto_sprite("right", 1, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90))],
            "left": [pygame.transform.flip(_make_moto_sprite("right", 0, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90)), True, False), pygame.transform.flip(_make_moto_sprite("right", 1, body=(40, 40, 46), body2=(26, 26, 32), accent=(220, 180, 90)), True, False)],
        },
        "moto_long": {
            "up": [_make_moto_sprite("up", 0, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True), _make_moto_sprite("up", 1, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True)],
            "down": [_make_moto_sprite("down", 0, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True), _make_moto_sprite("down", 1, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True)],
            "right": [_make_moto_sprite("right", 0, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True), _make_moto_sprite("right", 1, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True)],
            "left": [pygame.transform.flip(_make_moto_sprite("right", 0, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True), True, False), pygame.transform.flip(_make_moto_sprite("right", 1, body=(110, 90, 70), body2=(70, 60, 46), accent=(240, 210, 140), long=True), True, False)],
        },
    }

    def _make_cyclist_sprite(
        direction: str,
        frame: int,
        *,
        pal=_PLAYER_PAL,
    ) -> pygame.Surface:
        # A seated variant of the player used when riding the bike (2 pedal frames).
        w, h = 12, 16
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        hair = pal["H"]
        skin = pal["S"]
        coat = pal["C"]
        pants = pal["P"]
        boots = pal["B"]
        eye = pal["E"]
        coat_dark = (56, 72, 140)
        pants_dark = (40, 40, 50)

        phase = int(frame) % 2

        def draw_leg(x: int, y0: int, y1: int, *, col: tuple[int, int, int]) -> None:
            x = int(clamp(int(x), 0, w - 1))
            y0 = int(clamp(int(y0), 0, h - 1))
            y1 = int(clamp(int(y1), 0, h - 1))
            if y1 < y0:
                y0, y1 = y1, y0
            for y in range(y0, y1):
                surf.fill(col, pygame.Rect(x, y, 1, 1))
            surf.fill(boots, pygame.Rect(x, y1, 1, 1))

        if direction in ("down", "up"):
            # Head.
            if direction == "down":
                surf.fill(hair, pygame.Rect(4, 0, 4, 2))
                surf.fill(skin, pygame.Rect(4, 2, 4, 3))
                surf.set_at((5, 3), eye)
                surf.set_at((6, 3), eye)
            else:
                surf.fill(hair, pygame.Rect(4, 0, 4, 3))
                surf.fill(skin, pygame.Rect(5, 3, 2, 1))

            # Torso (lower + leaning).
            surf.fill(coat, pygame.Rect(3, 5, 6, 5))
            surf.fill(coat_dark, pygame.Rect(4, 7, 4, 3))
            # Arms forward to a "handlebar" hint.
            surf.fill(coat, pygame.Rect(2, 6, 1, 2))
            surf.fill(coat, pygame.Rect(9, 6, 1, 2))
            surf.fill(skin, pygame.Rect(2, 8, 1, 1))
            surf.fill(skin, pygame.Rect(9, 8, 1, 1))

            # Legs pedal around the crank: alternate boot height.
            lx = 5
            rx = 6
            if direction == "down":
                l_y = 13 + (0 if phase == 0 else 1)
                r_y = 13 + (1 if phase == 0 else 0)
            else:
                l_y = 14 - (0 if phase == 0 else 1)
                r_y = 14 - (1 if phase == 0 else 0)
            draw_leg(lx, 11, l_y, col=pants)
            draw_leg(rx, 11, r_y, col=pants_dark)
            surf.fill((30, 30, 34), pygame.Rect(4, 10, 4, 1))  # seat hint
            return surf

        # right
        # Head.
        surf.fill(hair, pygame.Rect(4, 0, 4, 2))
        surf.fill(skin, pygame.Rect(6, 2, 3, 3))
        surf.set_at((8, 3), eye)
        # Torso + near arm.
        surf.fill(coat, pygame.Rect(4, 5, 5, 5))
        surf.fill(coat_dark, pygame.Rect(4, 9, 5, 1))
        surf.fill(coat, pygame.Rect(8, 6, 2, 1))
        surf.fill(skin, pygame.Rect(9, 7, 1, 1))

        # Legs (2-frame pedal hint).
        near_shift = 1 if phase == 0 else 0
        far_shift = 0 if phase == 0 else 1
        draw_leg(5 + far_shift, 10, 14, col=pants_dark)
        draw_leg(6 + near_shift, 10, 13, col=pants)
        return surf

    _CYC_DOWN_FRAMES = [_make_cyclist_sprite("down", 0), _make_cyclist_sprite("down", 1)]
    _CYC_UP_FRAMES = [_make_cyclist_sprite("up", 0), _make_cyclist_sprite("up", 1)]
    _CYC_RIGHT_FRAMES = [_make_cyclist_sprite("right", 0), _make_cyclist_sprite("right", 1)]
    _CYC_LEFT_FRAMES: list[pygame.Surface] = []
    for _s in _CYC_RIGHT_FRAMES:
        _CYC_LEFT_FRAMES.append(pygame.transform.flip(_s, True, False))

    _CYCLIST_FRAMES: dict[str, list[pygame.Surface]] = {
        "down": _CYC_DOWN_FRAMES,
        "up": _CYC_UP_FRAMES,
        "right": _CYC_RIGHT_FRAMES,
        "left": _CYC_LEFT_FRAMES,
    }

    _RV_INT_LAYOUT: list[str] = [
        "WWVWWWVWWWVWWWVWWWVWW",
        "WBBBBBBB..KKKKK..CC.W",
        "WBBBBBBB..KKKKKS.CC.W",
        "W..SS......TT..CC...D",
        "W....HH....TT....CC.W",
        "W..SS............SS.W",
        "WWWWVWWWVWWWVWWWVWWWW",
    ]
    _RV_INT_W = 21
    _RV_INT_H = 7
    _RV_INT_TILE_SIZE = 20  # px per tile (interior scene)
    _RV_INT_SPRITE_SCALE = 2  # 2x player sprite inside RV
    _RV_INT_PLAYER_W = 10
    _RV_INT_PLAYER_H = 14
    _RV_INT_LEGEND: dict[str, int] = {
        "W": T_WALL,
        "V": T_WALL,  # window (drawn specially)
        ".": T_FLOOR,
        "D": T_DOOR,
        "S": T_SHELF,
        "K": T_TABLE,  # kitchen counter
        "H": T_TABLE,  # workbench
        "T": T_TABLE,
        "B": T_BED,
        "C": T_TABLE,  # cab seats (placeholder)
    }

    # High-rise apartment (multi-floor) interior.
    _HR_INT_LOBBY_LAYOUT: list[str] = [
        "WWWWWWWWWWWWWWWWWWWWWWW",
        "W.....................W",
        "W....CC.......CC......W",
        "W....CC.......CC......W",
        "W.....................W",
        "W.........EE..........D",
        "W.........EE..........W",
        "W.....................W",
        "W....CC.......CC......W",
        "W....CC.......CC......W",
        "WWWWWWWWWWWWWWWWWWWWWWW",
    ]
    _HR_INT_HALL_BASE: list[str] = [
        "WWWWWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWW.......WWWWWWWW",
        "WWWWWWWA.......AWWWWWWW",
        "WWWWWWWW.......WWWWWWWW",
        "WWWWWWWA.......AWWWWWWW",
        "WWWWWWWW.EE....WWWWWWWW",
        "WWWWWWWA.EE....AWWWWWWW",
        "WWWWWWWW.......WWWWWWWW",
        "WWWWWWWA.......AWWWWWWW",
        "WWWWWWWW.......WWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWWWWW",
    ]
    _HR_INT_HOME_LAYOUT: list[str] = [
        "WWWWWWWWWWWWWWWWWWWWWWW",
        "WVV.................VVW",
        "W..BBBBBB....SS.......W",
        "W..BBBBBB....SS...CCC.W",
        "W...........TTTT..CCC.W",
        "W..KKKKKF...TTTT......D",
        "W..KKKKKF.......SS....W",
        "W......CCCC.....SS....W",
        "W....SS.........TTTT..W",
        "W.....................W",
        "WWWWWWWWWWWWWWWWWWWWWWW",
    ]
    _HR_INT_W = 23
    _HR_INT_H = 11
    _HR_INT_TILE_SIZE = 20
    _HR_INT_SPRITE_SCALE = 2
    _HR_INT_MAX_FLOORS_DEFAULT = 12
    _HR_INT_APT_DOORS: list[tuple[int, int]] = [(7, 2), (15, 2), (7, 4), (15, 4), (7, 6), (15, 6), (7, 8), (15, 8)]

    # School (multi-floor) interior.
    _SCH_INT_LOBBY_LAYOUT: list[str] = [
        "WWWWWWWWWWWWWWWWWWWWWWW",
        "WVV.................VVW",
        "WLL...TTTT..TTTT....LLW",
        "WLL...TTTT..TTTT....LLW",
        "W.....................W",
        "W....^^.....EE.....vv.D",
        "W....^^.....EE.....vv.W",
        "W.....................W",
        "WLL...TTTT..TTTT....LLW",
        "WLL...TTTT..TTTT....LLW",
        "WWWWWWWWWWWWWWWWWWWWWWW",
    ]
    _SCH_INT_FLOOR_LAYOUT: list[str] = [
        "WWWWWWWWWWWWWWWWWWWWWWW",
        "WVV.................VVW",
        "WLL...TTTT..TTTT....LLW",
        "WLL...TTTT..TTTT....LLW",
        "W.....................W",
        "W....^^.....EE.....vv.W",
        "W....^^.....EE.....vv.W",
        "W.....................W",
        "WLL...TTTT..TTTT....LLW",
        "WLL...TTTT..TTTT....LLW",
        "WWWWWWWWWWWWWWWWWWWWWWW",
    ]
    _SCH_INT_W = 23
    _SCH_INT_H = 11
    _SCH_INT_TILE_SIZE = 20
    _SCH_INT_SPRITE_SCALE = 2
    _SCH_INT_MAX_FLOORS_DEFAULT = 4

    @dataclass(frozen=True)
    class _ItemDef:
        id: str
        name: str
        stack: int
        color: tuple[int, int, int]
        kind: str

    _ITEMS: dict[str, _ItemDef] = {
        "cola": _ItemDef("cola", "可乐", stack=3, color=(220, 70, 80), kind="drink"),
        "food_can": _ItemDef("food_can", "罐头", stack=6, color=(210, 170, 90), kind="food"),
        "water": _ItemDef("water", "瓶装水", stack=3, color=(120, 170, 230), kind="drink"),
        "bandage": _ItemDef("bandage", "绷带", stack=12, color=(230, 230, 240), kind="med"),
        "medkit": _ItemDef("medkit", "急救包", stack=2, color=(240, 120, 120), kind="med"),
        "pistol": _ItemDef("pistol", "9mm手枪", stack=1, color=(190, 190, 200), kind="gun"),
        "ammo_9mm": _ItemDef("ammo_9mm", "9mm弹", stack=60, color=(230, 220, 140), kind="ammo"),
        "scrap": _ItemDef("scrap", "废铁", stack=20, color=(150, 150, 160), kind="mat"),
        "wood": _ItemDef("wood", "木材", stack=20, color=(150, 110, 70), kind="mat"),
    }

    _ICON_PAL = {
        "K": (18, 18, 22),
        "G": (190, 190, 200),
        "W": (240, 240, 240),
        "R": (240, 120, 120),
        "C": (220, 70, 80),
        "c": (170, 50, 60),
        "B": (120, 170, 230),
        "Y": (230, 220, 140),
        "O": (210, 170, 90),
        "S": (150, 150, 160),
        "T": (150, 110, 70),
    }

    _ICON_PISTOL = sprite_from_pixels(
        [
            "        ",
            "   GGG  ",
            "  GKKG  ",
            "  GGGG  ",
            "    KG  ",
            "   GKG  ",
            "    G   ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_AMMO = sprite_from_pixels(
        [
            "        ",
            "  YYYY  ",
            "  YYYY  ",
            "  YYYY  ",
            "   YY   ",
            "   YY   ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_FOOD = sprite_from_pixels(
        [
            "        ",
            "  OOOO  ",
            " OOOOOO ",
            " OOKKOO ",
            " OOOOOO ",
            "  OOOO  ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_WATER = sprite_from_pixels(
        [
            "        ",
            "   BB   ",
            "   BB   ",
            "  BBBB  ",
            "  BBBB  ",
            "  BBBB  ",
            "   BB   ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_COLA = sprite_from_pixels(
        [
            "        ",
            "  CCCC  ",
            " CCCCC  ",
            " CWWWC  ",
            " CCCCC  ",
            "  CCCC  ",
            "   CC   ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_BANDAGE = sprite_from_pixels(
        [
            "        ",
            "  WWWW  ",
            "  WKKW  ",
            "  WWWW  ",
            "  WKKW  ",
            "  WWWW  ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_MEDKIT = sprite_from_pixels(
        [
            "        ",
            "  RRRR  ",
            "  RWWR  ",
            "  RWWR  ",
            "  RRRR  ",
            "   RR   ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_SCRAP = sprite_from_pixels(
        [
            "        ",
            "  SSSS  ",
            " SSKKSS ",
            " SSSSSS ",
            "  SSSS  ",
            "   SS   ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )
    _ICON_WOOD = sprite_from_pixels(
        [
            "        ",
            "  TTTT  ",
            " TTKKTT ",
            " TTTTTT ",
            "  TTTT  ",
            "   TT   ",
            "        ",
            "        ",
        ],
        _ICON_PAL,
    )

    _ITEM_ICONS: dict[str, pygame.Surface] = {
        "pistol": _ICON_PISTOL,
        "ammo_9mm": _ICON_AMMO,
        "food_can": _ICON_FOOD,
        "water": _ICON_WATER,
        "cola": _ICON_COLA,
        "bandage": _ICON_BANDAGE,
        "medkit": _ICON_MEDKIT,
        "scrap": _ICON_SCRAP,
        "wood": _ICON_WOOD,
    }

    def _make_item_sprite_pistol() -> pygame.Surface:
        surf = pygame.Surface((20, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        dark = (62, 62, 72)
        mid = (92, 92, 104)
        hi = (150, 150, 162)

        # Slide + barrel.
        pygame.draw.rect(surf, dark, pygame.Rect(2, 3, 14, 3))
        pygame.draw.rect(surf, hi, pygame.Rect(3, 4, 12, 1))
        pygame.draw.rect(surf, outline, pygame.Rect(2, 3, 14, 3), 1)
        pygame.draw.rect(surf, dark, pygame.Rect(16, 4, 2, 2))
        pygame.draw.rect(surf, outline, pygame.Rect(16, 4, 2, 2), 1)
        surf.set_at((18, 5), outline)  # muzzle
        pygame.draw.rect(surf, mid, pygame.Rect(9, 3, 3, 1))  # ejection port hint

        # Grip + trigger guard.
        pygame.draw.rect(surf, mid, pygame.Rect(8, 6, 5, 5))
        pygame.draw.rect(surf, outline, pygame.Rect(8, 6, 5, 5), 1)
        pygame.draw.rect(surf, dark, pygame.Rect(9, 6, 3, 2))
        pygame.draw.rect(surf, outline, pygame.Rect(9, 6, 3, 2), 1)
        pygame.draw.line(surf, outline, (7, 7), (8, 7), 1)
        # Trigger + grip texture.
        surf.set_at((9, 8), outline)
        for y in range(7, 11):
            if y % 2 == 0:
                surf.set_at((11, y), dark)

        return surf

    def _make_item_sprite_ammo() -> pygame.Surface:
        surf = pygame.Surface((16, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        box = (120, 120, 132)
        box2 = (92, 92, 104)
        brass = (230, 220, 140)
        brass2 = (200, 180, 90)

        pygame.draw.rect(surf, box, pygame.Rect(2, 4, 12, 6), border_radius=2)
        pygame.draw.rect(surf, outline, pygame.Rect(2, 4, 12, 6), 1, border_radius=2)
        pygame.draw.rect(surf, box2, pygame.Rect(2, 8, 12, 2), border_radius=1)
        for i in range(3):
            x = 4 + i * 3
            pygame.draw.rect(surf, brass, pygame.Rect(x, 2, 2, 4))
            pygame.draw.rect(surf, brass2, pygame.Rect(x, 4, 2, 1))
            pygame.draw.rect(surf, outline, pygame.Rect(x, 2, 2, 4), 1)
        return surf

    def _make_item_sprite_food_can() -> pygame.Surface:
        surf = pygame.Surface((14, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        can = (210, 170, 90)
        can2 = (170, 130, 70)
        label = (230, 220, 140)
        label2 = (200, 180, 90)

        pygame.draw.rect(surf, can, pygame.Rect(3, 2, 8, 9), border_radius=2)
        pygame.draw.rect(surf, outline, pygame.Rect(3, 2, 8, 9), 1, border_radius=2)
        pygame.draw.line(surf, label, (4, 5), (9, 5), 2)
        pygame.draw.line(surf, label2, (4, 7), (9, 7), 1)
        pygame.draw.line(surf, can2, (4, 3), (9, 3), 1)
        return surf

    def _make_item_sprite_water() -> pygame.Surface:
        surf = pygame.Surface((12, 14), pygame.SRCALPHA)
        outline = (10, 10, 12)
        water = (120, 170, 230)
        water2 = (90, 130, 190)
        cap = (220, 220, 230)

        # Cap.
        pygame.draw.rect(surf, cap, pygame.Rect(5, 1, 2, 2), border_radius=1)
        pygame.draw.rect(surf, outline, pygame.Rect(5, 1, 2, 2), 1, border_radius=1)

        # Neck.
        pygame.draw.rect(surf, water, pygame.Rect(5, 3, 2, 2), border_radius=1)
        pygame.draw.rect(surf, outline, pygame.Rect(5, 3, 2, 2), 1, border_radius=1)

        # Body.
        body = pygame.Rect(3, 5, 6, 8)
        pygame.draw.rect(surf, water, body, border_radius=3)
        pygame.draw.rect(surf, outline, body, 1, border_radius=3)

        # Label band.
        label = pygame.Rect(3, 8, 6, 2)
        pygame.draw.rect(surf, cap, label, border_radius=1)
        pygame.draw.rect(surf, outline, label, 1, border_radius=1)
        surf.set_at((5, 9), (255, 255, 255))

        # Highlights.
        pygame.draw.line(surf, water2, (4, 6), (4, 12), 1)
        pygame.draw.line(surf, cap, (8, 6), (8, 12), 1)
        return surf

    def _make_item_sprite_cola() -> pygame.Surface:
        surf = pygame.Surface((12, 14), pygame.SRCALPHA)
        outline = (10, 10, 12)
        red = (220, 70, 80)
        red2 = (170, 50, 60)
        white = (240, 240, 240)
        steel = (220, 220, 230)

        # Top.
        pygame.draw.rect(surf, steel, pygame.Rect(4, 1, 4, 2), border_radius=1)
        pygame.draw.rect(surf, outline, pygame.Rect(4, 1, 4, 2), 1, border_radius=1)

        # Can body.
        body = pygame.Rect(3, 3, 6, 10)
        pygame.draw.rect(surf, red, body, border_radius=2)
        pygame.draw.rect(surf, outline, body, 1, border_radius=2)
        pygame.draw.rect(surf, red2, pygame.Rect(3, 4, 2, 8), border_radius=1)
        pygame.draw.line(surf, steel, (8, 4), (8, 11), 1)  # right highlight

        # Logo wave (simple, readable at 2x in world).
        pygame.draw.line(surf, white, (4, 7), (7, 6), 2)
        pygame.draw.line(surf, white, (4, 9), (7, 8), 1)
        pygame.draw.line(surf, white, (4, 11), (7, 10), 1)
        pygame.draw.circle(surf, white, (6, 8), 1)  # badge

        # Bottom rim.
        pygame.draw.line(surf, steel, (4, 12), (7, 12), 1)
        surf.set_at((6, 2), outline)  # pull-tab hint
        return surf

    def _make_item_sprite_bandage() -> pygame.Surface:
        surf = pygame.Surface((14, 10), pygame.SRCALPHA)
        outline = (10, 10, 12)
        cloth = (240, 240, 240)
        cloth2 = (200, 200, 210)

        # Rolled bandage + loose tail.
        roll = pygame.Rect(2, 2, 7, 6)
        pygame.draw.ellipse(surf, cloth, roll)
        pygame.draw.ellipse(surf, outline, roll, 1)
        pygame.draw.ellipse(surf, cloth2, roll.inflate(-3, -3))
        pygame.draw.line(surf, outline, (roll.centerx, roll.top + 1), (roll.centerx, roll.bottom - 2), 1)

        tail = pygame.Rect(8, 4, 4, 3)
        pygame.draw.rect(surf, cloth, tail, border_radius=1)
        pygame.draw.rect(surf, outline, tail, 1, border_radius=1)
        pygame.draw.line(surf, cloth2, (tail.left + 1, tail.top + 1), (tail.right - 2, tail.top + 1), 1)
        return surf

    def _make_item_sprite_medkit() -> pygame.Surface:
        surf = pygame.Surface((16, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        red = (240, 120, 120)
        red2 = (190, 90, 90)
        white = (240, 240, 240)

        pygame.draw.rect(surf, red, pygame.Rect(3, 3, 10, 7), border_radius=2)
        pygame.draw.rect(surf, outline, pygame.Rect(3, 3, 10, 7), 1, border_radius=2)
        pygame.draw.rect(surf, red2, pygame.Rect(3, 8, 10, 2), border_radius=1)
        pygame.draw.rect(surf, outline, pygame.Rect(6, 1, 4, 2), 1)  # handle
        pygame.draw.rect(surf, white, pygame.Rect(7, 5, 2, 4))
        pygame.draw.rect(surf, white, pygame.Rect(6, 6, 4, 2))
        return surf

    def _make_item_sprite_scrap() -> pygame.Surface:
        surf = pygame.Surface((14, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        metal = (150, 150, 160)
        metal2 = (110, 110, 122)

        pygame.draw.polygon(surf, metal, [(2, 7), (5, 2), (12, 3), (10, 10), (4, 11)])
        pygame.draw.polygon(surf, outline, [(2, 7), (5, 2), (12, 3), (10, 10), (4, 11)], 1)
        pygame.draw.polygon(surf, metal2, [(6, 5), (11, 4), (9, 9), (5, 9)])
        pygame.draw.line(surf, metal2, (3, 8), (9, 6), 1)

        # Bolt plate.
        plate = pygame.Rect(1, 2, 5, 3)
        pygame.draw.rect(surf, metal2, plate, border_radius=1)
        pygame.draw.rect(surf, outline, plate, 1, border_radius=1)
        pygame.draw.circle(surf, metal, (plate.left + 2, plate.centery), 1)
        pygame.draw.circle(surf, outline, (plate.left + 2, plate.centery), 1, 1)
        return surf

    def _make_item_sprite_wood() -> pygame.Surface:
        surf = pygame.Surface((16, 10), pygame.SRCALPHA)
        outline = (10, 10, 12)
        wood = (150, 110, 70)
        wood2 = (120, 86, 56)
        wood3 = (176, 132, 86)

        # Back plank.
        back = pygame.Rect(4, 1, 10, 4)
        pygame.draw.rect(surf, wood2, back, border_radius=2)
        pygame.draw.rect(surf, outline, back, 1, border_radius=2)
        pygame.draw.line(surf, wood3, (back.left + 2, back.top + 1), (back.right - 3, back.top + 1), 1)

        # Front plank.
        front = pygame.Rect(2, 4, 12, 4)
        pygame.draw.rect(surf, wood, front, border_radius=2)
        pygame.draw.rect(surf, outline, front, 1, border_radius=2)
        for x in (4, 7, 10):
            pygame.draw.line(surf, wood2, (x, 5), (x, 7), 1)
        pygame.draw.circle(surf, wood2, (9, 6), 1)  # knot
        return surf

    _ITEM_SPRITES: dict[str, pygame.Surface] = {
        "pistol": _make_item_sprite_pistol(),
        "ammo_9mm": _make_item_sprite_ammo(),
        "food_can": _make_item_sprite_food_can(),
        "water": _make_item_sprite_water(),
        "cola": _make_item_sprite_cola(),
        "bandage": _make_item_sprite_bandage(),
        "medkit": _make_item_sprite_medkit(),
        "scrap": _make_item_sprite_scrap(),
        "wood": _make_item_sprite_wood(),
    }

    _ITEM_SPRITES_WORLD: dict[str, pygame.Surface] = {}
    for _iid, _spr in _ITEM_SPRITES.items():
        try:
            # World loot should be smaller than the player sprite (12x16).
            # We keep the detailed sprite, but scale it down to a small icon-like
            # footprint so ground drops don't look oversized.
            max_px = 11
            sw, sh = int(_spr.get_width()), int(_spr.get_height())
            biggest = max(1, max(int(sw), int(sh)))
            scale = min(1.0, float(max_px) / float(biggest))
            if scale < 1.0:
                nw = max(1, int(round(float(sw) * scale)))
                nh = max(1, int(round(float(sh) * scale)))
                _ITEM_SPRITES_WORLD[_iid] = pygame.transform.scale(_spr, (int(nw), int(nh)))
            else:
                _ITEM_SPRITES_WORLD[_iid] = _spr
        except Exception:
            _ITEM_SPRITES_WORLD[_iid] = _spr

    for _iid, _spr in _ITEM_SPRITES.items():
        try:
            if _iid not in _ITEM_ICONS:
                _ITEM_ICONS[_iid] = pygame.transform.scale(_spr, (8, 8))
        except Exception:
            pass

    # Keep pistol visuals consistent across world + inventory + in-hand.
    try:
        _ps = _ITEM_SPRITES_WORLD.get("pistol")
        if _ps is not None:
            bw, bh = int(_ps.get_width()), int(_ps.get_height())
            scale = min(8.0 / max(1.0, float(bw)), 8.0 / max(1.0, float(bh)))
            tw = int(max(1, round(float(bw) * scale)))
            th = int(max(1, round(float(bh) * scale)))
            thumb = pygame.transform.scale(_ps, (tw, th))
            icon = pygame.Surface((8, 8), pygame.SRCALPHA)
            icon.blit(thumb, (int((8 - tw) // 2), int((8 - th) // 2)))
            _ITEM_ICONS["pistol"] = icon
    except Exception:
        pass

    def _make_hand_pistol_sprite(_src: pygame.Surface = _ITEM_SPRITES_WORLD["pistol"]) -> pygame.Surface:
        # Keep the held pistol consistent with the pickup sprite.
        return _src

    _GUN_HAND_SPRITES: dict[str, pygame.Surface] = {
        "pistol": _make_hand_pistol_sprite(),
    }

    @dataclass
    class _ItemStack:
        item_id: str
        qty: int

    @dataclass
    class _WorldItem:
        pos: pygame.Vector2
        item_id: str
        qty: int

    @dataclass
    class _MapMarker:
        tx: int
        ty: int
        label: str
        color: tuple[int, int, int] = (255, 220, 140)

    @dataclass
    class _ParkedCar:
        pos: pygame.Vector2
        model_id: str
        heading: float = 0.0
        steer_state: int = 0
        frame: int = 0

    @dataclass
    class _ParkedBike:
        pos: pygame.Vector2
        model_id: str
        dir: str = "right"
        frame: int = 0
        fuel: float = 0.0

    @dataclass
    class _WorldProp:
        pos: pygame.Vector2
        prop_id: str
        variant: int = 0
        dir: str = "down"

    @dataclass(frozen=True)
    class _PropDef:
        id: str
        name: str
        category: str
        solid: bool = False
        collider: tuple[int, int] = (10, 10)  # px, for collisions (centered on pos)
        sprites: tuple[pygame.Surface, ...] = ()

    def _make_prop_sprite_sign(*, bg: tuple[int, int, int], accent: tuple[int, int, int], icon: str) -> pygame.Surface:
        surf = pygame.Surface((14, 10), pygame.SRCALPHA)
        outline = (10, 10, 12)
        border = pygame.Rect(1, 1, 12, 8)
        pygame.draw.rect(surf, bg, border, border_radius=2)
        pygame.draw.rect(surf, outline, border, 1, border_radius=2)
        pygame.draw.rect(surf, accent, pygame.Rect(1, 7, 12, 2), border_radius=1)
        # Small icon in the middle (readable at 1x).
        cx, cy = 7, 4
        if icon == "cross":
            red = (240, 80, 90)
            pygame.draw.rect(surf, red, pygame.Rect(cx - 1, cy - 2, 2, 5))
            pygame.draw.rect(surf, red, pygame.Rect(cx - 2, cy - 1, 5, 2))
        elif icon == "cart":
            white = (240, 240, 240)
            pygame.draw.rect(surf, white, pygame.Rect(cx - 3, cy - 1, 6, 3), 1)
            pygame.draw.line(surf, white, (cx - 4, cy - 2), (cx - 1, cy - 2), 1)
            pygame.draw.line(surf, white, (cx - 1, cy - 2), (cx + 1, cy + 1), 1)
            surf.set_at((cx - 2, cy + 2), white)
            surf.set_at((cx + 2, cy + 2), white)
        elif icon == "school":
            blue = (130, 190, 240)
            pygame.draw.rect(surf, blue, pygame.Rect(cx - 3, cy - 2, 6, 5), 1)
            pygame.draw.line(surf, blue, (cx - 3, cy - 1), (cx + 2, cy - 2), 1)
            pygame.draw.line(surf, blue, (cx - 3, cy + 1), (cx + 2, cy), 1)
        elif icon == "bars":
            steel = (220, 220, 230)
            for dx in (-2, 0, 2):
                pygame.draw.line(surf, steel, (cx + dx, cy - 2), (cx + dx, cy + 2), 1)
            pygame.draw.line(surf, steel, (cx - 3, cy - 2), (cx + 3, cy - 2), 1)
            pygame.draw.line(surf, steel, (cx - 3, cy + 2), (cx + 3, cy + 2), 1)
        elif icon == "tower":
            steel = (220, 220, 230)
            pygame.draw.rect(surf, steel, pygame.Rect(cx - 2, cy - 3, 5, 7), 1)
            pygame.draw.rect(surf, steel, pygame.Rect(cx - 1, cy - 2, 1, 1))
            pygame.draw.rect(surf, steel, pygame.Rect(cx + 1, cy - 1, 1, 1))
            pygame.draw.rect(surf, steel, pygame.Rect(cx - 1, cy + 1, 1, 1))
        elif icon == "book":
            ink = (240, 240, 240)
            pygame.draw.rect(surf, ink, pygame.Rect(cx - 4, cy - 2, 4, 5), 1)
            pygame.draw.rect(surf, ink, pygame.Rect(cx, cy - 2, 4, 5), 1)
            pygame.draw.line(surf, ink, (cx, cy - 2), (cx, cy + 2), 1)
            surf.set_at((cx - 3, cy - 1), ink)
            surf.set_at((cx + 2, cy), ink)
        elif icon == "pagoda":
            gold = (240, 210, 130)
            red = (220, 90, 80)
            pygame.draw.line(surf, gold, (cx - 4, cy - 2), (cx + 4, cy - 2), 1)
            pygame.draw.line(surf, gold, (cx - 3, cy - 1), (cx + 3, cy - 1), 1)
            pygame.draw.line(surf, red, (cx - 4, cy), (cx + 4, cy), 1)
            pygame.draw.rect(surf, red, pygame.Rect(cx - 2, cy, 5, 3), 1)
            surf.set_at((cx, cy - 3), gold)
        else:
            steel = (220, 220, 230)
            pygame.draw.rect(surf, steel, pygame.Rect(cx - 2, cy - 2, 4, 4), 1)
        return surf

    def _make_prop_sprite_billboard(*, variant: int = 0) -> pygame.Surface:
        surf = pygame.Surface((18, 10), pygame.SRCALPHA)
        outline = (10, 10, 12)
        paper = (200, 200, 210)
        grime = (130, 130, 140)
        frame = (70, 70, 86)
        board = pygame.Rect(1, 1, 16, 7)
        pygame.draw.rect(surf, frame, board, border_radius=2)
        pygame.draw.rect(surf, outline, board, 1, border_radius=2)
        pygame.draw.rect(surf, paper, pygame.Rect(2, 2, 14, 5), border_radius=1)
        # Torn / dirty lines.
        for i in range(3):
            y = 2 + i * 2
            pygame.draw.line(surf, grime, (3, y), (15, y), 1)
        if int(variant) % 2 == 1:
            pygame.draw.rect(surf, (240, 200, 120), pygame.Rect(2, 2, 4, 5))
        # Poles.
        pygame.draw.rect(surf, outline, pygame.Rect(5, 8, 2, 2))
        pygame.draw.rect(surf, outline, pygame.Rect(11, 8, 2, 2))
        return surf

    def _make_prop_sprite_seesaw(*, variant: int = 0) -> pygame.Surface:
        surf = pygame.Surface((16, 10), pygame.SRCALPHA)
        outline = (10, 10, 12)
        wood = (170, 130, 70)
        rust = (150, 90, 60)
        plank = pygame.Rect(1, 4, 14, 2)
        pygame.draw.rect(surf, wood, plank, border_radius=1)
        pygame.draw.rect(surf, outline, plank, 1, border_radius=1)
        pivot = pygame.Rect(7, 5, 2, 3)
        pygame.draw.rect(surf, rust, pivot, border_radius=1)
        pygame.draw.rect(surf, outline, pivot, 1, border_radius=1)
        # One broken seat.
        if int(variant) % 2 == 0:
            surf.set_at((2, 3), outline)
            surf.set_at((13, 6), outline)
        return surf

    def _make_prop_sprite_carousel(*, variant: int = 0) -> pygame.Surface:
        surf = pygame.Surface((16, 16), pygame.SRCALPHA)
        outline = (10, 10, 12)
        rust = (150, 90, 60)
        metal = (140, 140, 150)
        ring = pygame.Rect(2, 2, 12, 12)
        pygame.draw.ellipse(surf, metal, ring)
        pygame.draw.ellipse(surf, outline, ring, 1)
        pygame.draw.ellipse(surf, (110, 110, 122), ring.inflate(-6, -6))
        # Spokes + seats (some missing).
        cx, cy = 8, 8
        for i, (dx, dy) in enumerate(((0, -5), (4, -3), (5, 0), (4, 3), (0, 5), (-4, 3), (-5, 0), (-4, -3))):
            pygame.draw.line(surf, rust, (cx, cy), (cx + dx, cy + dy), 1)
            if int(variant) % 3 == 0 and i in (2, 6):
                continue
            surf.set_at((cx + dx, cy + dy), outline)
        return surf

    def _make_prop_sprite_swing(*, variant: int = 0) -> pygame.Surface:
        surf = pygame.Surface((16, 12), pygame.SRCALPHA)
        outline = (10, 10, 12)
        metal = (160, 160, 170)
        rust = (150, 90, 60)
        # Frame.
        pygame.draw.line(surf, metal, (3, 10), (6, 2), 1)
        pygame.draw.line(surf, metal, (13, 10), (10, 2), 1)
        pygame.draw.line(surf, metal, (6, 2), (10, 2), 1)
        pygame.draw.line(surf, outline, (3, 10), (6, 2), 1)
        pygame.draw.line(surf, outline, (13, 10), (10, 2), 1)
        pygame.draw.line(surf, outline, (6, 2), (10, 2), 1)
        # Chains + seat (one chain broken sometimes).
        pygame.draw.line(surf, rust, (7, 3), (7, 8), 1)
        if int(variant) % 2 == 0:
            pygame.draw.line(surf, rust, (9, 3), (9, 8), 1)
        pygame.draw.rect(surf, (90, 90, 110), pygame.Rect(6, 8, 4, 2), border_radius=1)
        pygame.draw.rect(surf, outline, pygame.Rect(6, 8, 4, 2), 1, border_radius=1)
        return surf

    _PROP_DEFS: dict[str, _PropDef] = {
        # Building signs / markers.
        "sign_hospital": _PropDef(
            id="sign_hospital",
            name="医院招牌",
            category="招牌",
            solid=False,
            collider=(10, 8),
            sprites=(_make_prop_sprite_sign(bg=(220, 220, 230), accent=(200, 80, 90), icon="cross"),),
        ),
        "sign_shop": _PropDef(
            id="sign_shop",
            name="超市招牌",
            category="招牌",
            solid=False,
            collider=(10, 8),
            sprites=(_make_prop_sprite_sign(bg=(60, 60, 72), accent=(230, 200, 120), icon="cart"),),
        ),
        "sign_shop_big": _PropDef(
            id="sign_shop_big",
            name="大型超市招牌",
            category="招牌",
            solid=False,
            collider=(12, 8),
            sprites=(_make_prop_sprite_sign(bg=(70, 70, 86), accent=(240, 200, 120), icon="cart"),),
        ),
        "sign_school": _PropDef(
            id="sign_school",
            name="学校标识牌",
            category="招牌",
            solid=False,
            collider=(10, 8),
            sprites=(_make_prop_sprite_sign(bg=(36, 44, 60), accent=(120, 180, 240), icon="school"),),
        ),
        "sign_prison": _PropDef(
            id="sign_prison",
            name="监狱警示牌",
            category="招牌",
            solid=False,
            collider=(12, 8),
            sprites=(_make_prop_sprite_sign(bg=(40, 40, 48), accent=(160, 160, 170), icon="bars"),),
        ),
        "sign_highrise": _PropDef(
            id="sign_highrise",
            name="高层住宅门牌",
            category="门牌",
            solid=False,
            collider=(10, 8),
            sprites=(_make_prop_sprite_sign(bg=(50, 50, 60), accent=(120, 120, 132), icon="tower"),),
        ),
        "sign_bookstore": _PropDef(
            id="sign_bookstore",
            name="新华书店招牌",
            category="招牌",
            solid=False,
            collider=(12, 8),
            sprites=(_make_prop_sprite_sign(bg=(38, 52, 88), accent=(240, 210, 130), icon="book"),),
        ),
        "sign_chinese": _PropDef(
            id="sign_chinese",
            name="中式建筑牌匾",
            category="招牌",
            solid=False,
            collider=(12, 8),
            sprites=(_make_prop_sprite_sign(bg=(70, 36, 36), accent=(240, 210, 130), icon="pagoda"),),
        ),
        "sign_home": _PropDef(
            id="sign_home",
            name="住宅门牌",
            category="门牌",
            solid=False,
            collider=(10, 8),
            sprites=(_make_prop_sprite_sign(bg=(58, 52, 46), accent=(170, 130, 70), icon="home"),),
        ),
        # City dressing.
        "billboard": _PropDef(
            id="billboard",
            name="破旧广告牌",
            category="广告",
            solid=False,
            collider=(16, 8),
            sprites=(
                _make_prop_sprite_billboard(variant=0),
                _make_prop_sprite_billboard(variant=1),
                _make_prop_sprite_billboard(variant=2),
            ),
        ),
        # Park toys (abandoned).
        "toy_seesaw": _PropDef(
            id="toy_seesaw",
            name="废弃跷跷板",
            category="公园设施",
            solid=True,
            collider=(16, 10),
            sprites=(_make_prop_sprite_seesaw(variant=0), _make_prop_sprite_seesaw(variant=1)),
        ),
        "toy_carousel": _PropDef(
            id="toy_carousel",
            name="废弃旋转木马",
            category="公园设施",
            solid=True,
            collider=(16, 16),
            sprites=(_make_prop_sprite_carousel(variant=0), _make_prop_sprite_carousel(variant=1), _make_prop_sprite_carousel(variant=2)),
        ),
        "toy_swing": _PropDef(
            id="toy_swing",
            name="废弃秋千",
            category="公园设施",
            solid=True,
            collider=(16, 12),
            sprites=(_make_prop_sprite_swing(variant=0), _make_prop_sprite_swing(variant=1)),
        ),
    }

    @dataclass
    class _Inventory:
        slots: list["HardcoreSurvivalState._ItemStack | None"]
        cols: int = 4

        def add(self, item_id: str, qty: int, item_defs: dict[str, "HardcoreSurvivalState._ItemDef"]) -> int:
            if qty <= 0:
                return 0
            idef = item_defs.get(item_id)
            if idef is None:
                return qty
            stack_max = max(1, int(idef.stack))

            left = int(qty)
            for slot in self.slots:
                if slot is None or slot.item_id != item_id:
                    continue
                if slot.qty >= stack_max:
                    continue
                take = min(stack_max - slot.qty, left)
                slot.qty += int(take)
                left -= int(take)
                if left <= 0:
                    return 0

            for i, slot in enumerate(self.slots):
                if slot is not None:
                    continue
                take = min(stack_max, left)
                self.slots[i] = HardcoreSurvivalState._ItemStack(item_id=item_id, qty=int(take))
                left -= int(take)
                if left <= 0:
                    return 0
            return left

        def drop_slot(self, index: int) -> "HardcoreSurvivalState._ItemStack | None":
            if not (0 <= index < len(self.slots)):
                return None
            stack = self.slots[index]
            self.slots[index] = None
            return stack

        def count(self, item_id: str) -> int:
            total = 0
            for slot in self.slots:
                if slot is not None and slot.item_id == item_id:
                    total += int(slot.qty)
            return int(total)

        def remove(self, item_id: str, qty: int) -> int:
            need = int(qty)
            if need <= 0:
                return 0
            removed = 0
            for i, slot in enumerate(self.slots):
                if need <= 0:
                    break
                if slot is None or slot.item_id != item_id:
                    continue
                take = min(int(slot.qty), need)
                slot.qty -= int(take)
                need -= int(take)
                removed += int(take)
                if slot.qty <= 0:
                    self.slots[i] = None
            return int(removed)

    @dataclass(frozen=True)
    class _GunDef:
        id: str
        name: str
        ammo_item: str
        mag_size: int
        fire_rate: float
        reload_s: float
        damage: int
        bullet_speed: float
        spread_deg: float

    _GUNS: dict[str, _GunDef] = {
        "pistol": _GunDef(
            id="pistol",
            name="9mm手枪",
            ammo_item="ammo_9mm",
            mag_size=12,
            fire_rate=4.0,
            reload_s=1.15,
            damage=12,
            bullet_speed=260.0,
            spread_deg=4.0,
        )
    }

    @dataclass
    class _Gun:
        gun_id: str
        mag: int
        cooldown_left: float = 0.0
        reload_left: float = 0.0
        reload_total: float = 0.0

    @dataclass
    class _Bullet:
        pos: pygame.Vector2
        vel: pygame.Vector2
        ttl: float
        dmg: int

    @dataclass(frozen=True)
    class _MonsterDef:
        id: str
        name: str
        hp: int
        speed: float
        dmg: int
        sense: float
        attack_range: float
        attack_cd: float
        roam_speed: float
        scream_cd: float = 0.0
        scream_radius: float = 0.0
        scream_spawn: int = 0

    _MONSTER_DEFS: dict[str, _MonsterDef] = {
        "walker": _MonsterDef(
            id="walker",
            name="行尸",
            hp=26,
            speed=16.0,
            dmg=8,
            sense=120.0,
            attack_range=10.0,
            attack_cd=0.65,
            roam_speed=7.0,
        ),
        "runner": _MonsterDef(
            id="runner",
            name="奔跑者",
            hp=16,
            speed=24.0,
            dmg=12,
            sense=140.0,
            attack_range=12.0,
            attack_cd=0.90,
            roam_speed=11.0,
        ),
        "screamer": _MonsterDef(
            id="screamer",
            name="尖叫者",
            hp=18,
            speed=20.0,
            dmg=7,
            sense=150.0,
            attack_range=10.0,
            attack_cd=0.80,
            roam_speed=9.0,
            scream_cd=10.0,
            scream_radius=110.0,
            scream_spawn=4,
        ),
    }

    @dataclass
    class _Zombie:
        pos: pygame.Vector2
        vel: pygame.Vector2
        hp: int
        speed: float
        kind: str = "walker"
        w: int = 10
        h: int = 10
        dir: str = "down"
        anim: float = 0.0
        attack_left: float = 0.0
        scream_left: float = 0.0
        wander_dir: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0, 0))
        wander_left: float = 0.0
        stagger_left: float = 0.0
        stagger_vel: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0, 0))

        def rect(self) -> pygame.Rect:
            return pygame.Rect(
                iround(float(self.pos.x) - float(self.w) / 2.0),
                iround(float(self.pos.y) - float(self.h) / 2.0),
                int(self.w),
                int(self.h),
            )

    @dataclass
    class _RV:
        pos: pygame.Vector2
        vel: pygame.Vector2
        model_id: str = "rv"
        fuel: float = 100.0
        heading: float = 0.0  # radians; 0 points right
        speed: float = 0.0  # px/s (signed)
        steer: float = 0.0  # radians (signed)
        w: int = 26
        h: int = 16

        def rect(self) -> pygame.Rect:
            return pygame.Rect(
                iround(float(self.pos.x) - float(self.w) / 2.0),
                iround(float(self.pos.y) - float(self.h) / 2.0),
                int(self.w),
                int(self.h),
            )

    @dataclass
    class _Bike:
        pos: pygame.Vector2
        vel: pygame.Vector2
        model_id: str = "bike"
        fuel: float = 0.0
        w: int = 14
        h: int = 10

        def rect(self) -> pygame.Rect:
            return pygame.Rect(
                iround(float(self.pos.x) - float(self.w) / 2.0),
                iround(float(self.pos.y) - float(self.h) / 2.0),
                int(self.w),
                int(self.h),
            )

    def _hash_u32(self, n: int) -> int:
        n &= 0xFFFFFFFF
        n ^= (n >> 16) & 0xFFFFFFFF
        n = (n * 0x7FEB352D) & 0xFFFFFFFF
        n ^= (n >> 15) & 0xFFFFFFFF
        n = (n * 0x846CA68B) & 0xFFFFFFFF
        n ^= (n >> 16) & 0xFFFFFFFF
        return n & 0xFFFFFFFF

    def _hash2_u32(self, x: int, y: int, seed: int) -> int:
        n = (x * 0x1F123BB5 + y * 0x05491333 + seed * 0x9E3779B9) & 0xFFFFFFFF
        return self._hash_u32(n)

    def _rand01(self, x: int, y: int, seed: int) -> float:
        return self._hash2_u32(x, y, seed) / 4294967296.0

    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _noise2(self, x: float, y: float, seed: int) -> float:
        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi
        v00 = self._rand01(int(xi), int(yi), seed)
        v10 = self._rand01(int(xi) + 1, int(yi), seed)
        v01 = self._rand01(int(xi), int(yi) + 1, seed)
        v11 = self._rand01(int(xi) + 1, int(yi) + 1, seed)
        u = self._fade(float(xf))
        v = self._fade(float(yf))
        return self._lerp(self._lerp(v00, v10, u), self._lerp(v01, v11, u), v)

    def _fractal(self, x: float, y: float, seed: int) -> float:
        value = 0.0
        amp = 1.0
        total = 0.0
        freq = 1.0 / 64.0
        for octave in range(4):
            value += self._noise2(x * freq, y * freq, seed + octave * 101) * amp
            total += amp
            amp *= 0.5
            freq *= 2.0
        if total <= 0.0:
            return 0.0
        return value / total

    @dataclass(frozen=True)
    class _SpecialBuilding:
        kind: str  # e.g. "highrise"
        name: str
        tx0: int
        ty0: int
        w: int
        h: int
        door_tiles: tuple[tuple[int, int], ...]
        floors: int = 0

    @dataclass
    class _Chunk:
        cx: int
        cy: int
        tiles: list[int]
        items: list["HardcoreSurvivalState._WorldItem"] = field(default_factory=list)
        buildings: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)  # (tx0, ty0, w, h, roof_kind, floors) in world tiles
        special_buildings: list["HardcoreSurvivalState._SpecialBuilding"] = field(default_factory=list)
        cars: list["HardcoreSurvivalState._ParkedCar"] = field(default_factory=list)
        bikes: list["HardcoreSurvivalState._ParkedBike"] = field(default_factory=list)
        props: list["HardcoreSurvivalState._WorldProp"] = field(default_factory=list)
        town_kind: str | None = None

    class _World:
        def __init__(self, state: "HardcoreSurvivalState", seed: int) -> None:
            self.state = state
            self.seed = int(seed) & 0xFFFFFFFF
            self.chunks: dict[tuple[int, int], HardcoreSurvivalState._Chunk] = {}
            # Chunk streaming: we avoid generating new chunks during render (draw)
            # to prevent big hitches when the camera crosses chunk corners.
            self._gen_queue: list[tuple[int, int]] = []
            self._gen_queue_i = 0
            self._gen_queue_set: set[tuple[int, int]] = set()
            self.road_off_x = state._hash2_u32(7, 11, self.seed) % state.ROAD_PERIOD
            self.road_off_y = state._hash2_u32(13, 17, self.seed) % state.ROAD_PERIOD

            # Spawn-city: a dense urban region centered on the spawn road intersection.
            stx, sty = self.spawn_tile()
            self.city_cx = int(stx) // state.CHUNK_SIZE
            self.city_cy = int(sty) // state.CHUNK_SIZE
            self.city_radius = 5  # chunks
            self.city_period = 24  # minor street grid (tiles)
            self.city_w = 1  # half-width of minor streets (tiles)
            self.city_off_x = state._hash2_u32(29, 37, self.seed) % max(1, int(self.city_period))
            self.city_off_y = state._hash2_u32(41, 53, self.seed) % max(1, int(self.city_period))

            # Coastline (sea) near the city (for beach / boardwalk / wetlands).
            self.coast_dir = int(state._hash2_u32(61, 67, self.seed) % 4)  # 0=N,1=E,2=S,3=W
            coast_offset_chunks = int(self.city_radius) + 4
            if self.coast_dir == 0:  # north
                self.coast_line = (int(self.city_cy) - coast_offset_chunks) * int(state.CHUNK_SIZE)
            elif self.coast_dir == 1:  # east
                self.coast_line = (int(self.city_cx) + coast_offset_chunks) * int(state.CHUNK_SIZE)
            elif self.coast_dir == 2:  # south
                self.coast_line = (int(self.city_cy) + coast_offset_chunks) * int(state.CHUNK_SIZE)
            else:  # west
                self.coast_line = (int(self.city_cx) - coast_offset_chunks) * int(state.CHUNK_SIZE)

            # Abandoned highways: two wide bands around the city outskirts.
            self.highway_w = 4  # half-width (tiles)
            hw_off_y = int(state._hash2_u32(97, 101, self.seed) % (state.CHUNK_SIZE * 2)) - int(state.CHUNK_SIZE)
            hw_off_x = int(state._hash2_u32(103, 107, self.seed) % (state.CHUNK_SIZE * 2)) - int(state.CHUNK_SIZE)
            self.highway_y = (int(self.city_cy) + int(self.city_radius) + 2) * int(state.CHUNK_SIZE) + int(hw_off_y)
            self.highway_x = (int(self.city_cx) - int(self.city_radius) - 2) * int(state.CHUNK_SIZE) + int(hw_off_x)

            # Chinese residential compound (小区) near spawn: one enclosed community
            # with ~7-8 tall towers and a low-rise podium ("裙楼") around them.
            self.compound_w_chunks = 5
            self.compound_h_chunks = 5
            self.compound_cx0 = int(self.city_cx) - int(self.compound_w_chunks // 2)
            # Place the compound just north of spawn so you can approach the gate.
            self.compound_cy1 = int(self.city_cy) - 1
            self.compound_cy0 = int(self.compound_cy1) - int(self.compound_h_chunks - 1)
            self.compound_cx1 = int(self.compound_cx0) + int(self.compound_w_chunks) - 1
            self.compound_cy1 = int(self.compound_cy0) + int(self.compound_h_chunks) - 1
            self.compound_tx0 = int(self.compound_cx0) * int(state.CHUNK_SIZE)
            self.compound_ty0 = int(self.compound_cy0) * int(state.CHUNK_SIZE)
            self.compound_tx1 = (int(self.compound_cx1) + 1) * int(state.CHUNK_SIZE) - 1
            self.compound_ty1 = (int(self.compound_cy1) + 1) * int(state.CHUNK_SIZE) - 1

            # Internal road stripes (local coords within each chunk).
            self.compound_road_v_x0 = 14
            self.compound_road_v_x1 = 17
            self.compound_road_h_y0 = 24
            self.compound_road_h_y1 = 26

            # Gate opening (south wall) aligned with the central chunk column.
            gate_cx = int(self.compound_cx0) + int(self.compound_w_chunks // 2)
            self.compound_gate_x0 = int(gate_cx) * int(state.CHUNK_SIZE) + int(self.compound_road_v_x0)
            self.compound_gate_x1 = int(gate_cx) * int(state.CHUNK_SIZE) + int(self.compound_road_v_x1)

            # Reserve entrance/parking chunks near the gate.
            self.compound_gate_chunks: set[tuple[int, int]] = {
                (int(gate_cx), int(self.compound_cy1)),
                (int(gate_cx) - 1, int(self.compound_cy1)),
                (int(gate_cx) + 1, int(self.compound_cy1)),
            }
            self.compound_parking_chunks: set[tuple[int, int]] = {
                (int(gate_cx) - 1, int(self.compound_cy1)),
                (int(gate_cx) + 1, int(self.compound_cy1)),
            }
            self.compound_towers_by_chunk: dict[
                tuple[int, int], list[tuple[int, int, int, int, int, int]]
            ] = self._plan_compound_towers(target=8)

        def spawn_tile(self) -> tuple[int, int]:
            tx = self.state.ROAD_HALF - int(self.road_off_x)
            ty = self.state.ROAD_HALF - int(self.road_off_y)
            return tx, ty

        def _is_road_x(self, tx: int) -> bool:
            d = ((tx + int(self.road_off_x)) % self.state.ROAD_PERIOD) - self.state.ROAD_HALF
            return abs(int(d)) <= self.state.ROAD_W

        def _is_road_y(self, ty: int) -> bool:
            d = ((ty + int(self.road_off_y)) % self.state.ROAD_PERIOD) - self.state.ROAD_HALF
            return abs(int(d)) <= self.state.ROAD_W

        def is_road(self, tx: int, ty: int) -> bool:
            return self._is_road_x(tx) or self._is_road_y(ty)

        def is_highway(self, tx: int, ty: int) -> bool:
            tx = int(tx)
            ty = int(ty)
            w = int(getattr(self, "highway_w", 0))
            if w <= 0:
                return False
            hx = int(getattr(self, "highway_x", 0))
            hy = int(getattr(self, "highway_y", 0))
            return abs(tx - hx) <= w or abs(ty - hy) <= w

        def _coast_distance(self, tx: int, ty: int) -> int:
            # Signed distance from coastline (>=0 is sea side).
            tx = int(tx)
            ty = int(ty)
            cd = int(getattr(self, "coast_dir", 0)) & 3
            line = int(getattr(self, "coast_line", 0))
            if cd == 0:  # north
                return int(line - ty)
            if cd == 2:  # south
                return int(ty - line)
            if cd == 1:  # east
                return int(tx - line)
            return int(line - tx)  # west

        def _is_city_chunk(self, cx: int, cy: int) -> bool:
            dx = abs(int(cx) - int(getattr(self, "city_cx", 0)))
            dy = abs(int(cy) - int(getattr(self, "city_cy", 0)))
            r = max(0, int(getattr(self, "city_radius", 0)))
            return max(dx, dy) <= r

        def _is_compound_chunk(self, cx: int, cy: int) -> bool:
            cx0 = int(getattr(self, "compound_cx0", 0))
            cy0 = int(getattr(self, "compound_cy0", 0))
            cx1 = int(getattr(self, "compound_cx1", -1))
            cy1 = int(getattr(self, "compound_cy1", -1))
            return int(cx0) <= int(cx) <= int(cx1) and int(cy0) <= int(cy) <= int(cy1)

        def _plan_compound_towers(
            self,
            *,
            target: int,
        ) -> dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]]:
            target = int(target)
            if target <= 0:
                return {}

            cx0 = int(getattr(self, "compound_cx0", 0))
            cy0 = int(getattr(self, "compound_cy0", 0))
            cx1 = int(getattr(self, "compound_cx1", -1))
            cy1 = int(getattr(self, "compound_cy1", -1))
            if cx1 < cx0 or cy1 < cy0:
                return {}

            # One tower per inner chunk (7-8 total); keep the perimeter for the podium/skirt building.
            tower_w = 12
            tower_h = 22
            inner_cx0 = int(cx0) + 1
            inner_cx1 = int(cx1) - 1
            inner_cy0 = int(cy0) + 1
            inner_cy1 = int(cy1) - 1
            if inner_cx1 < inner_cx0 or inner_cy1 < inner_cy0:
                return {}

            inner_chunks: list[tuple[int, int]] = []
            for cy in range(int(inner_cy0), int(inner_cy1) + 1):
                for cx in range(int(inner_cx0), int(inner_cx1) + 1):
                    inner_chunks.append((int(cx), int(cy)))
            if not inner_chunks:
                return {}

            # Keep the very center chunk empty as a courtyard/green space when possible.
            center = ((int(inner_cx0) + int(inner_cx1)) // 2, (int(inner_cy0) + int(inner_cy1)) // 2)
            inner_chunks_usable = [c for c in inner_chunks if c != center]
            if not inner_chunks_usable:
                inner_chunks_usable = inner_chunks

            rng = random.Random(self.state._hash2_u32(211, 223, self.seed ^ 0x51C17011))
            rng.shuffle(inner_chunks_usable)
            picked = inner_chunks_usable[: int(clamp(int(target), 1, len(inner_chunks_usable)))]

            plan: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
            for cx, cy in picked:
                # Alternate left/right so we don't overwrite the internal road stripe at x=14..17.
                x0 = 2 if ((int(cx) + int(cy)) % 2 == 0) else 18
                y0 = 2
                base_tx = int(cx) * int(self.state.CHUNK_SIZE)
                base_ty = int(cy) * int(self.state.CHUNK_SIZE)
                tx0w = int(base_tx + int(x0))
                ty0w = int(base_ty + int(y0))
                rr = random.Random(self.state._hash2_u32(int(tx0w), int(ty0w), self.seed ^ 0xA17F1A2B))
                roof_var = int(rr.randrange(0, 256))
                floors = int(rr.randint(20, 30))
                plan.setdefault((int(cx), int(cy)), []).append(
                    (int(x0), int(y0), int(tower_w), int(tower_h), int(roof_var), int(floors))
                )
            return plan

        def _is_city_street_x(self, tx: int) -> bool:
            period = max(8, int(getattr(self, "city_period", 24)))
            half = int(period // 2)
            w = max(1, int(getattr(self, "city_w", 1)))
            off = int(getattr(self, "city_off_x", 0)) % period
            d = ((int(tx) + off) % period) - half
            return abs(int(d)) <= w

        def _is_city_street_y(self, ty: int) -> bool:
            period = max(8, int(getattr(self, "city_period", 24)))
            half = int(period // 2)
            w = max(1, int(getattr(self, "city_w", 1)))
            off = int(getattr(self, "city_off_y", 0)) % period
            d = ((int(ty) + off) % period) - half
            return abs(int(d)) <= w

        def is_city_street(self, tx: int, ty: int) -> bool:
            return self._is_city_street_x(tx) or self._is_city_street_y(ty)

        def get_tile(self, tx: int, ty: int) -> int:
            cx = tx // self.state.CHUNK_SIZE
            cy = ty // self.state.CHUNK_SIZE
            lx = tx - cx * self.state.CHUNK_SIZE
            ly = ty - cy * self.state.CHUNK_SIZE
            chunk = self.get_chunk(cx, cy)
            return chunk.tiles[ly * self.state.CHUNK_SIZE + lx]

        def peek_chunk(self, cx: int, cy: int) -> "HardcoreSurvivalState._Chunk | None":
            return self.chunks.get((int(cx), int(cy)))

        def peek_tile(self, tx: int, ty: int, *, default: int | None = None) -> int:
            cx = tx // self.state.CHUNK_SIZE
            cy = ty // self.state.CHUNK_SIZE
            lx = tx - cx * self.state.CHUNK_SIZE
            ly = ty - cy * self.state.CHUNK_SIZE
            chunk = self.peek_chunk(cx, cy)
            if chunk is None:
                return int(self.state.T_GRASS if default is None else default)
            return int(chunk.tiles[int(ly) * int(self.state.CHUNK_SIZE) + int(lx)])

        def request_chunk(self, cx: int, cy: int) -> None:
            key = (int(cx), int(cy))
            if key in self.chunks or key in self._gen_queue_set:
                return
            self._gen_queue.append(key)
            self._gen_queue_set.add(key)

        def pump_generation(self, *, max_chunks: int = 1) -> int:
            made = 0
            max_chunks = max(0, int(max_chunks))
            while made < max_chunks and self._gen_queue_i < len(self._gen_queue):
                key = self._gen_queue[self._gen_queue_i]
                self._gen_queue_i += 1
                self._gen_queue_set.discard(key)
                if key not in self.chunks:
                    self.chunks[key] = self._gen_chunk(int(key[0]), int(key[1]))
                made += 1

            # Compact occasionally to avoid unbounded growth.
            if self._gen_queue_i > 256 and self._gen_queue_i > (len(self._gen_queue) // 2):
                self._gen_queue = self._gen_queue[self._gen_queue_i :]
                self._gen_queue_i = 0
            return int(made)

        def get_chunk(self, cx: int, cy: int) -> "HardcoreSurvivalState._Chunk":
            key = (int(cx), int(cy))
            chunk = self.chunks.get(key)
            if chunk is None:
                chunk = self._gen_chunk(int(cx), int(cy))
                self.chunks[key] = chunk
            return chunk

        def _chunk_has_intersection(self, cx: int, cy: int) -> bool:
            base_tx = cx * self.state.CHUNK_SIZE
            base_ty = cy * self.state.CHUNK_SIZE
            has_x = any(self._is_road_x(base_tx + i) for i in range(self.state.CHUNK_SIZE))
            has_y = any(self._is_road_y(base_ty + i) for i in range(self.state.CHUNK_SIZE))
            return has_x and has_y

        def _is_town_chunk(self, cx: int, cy: int) -> bool:
            if self._chunk_has_intersection(cx, cy):
                return True
            r = self.state._rand01(cx, cy, self.seed ^ 0xA17F1A2B)
            return r >= 0.92

        def _gen_chunk(self, cx: int, cy: int) -> "HardcoreSurvivalState._Chunk":
            tiles = [self.state.T_GRASS] * (self.state.CHUNK_SIZE * self.state.CHUNK_SIZE)
            base_tx = cx * self.state.CHUNK_SIZE
            base_ty = cy * self.state.CHUNK_SIZE
            items: list[HardcoreSurvivalState._WorldItem] = []
            buildings: list[tuple[int, int, int, int, int, int]] = []
            special_buildings: list[HardcoreSurvivalState._SpecialBuilding] = []
            cars: list[HardcoreSurvivalState._ParkedCar] = []
            bikes: list[HardcoreSurvivalState._ParkedBike] = []
            props: list[HardcoreSurvivalState._WorldProp] = []
            door_reserved: set[tuple[int, int]] = set()

            is_city = self._is_city_chunk(int(cx), int(cy))

            for ly in range(self.state.CHUNK_SIZE):
                ty = base_ty + ly
                for lx in range(self.state.CHUNK_SIZE):
                    tx = base_tx + lx
                    if is_city:
                        # City ground zoning: highways -> roads -> sidewalks -> district floors.
                        if self.is_highway(tx, ty):
                            tile = int(self.state.T_HIGHWAY)
                        elif self.is_road(tx, ty) or self.is_city_street(tx, ty):
                            tile = int(self.state.T_ROAD)
                        else:
                            near_road = bool(
                                self.is_highway(tx - 1, ty)
                                or self.is_highway(tx + 1, ty)
                                or self.is_highway(tx, ty - 1)
                                or self.is_highway(tx, ty + 1)
                                or self.is_road(tx - 1, ty)
                                or self.is_road(tx + 1, ty)
                                or self.is_road(tx, ty - 1)
                                or self.is_road(tx, ty + 1)
                                or self.is_city_street(tx - 1, ty)
                                or self.is_city_street(tx + 1, ty)
                                or self.is_city_street(tx, ty - 1)
                                or self.is_city_street(tx, ty + 1)
                            )
                            if near_road:
                                tile = int(self.state.T_SIDEWALK)
                            else:
                                district = float(
                                    self.state._noise2(float(tx) / 96.0, float(ty) / 96.0, self.seed ^ 0x7A11C1D3)
                                )
                                grit = float(
                                    self.state._noise2(float(tx) / 24.0, float(ty) / 24.0, self.seed ^ 0x1B7D3D39)
                                )
                                if district > 0.70:
                                    tile = int(self.state.T_BRICK)
                                elif district > 0.42:
                                    tile = int(self.state.T_CONCRETE)
                                else:
                                    tile = int(self.state.T_PAVEMENT)

                                # Occasional brick courtyards / plazas inside blocks.
                                if tile != int(self.state.T_BRICK) and grit > 0.885:
                                    tile = int(self.state.T_BRICK)
                                elif tile == int(self.state.T_PAVEMENT) and grit < 0.14:
                                    tile = int(self.state.T_CONCRETE)

                                # Small parks inside the city (keep roads/sidewalks clean).
                                park = float(
                                    self.state._noise2(float(tx) / 48.0, float(ty) / 48.0, self.seed ^ 0x51C17011)
                                )
                                if park > 0.905:
                                    tile = int(self.state.T_FOREST)
                                elif park > 0.84:
                                    tile = int(self.state.T_GRASS)
                    else:
                        n = self.state._fractal(float(tx), float(ty), self.seed)
                        coast_d = int(self._coast_distance(tx, ty))
                        if coast_d >= 10:
                            tile = self.state.T_WATER
                        elif coast_d >= 3:
                            # Shallows with occasional sand bars.
                            tile = self.state.T_WATER
                            shoal = float(self.state._noise2(float(tx) / 18.0, float(ty) / 18.0, self.seed ^ 0x0CEAD511))
                            if shoal > 0.88:
                                tile = self.state.T_SAND
                        elif coast_d >= 0:
                            tile = self.state.T_WATER
                        elif coast_d >= -4:
                            tile = self.state.T_SAND
                        elif coast_d >= -7:
                            # Boardwalk band just inland from the beach.
                            bw = float(self.state._noise2(float(tx) / 12.0, float(ty) / 12.0, self.seed ^ 0xB04A7A1C))
                            tile = self.state.T_BOARDWALK if bw > 0.55 else self.state.T_SAND
                        elif coast_d >= -18:
                            # Coastal wetlands behind the beach.
                            wet = float(self.state._noise2(float(tx) / 24.0, float(ty) / 24.0, self.seed ^ 0x717E17D5))
                            if wet > 0.92:
                                tile = self.state.T_WATER
                            elif wet > 0.78:
                                tile = self.state.T_MARSH
                            else:
                                if n < 0.30:
                                    tile = self.state.T_FOREST
                                else:
                                    tile = self.state.T_GRASS
                        else:
                            if n < 0.24:
                                tile = self.state.T_WATER
                            elif n < 0.36:
                                tile = self.state.T_FOREST
                            else:
                                tile = self.state.T_GRASS

                        # Roads/highways override terrain (but keep deep sea water intact).
                        if tile != self.state.T_WATER and self.is_highway(tx, ty):
                            tile = self.state.T_HIGHWAY
                        elif self.is_road(tx, ty):
                            tile = self.state.T_ROAD
                    tiles[ly * self.state.CHUNK_SIZE + lx] = tile

            town_kind: str | None = None
            if is_city:
                rng = random.Random(self.state._hash2_u32(cx, cy, self.seed ^ 0xC17C1701))
                self._stamp_city_chunk(
                    tiles,
                    items,
                    buildings,
                    special_buildings,
                    cars,
                    bikes,
                    props,
                    cx,
                    cy,
                    rng,
                    reserved=door_reserved,
                )
                town_kind = "城市"
            elif self._is_town_chunk(cx, cy):
                rng = random.Random(self.state._hash2_u32(cx, cy, self.seed))
                kinds = ["超市", "新华书店", "医院", "监狱", "学校", "高层住宅"]
                stx, sty = self.spawn_tile()
                is_spawn_chunk = (int(stx) // self.state.CHUNK_SIZE == int(cx)) and (int(sty) // self.state.CHUNK_SIZE == int(cy))
                if is_spawn_chunk:
                    plan = ["高层住宅", "学校"]
                else:
                    plan = [rng.choice(kinds)]
                    if plan[0] != "监狱" and rng.random() < 0.35:
                        pool = [k for k in kinds if k not in plan and k != "监狱"]
                        if pool:
                            plan.append(rng.choice(pool))

                town_kind = "+".join(plan)
                for kind in plan:
                    self._stamp_buildings(
                        tiles,
                        items,
                        buildings,
                        special_buildings,
                        props,
                        cx,
                        cy,
                        rng,
                        kind,
                        reserved=door_reserved,
                    )

            if not is_city:
                # Beach / wetlands / highway decoration on non-city chunks.
                rng2 = random.Random(self.state._hash2_u32(cx, cy, self.seed ^ 0x0D00BDEC))
                self._stamp_outdoor_decor(tiles, props, cx, cy, rng=rng2)

            return HardcoreSurvivalState._Chunk(
                cx=cx,
                cy=cy,
                tiles=tiles,
                items=items,
                buildings=buildings,
                special_buildings=special_buildings,
                cars=cars,
                bikes=bikes,
                props=props,
                town_kind=town_kind,
            )

        def _stamp_compound_highrise_tower(
            self,
            tiles: list[int],
            buildings: list[tuple[int, int, int, int, int, int]],
            special_buildings: list["HardcoreSurvivalState._SpecialBuilding"],
            cx: int,
            cy: int,
            *,
            x0: int,
            y0: int,
            w: int,
            h: int,
            roof_var: int,
            floors: int,
        ) -> None:
            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            x0 = int(x0)
            y0 = int(y0)
            w = int(w)
            h = int(h)
            roof_var = int(roof_var) & 0xFF
            floors = int(floors)

            # Fill the whole footprint with walls (sealed on the world-map).
            for y in range(int(y0), int(y0 + h)):
                for x in range(int(x0), int(x0 + w)):
                    if 0 <= x < int(self.state.CHUNK_SIZE) and 0 <= y < int(self.state.CHUNK_SIZE):
                        tiles[idx(int(x), int(y))] = int(self.state.T_WALL)

            # South-facing entrance (2 tiles wide).
            door_y = int(y0 + h - 1)
            door_x0 = int(clamp(int(x0 + w // 2 - 1), int(x0 + 1), int(x0 + w - 3)))
            door_tiles_local = [(int(door_x0), int(door_y)), (int(door_x0 + 1), int(door_y))]
            for dx, dy in door_tiles_local:
                if 0 <= dx < int(self.state.CHUNK_SIZE) and 0 <= dy < int(self.state.CHUNK_SIZE):
                    tiles[idx(int(dx), int(dy))] = int(self.state.T_DOOR)

            # Keep towers sealed on the world map: enter via the door portal only.
            # Register as a high-rise portal.
            tx0w = int(cx) * int(self.state.CHUNK_SIZE) + int(x0)
            ty0w = int(cy) * int(self.state.CHUNK_SIZE) + int(y0)
            doors_w = tuple(
                (int(cx) * int(self.state.CHUNK_SIZE) + int(dx), int(cy) * int(self.state.CHUNK_SIZE) + int(dy))
                for dx, dy in door_tiles_local
            )
            special_buildings.append(
                HardcoreSurvivalState._SpecialBuilding(
                    kind="highrise",
                    name="楂樺眰浣忓畢",
                    tx0=int(tx0w),
                    ty0=int(ty0w),
                    w=int(w),
                    h=int(h),
                    door_tiles=doors_w,
                    floors=int(floors),
                )
            )
            roof_kind = (6 << 8) | int(roof_var)
            buildings.append((int(tx0w), int(ty0w), int(w), int(h), int(roof_kind), int(floors)))

        def _stamp_compound_podium_block(
            self,
            tiles: list[int],
            buildings: list[tuple[int, int, int, int, int, int]],
            cx: int,
            cy: int,
            *,
            x0: int,
            y0: int,
            w: int,
            h: int,
            roof_style: int = 2,
            roof_var: int | None = None,
        ) -> None:
            # Simple low-rise skirt building chunk: solid mass with a roof (visual ring).
            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            x0 = int(x0)
            y0 = int(y0)
            w = int(w)
            h = int(h)
            if w <= 0 or h <= 0:
                return

            for y in range(int(y0), int(y0 + h)):
                if not (0 <= int(y) < int(self.state.CHUNK_SIZE)):
                    continue
                for x in range(int(x0), int(x0 + w)):
                    if not (0 <= int(x) < int(self.state.CHUNK_SIZE)):
                        continue
                    tiles[idx(int(x), int(y))] = int(self.state.T_WALL)

            tx0w = int(cx) * int(self.state.CHUNK_SIZE) + int(x0)
            ty0w = int(cy) * int(self.state.CHUNK_SIZE) + int(y0)
            if roof_var is None:
                roof_var = int(self.state._hash2_u32(int(tx0w), int(ty0w), self.seed ^ 0x7A11C1D3) & 0xFF)
            roof_kind = (int(roof_style) << 8) | (int(roof_var) & 0xFF)
            buildings.append((int(tx0w), int(ty0w), int(w), int(h), int(roof_kind), 0))

        def _stamp_compound_chunk(
            self,
            tiles: list[int],
            buildings: list[tuple[int, int, int, int, int, int]],
            special_buildings: list["HardcoreSurvivalState._SpecialBuilding"],
            cars: list["HardcoreSurvivalState._ParkedCar"],
            bikes: list["HardcoreSurvivalState._ParkedBike"],
            props: list["HardcoreSurvivalState._WorldProp"],
            cx: int,
            cy: int,
            rng: random.Random,
        ) -> None:
            # Walled "小区": ~7-8 tall towers (20+ floors) with a low-rise podium ring.
            cx = int(cx)
            cy = int(cy)

            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)

            tx0 = int(getattr(self, "compound_tx0", 0))
            ty0 = int(getattr(self, "compound_ty0", 0))
            tx1 = int(getattr(self, "compound_tx1", 0))
            ty1 = int(getattr(self, "compound_ty1", 0))
            gate_x0 = int(getattr(self, "compound_gate_x0", 0))
            gate_x1 = int(getattr(self, "compound_gate_x1", -1))
            rvx0 = int(getattr(self, "compound_road_v_x0", 14))
            rvx1 = int(getattr(self, "compound_road_v_x1", 17))
            rhy0 = int(getattr(self, "compound_road_h_y0", 24))
            rhy1 = int(getattr(self, "compound_road_h_y1", 26))

            ccx0 = int(getattr(self, "compound_cx0", cx))
            ccy0 = int(getattr(self, "compound_cy0", cy))
            ccx1 = int(getattr(self, "compound_cx1", cx))
            ccy1 = int(getattr(self, "compound_cy1", cy))
            gate_cx = int(ccx0) + int(int(getattr(self, "compound_w_chunks", 0)) // 2)
            near_gate = bool(int(cy) == int(ccy1) and abs(int(cx) - int(gate_cx)) <= 1)
            is_gate_chunk = bool(int(cy) == int(ccy1) and int(cx) == int(gate_cx))

            parking_chunks = set(getattr(self, "compound_parking_chunks", set()))
            is_parking_chunk = (int(cx), int(cy)) in parking_chunks

            # Base ground (overwrite the city district pattern in the compound area).
            for y in range(int(self.state.CHUNK_SIZE)):
                wy = int(base_ty + y)
                for x in range(int(self.state.CHUNK_SIZE)):
                    wx = int(base_tx + x)

                    tile = int(self.state.T_CONCRETE)
                    # Sparse greenery between blocks.
                    g = float(self.state._noise2(float(wx) / 18.0, float(wy) / 18.0, self.seed ^ 0x7A11C1D3))
                    if g > 0.94:
                        tile = int(self.state.T_GRASS)

                    if not is_parking_chunk:
                        if int(rvx0) <= int(x) <= int(rvx1) or int(rhy0) <= int(y) <= int(rhy1):
                            tile = int(self.state.T_ROAD)

                    # Outer compound wall (leave a driveable opening at the gate).
                    if int(wx) == int(tx0) or int(wx) == int(tx1) or int(wy) == int(ty0) or int(wy) == int(ty1):
                        if int(wy) == int(ty1) and int(gate_x0) <= int(wx) <= int(gate_x1):
                            tile = int(self.state.T_ROAD)  # gate opening (walk/drive through)
                        else:
                            tile = int(self.state.T_WALL)

                    tiles[idx(int(x), int(y))] = int(tile)

            # Podium ("裙楼") ring along the compound perimeter (keep the gate approach open).
            on_edge = bool(int(cx) == int(ccx0) or int(cx) == int(ccx1) or int(cy) == int(ccy0) or int(cy) == int(ccy1))
            if on_edge and (not is_parking_chunk):
                pod_t = 10
                if int(cy) == int(ccy0):
                    self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=2, y0=2, w=28, h=pod_t, roof_style=2)
                if int(cy) == int(ccy1) and (not near_gate):
                    # Keep a few tiles open near the south wall so the player can reach the gate.
                    self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=2, y0=18, w=28, h=pod_t, roof_style=2)
                if int(cx) == int(ccx0):
                    self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=2, y0=2, w=pod_t, h=28, roof_style=2)
                if int(cx) == int(ccx1):
                    self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=20, y0=2, w=pod_t, h=28, roof_style=2)

            # Gatehouse blocks (1F) framing the opening.
            if is_gate_chunk:
                # Two small wings near the south wall, leave the center corridor clear.
                self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=2, y0=22, w=10, h=8, roof_style=2, roof_var=64)
                self._stamp_compound_podium_block(tiles, buildings, cx, cy, x0=20, y0=22, w=10, h=8, roof_style=2, roof_var=96)
                # A sign prop near the gate so it reads as an entrance.
                gx = int(clamp(int((gate_x0 + gate_x1) // 2), int(base_tx), int(base_tx + self.state.CHUNK_SIZE - 1)))
                gy = int(ty1 - 1)
                props.append(
                    HardcoreSurvivalState._WorldProp(
                        pos=pygame.Vector2((float(gx) + 0.5) * float(self.state.TILE_SIZE), (float(gy) + 0.5) * float(self.state.TILE_SIZE)),
                        prop_id="sign_chinese",
                        variant=int(rng.randint(0, 3)),
                        dir="down",
                    )
                )

            # Parking lots near the gate (visual interest).
            if is_parking_chunk:
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
                self._spawn_two_wheelers(
                    tiles,
                    bikes,
                    cx,
                    cy,
                    rng=rng,
                    count=int(rng.randint(1, 3)),
                    models=["bike_lady", "bike_mountain", "bike_auto", "moto"],
                    buildings=buildings,
                )

            # Stamp planned towers for this chunk.
            towers = (getattr(self, "compound_towers_by_chunk", {}) or {}).get((int(cx), int(cy)), [])
            for x0, y0, w, h, roof_var, floors in towers:
                self._stamp_compound_highrise_tower(
                    tiles,
                    buildings,
                    special_buildings,
                    cx,
                    cy,
                    x0=int(x0),
                    y0=int(y0),
                    w=int(w),
                    h=int(h),
                    roof_var=int(roof_var),
                    floors=int(floors),
                )

        def _stamp_city_chunk(
            self,
            tiles: list[int],
            items: list["HardcoreSurvivalState._WorldItem"],
            buildings: list[tuple[int, int, int, int, int, int]],
            special_buildings: list["HardcoreSurvivalState._SpecialBuilding"],
            cars: list["HardcoreSurvivalState._ParkedCar"],
            bikes: list["HardcoreSurvivalState._ParkedBike"],
            props: list["HardcoreSurvivalState._WorldProp"],
            cx: int,
            cy: int,
            rng: random.Random,
            reserved: set[tuple[int, int]] | None = None,
        ) -> None:
            # Dense city composition: mix facilities + residential blocks.
            roll = float(rng.random())
            dist = max(abs(int(cx) - int(getattr(self, "city_cx", 0))), abs(int(cy) - int(getattr(self, "city_cy", 0))))
            radius = max(1, int(getattr(self, "city_radius", 1)))
            density = 1.0 - float(dist) / float(radius)
            buildings_n = int(clamp(3 + int(round(5.0 * density)), 3, 8))

            # Chunk-level districts so city blocks feel different.
            district = float(self.state._noise2(float(cx) / 3.4, float(cy) / 3.4, self.seed ^ 0x4C17C0DE))
            chinese_district = bool(district > 0.78)
            commercial_district = bool(district < 0.22)

            core = bool(density >= 0.72)
            stx, sty = self.spawn_tile()
            spawn_cx = int(stx) // self.state.CHUNK_SIZE
            spawn_cy = int(sty) // self.state.CHUNK_SIZE
            if self._is_compound_chunk(int(cx), int(cy)):
                self._stamp_compound_chunk(
                    tiles,
                    buildings,
                    special_buildings,
                    cars,
                    bikes,
                    props,
                    cx,
                    cy,
                    rng,
                )
                return
            is_spawn_chunk = (int(spawn_cx) == int(cx)) and (int(spawn_cy) == int(cy))
            # Starter "Chinese community": a small cluster of high-rises near spawn.
            in_community = (int(spawn_cx) <= int(cx) <= int(spawn_cx) + 1) and (int(spawn_cy) <= int(cy) <= int(spawn_cy) + 1)
            school_chunk = (int(cx) == int(spawn_cx) + 1) and (int(cy) == int(spawn_cy))
            # Dedicated compound exists now; disable the old starter-community cluster here.
            use_starter_community = False
            if use_starter_community and is_spawn_chunk:
                # Spawn block: guarantee a home high-rise; the school is in an adjacent
                # community chunk to keep this block more residential.
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)
                # Add a few extra towers so it reads like a "小区".
                for _ in range(int(rng.randint(1, 3))):
                    if rng.random() < 0.85:
                        self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)
                if rng.random() < 0.35:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅大", reserved=reserved)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, school=True, buildings=buildings)
                self._spawn_two_wheelers(
                    tiles,
                    bikes,
                    cx,
                    cy,
                    rng=rng,
                    count=rng.randint(6, 12),
                    models=["bike_lady", "bike_mountain", "bike_auto", "moto", "moto_lux", "moto_long"],
                    buildings=buildings,
                )
                buildings_n = max(3, buildings_n - 2)
            elif use_starter_community and in_community:
                # Community chunks: mostly residential towers; one chunk hosts the school.
                if school_chunk:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "学校", reserved=reserved)
                    self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, school=True, buildings=buildings)
                else:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)
                    if rng.random() < 0.55:
                        self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)
                    if core and rng.random() < 0.22:
                        self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅大", reserved=reserved)
                    if rng.random() < 0.55:
                        self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
                buildings_n = max(3, buildings_n - 1)
            elif core:
                big_roll = float(rng.random())
                if big_roll < 0.07:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "大型超市", reserved=reserved)
                    self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
                    buildings_n = max(3, buildings_n - 1)
                elif big_roll < 0.12:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅大", reserved=reserved)
                    if rng.random() < 0.35:
                        self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)
                    buildings_n = max(3, buildings_n - 1)
                elif big_roll < 0.18:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "大型监狱", reserved=reserved)
                    buildings_n = max(3, buildings_n - 1)
                elif big_roll < 0.24 and commercial_district:
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "新华书店", reserved=reserved)
                    self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
                    buildings_n = max(3, buildings_n - 1)

            if chinese_district and (not is_spawn_chunk) and rng.random() < (0.55 + 0.25 * density):
                # A small cluster of Chinese-style buildings.
                n = 2 if (density > 0.55 and rng.random() < 0.55) else 1
                for _ in range(int(n)):
                    self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "中式建筑", reserved=reserved)
                buildings_n = max(3, int(buildings_n) - int(n))

            if roll < 0.16:
                self._stamp_basketball_court(tiles, rng=rng)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
            elif roll < 0.32:
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "学校", reserved=reserved)
                self._stamp_basketball_court(tiles, rng=rng)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, school=True, buildings=buildings)
                self._spawn_two_wheelers(tiles, bikes, cx, cy, rng=rng, count=rng.randint(5, 12), models=["bike_lady", "bike_mountain", "bike_auto", "moto", "moto_lux", "moto_long"], buildings=buildings)
                buildings_n = max(3, buildings_n - 1)
            elif roll < 0.46:
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "医院", reserved=reserved)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
            elif roll < 0.60:
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "超市", reserved=reserved)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
            elif roll < 0.68 and commercial_district:
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "新华书店", reserved=reserved)
                self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
            elif roll < 0.74:
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, "高层住宅", reserved=reserved)

            for _ in range(int(buildings_n)):
                if chinese_district and rng.random() < 0.55:
                    kind = "中式建筑"
                else:
                    if commercial_district:
                        pool = ["住宅", "住宅", "住宅", "住宅", "超市", "医院", "新华书店", "高层住宅", "监狱"]
                    else:
                        pool = ["住宅", "住宅", "住宅", "住宅", "住宅", "住宅", "超市", "医院", "高层住宅", "监狱"]
                    kind = str(rng.choice(pool))
                if core:
                    if kind == "超市" and rng.random() < 0.45:
                        kind = "大型超市"
                    elif kind == "高层住宅" and rng.random() < 0.45:
                        kind = "高层住宅大"
                    elif kind == "监狱" and rng.random() < 0.35:
                        kind = "大型监狱"
                if kind in ("超市", "医院", "大型超市") and rng.random() < 0.55:
                    kind = "住宅"
                self._stamp_buildings(tiles, items, buildings, special_buildings, props, cx, cy, rng, kind, reserved=reserved)

            # Make parking lots common/visible in city blocks (esp. near core).
            # (Parking stamping is safe: it avoids overwriting buildings/roads.)
            if not is_spawn_chunk:
                lot_chance = float(clamp(0.24 + 0.40 * density, 0.18, 0.72))
                if rng.random() < lot_chance:
                    self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)
                if core and rng.random() < 0.28:
                    self._stamp_parking_lot(tiles, cars, bikes, cx, cy, rng=rng, buildings=buildings)

            self._stamp_city_decor(tiles, props, cx, cy, rng=rng)

        def _stamp_basketball_court(self, tiles: list[int], *, rng: random.Random) -> None:
            # Simple outdoor court: a rectangle of court tiles.
            w = rng.randint(10, 14)
            h = rng.randint(8, 10)
            x0 = rng.randint(2, self.state.CHUNK_SIZE - w - 2)
            y0 = rng.randint(2, self.state.CHUNK_SIZE - h - 2)

            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            for y in range(int(y0), int(y0 + h)):
                for x in range(int(x0), int(x0 + w)):
                    if int(tiles[idx(x, y)]) == int(self.state.T_ROAD):
                        return
            for y in range(int(y0), int(y0 + h)):
                for x in range(int(x0), int(x0 + w)):
                    tiles[idx(x, y)] = int(self.state.T_COURT)

        def _stamp_parking_lot(
            self,
            tiles: list[int],
            cars: list["HardcoreSurvivalState._ParkedCar"],
            bikes: list["HardcoreSurvivalState._ParkedBike"],
            cx: int,
            cy: int,
            *,
            rng: random.Random,
            school: bool = False,
            buildings: list[tuple[int, int, int, int, int, int]] | None = None,
        ) -> None:
            # Parking lot: stamp tiles + spawn parked vehicles (visual props).
            allowed_base = {
                int(self.state.T_GRASS),
                int(self.state.T_FOREST),
                int(self.state.T_PAVEMENT),
                int(self.state.T_PARKING),
                int(self.state.T_SIDEWALK),
                int(self.state.T_BRICK),
                int(self.state.T_CONCRETE),
            }

            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            placed: tuple[int, int, int, int] | None = None
            for _ in range(18):
                w = rng.randint(12, 20)
                h = rng.randint(9, 14)
                x0 = rng.randint(2, self.state.CHUNK_SIZE - w - 2)
                y0 = rng.randint(2, self.state.CHUNK_SIZE - h - 2)

                blocked = False
                for y in range(int(y0), int(y0 + h)):
                    for x in range(int(x0), int(x0 + w)):
                        t = int(tiles[idx(x, y)])
                        if t not in allowed_base:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                placed = (int(x0), int(y0), int(w), int(h))
                break
            if placed is None:
                return

            x0, y0, w, h = placed
            for y in range(int(y0), int(y0 + h)):
                for x in range(int(x0), int(x0 + w)):
                    tiles[idx(x, y)] = int(self.state.T_PARKING)

            # Spawn cars in a loose grid.
            car_pool = [mid for mid in self.state._CAR_MODELS.keys() if str(mid) != "rv"]
            if not car_pool:
                return
            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)
            spots: list[tuple[int, int]] = []
            for yy in range(int(y0) + 1, int(y0 + h) - 1, 3):
                for xx in range(int(x0) + 1, int(x0 + w) - 1, 4):
                    if int(tiles[idx(xx, yy)]) != int(self.state.T_PARKING):
                        continue
                    spots.append((int(xx), int(yy)))
            if not spots:
                return
            rng.shuffle(spots)
            n_cars = int(clamp(int(rng.randint(3, 9)), 0, len(spots)))
            for i in range(n_cars):
                sx, sy = spots[i]
                tx = base_tx + int(sx)
                ty = base_ty + int(sy)
                px = (float(tx) + 0.5) * float(self.state.TILE_SIZE)
                py = (float(ty) + 0.5) * float(self.state.TILE_SIZE)
                model_id = str(rng.choice(car_pool))
                heading = 0.0 if (i % 2 == 0) else (math.pi / 2)
                cars.append(HardcoreSurvivalState._ParkedCar(pos=pygame.Vector2(px, py), model_id=model_id, heading=float(heading)))

            if school:
                # A few bikes near the parking lot.
                self._spawn_two_wheelers(
                    tiles,
                    bikes,
                    cx,
                    cy,
                    rng=rng,
                    count=rng.randint(2, 5),
                    models=["bike_lady", "bike_mountain", "bike_auto"],
                    buildings=buildings,
                )

        def _spawn_two_wheelers(
            self,
            tiles: list[int],
            bikes: list["HardcoreSurvivalState._ParkedBike"],
            cx: int,
            cy: int,
            *,
            rng: random.Random,
            count: int,
            models: list[str],
            buildings: list[tuple[int, int, int, int, int, int]] | None = None,
        ) -> None:
            if count <= 0 or not models:
                return

            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)
            candidates: list[tuple[int, int]] = []
            for y in range(2, self.state.CHUNK_SIZE - 2):
                for x in range(2, self.state.CHUNK_SIZE - 2):
                    t = int(tiles[idx(int(x), int(y))])
                    if t in (int(self.state.T_ROAD), int(self.state.T_WALL), int(self.state.T_DOOR)):
                        continue
                    if t not in (
                        int(self.state.T_PAVEMENT),
                        int(self.state.T_PARKING),
                        int(self.state.T_SIDEWALK),
                        int(self.state.T_BRICK),
                        int(self.state.T_CONCRETE),
                    ):
                        continue
                    # Avoid spawning right next to building walls/doors (looks like bikes are "on the facade").
                    near_wall = False
                    for ny in range(int(y) - 2, int(y) + 3):
                        for nx in range(int(x) - 2, int(x) + 3):
                            if not (0 <= int(nx) < int(self.state.CHUNK_SIZE) and 0 <= int(ny) < int(self.state.CHUNK_SIZE)):
                                continue
                            nt = int(tiles[idx(int(nx), int(ny))])        
                            if nt in (int(self.state.T_WALL), int(self.state.T_DOOR)):
                                near_wall = True
                                break
                        if near_wall:
                            break
                    if near_wall:
                        continue
                    candidates.append((int(x), int(y)))
            if not candidates:
                return
            rng.shuffle(candidates)
            want = int(clamp(int(count), 0, len(candidates)))
            dirs = ["up", "down", "left", "right"]
            tile_size = int(self.state.TILE_SIZE)
            chunk_size = int(self.state.CHUNK_SIZE)
            t_floor = int(self.state.T_FLOOR)
            t_door = int(self.state.T_DOOR)

            def tile_blocks_vehicle(t: int) -> bool:
                t = int(t)
                if t in (t_floor, t_door):
                    return True
                tdef = self.state._TILES.get(int(t))
                if tdef is None:
                    return False
                return bool(getattr(tdef, "solid", False))

            def rect_clear(r: pygame.Rect) -> bool:
                left = int(math.floor(r.left / tile_size))
                right = int(math.floor((r.right - 1) / tile_size))
                top = int(math.floor(r.top / tile_size))
                bottom = int(math.floor((r.bottom - 1) / tile_size))
                for ty in range(int(top), int(bottom) + 1):
                    ly = int(ty - base_ty)
                    if not (0 <= ly < chunk_size):
                        return False
                    for tx in range(int(left), int(right) + 1):
                        lx = int(tx - base_tx)
                        if not (0 <= lx < chunk_size):
                            return False
                        if tile_blocks_vehicle(int(tiles[idx(int(lx), int(ly))])):
                            return False
                return True

            def inside_any_building(tx: int, ty: int) -> bool:
                if not buildings:
                    return False
                for bx0, by0, bw, bh, _roof_kind, _floors in buildings:
                    if int(bx0) <= int(tx) < int(bx0) + int(bw) and int(by0) <= int(ty) < int(by0) + int(bh):
                        return True
                return False

            placed = 0
            for sx, sy in candidates:
                if placed >= want:
                    break
                tx = base_tx + int(sx)
                ty = base_ty + int(sy)
                if inside_any_building(int(tx), int(ty)):
                    continue

                mid = str(rng.choice(models))
                w_px, h_px = self.state._two_wheel_collider_px(mid)
                px = (float(tx) + 0.5) * float(tile_size)
                py = (float(ty) + 0.5) * float(tile_size)
                rect = pygame.Rect(
                    int(iround(float(px) - float(w_px) / 2.0)),
                    int(iround(float(py) - float(h_px) / 2.0)),
                    int(w_px),
                    int(h_px),
                )
                if not rect_clear(rect):
                    continue

                d = str(rng.choice(dirs))
                fuel = 0.0
                if str(mid).startswith("moto"):
                    fuel = float(rng.uniform(25.0, 95.0))
                elif str(mid) == "bike_auto":
                    fuel = float(rng.uniform(40.0, 100.0))
                bikes.append(
                    HardcoreSurvivalState._ParkedBike(pos=pygame.Vector2(px, py), model_id=mid, dir=d, fuel=float(fuel))
                )
                placed += 1

        def _stamp_outdoor_decor(
            self,
            tiles: list[int],
            props: list["HardcoreSurvivalState._WorldProp"],
            cx: int,
            cy: int,
            *,
            rng: random.Random,
        ) -> None:
            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)

            def far_from_existing(px: float, py: float, *, min_d: float) -> bool:
                md2 = float(min_d) * float(min_d)
                for p in props:
                    pos = getattr(p, "pos", None)
                    if pos is None:
                        continue
                    d2 = float((pygame.Vector2(pos) - pygame.Vector2(px, py)).length_squared())
                    if d2 < md2:
                        return False
                return True

            t_sand = int(self.state.T_SAND)
            t_board = int(self.state.T_BOARDWALK)
            t_water = int(self.state.T_WATER)
            t_highway = int(self.state.T_HIGHWAY)

            # Beach amusements: place a couple of abandoned toys near the boardwalk.
            beach_cand: list[tuple[int, int]] = []
            for y in range(2, int(self.state.CHUNK_SIZE) - 2):
                for x in range(2, int(self.state.CHUNK_SIZE) - 2):
                    t = int(tiles[idx(int(x), int(y))])
                    if t not in (t_sand, t_board):
                        continue
                    near = False
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        nt = int(tiles[idx(int(nx), int(ny))])
                        if nt in (t_board, t_water):
                            near = True
                            break
                    if near:
                        beach_cand.append((int(x), int(y)))
            if len(beach_cand) >= 36:
                rng.shuffle(beach_cand)
                want = int(rng.randint(1, 3))
                toy_pool = ["toy_seesaw", "toy_carousel", "toy_swing"]
                for x, y in beach_cand:
                    if want <= 0:
                        break
                    wx = int(base_tx + int(x))
                    wy = int(base_ty + int(y))
                    px = (float(wx) + 0.5) * float(self.state.TILE_SIZE)
                    py = (float(wy) + 0.5) * float(self.state.TILE_SIZE)
                    if not far_from_existing(px, py, min_d=30.0):
                        continue
                    props.append(
                        HardcoreSurvivalState._WorldProp(
                            pos=pygame.Vector2(px, py),
                            prop_id=str(rng.choice(toy_pool)),
                            variant=int(rng.randint(0, 7)),
                            dir=str(rng.choice(["left", "right", "up", "down"])),
                        )
                    )
                    want -= 1

            # Abandoned highways: occasional billboards near the shoulder.
            bill_cand: list[tuple[int, int]] = []
            for y in range(2, int(self.state.CHUNK_SIZE) - 2):
                for x in range(2, int(self.state.CHUNK_SIZE) - 2):
                    t = int(tiles[idx(int(x), int(y))])
                    if t in (t_highway, int(self.state.T_WALL), int(self.state.T_DOOR)):
                        continue
                    if not (
                        int(tiles[idx(int(x) + 1, int(y))]) == t_highway
                        or int(tiles[idx(int(x) - 1, int(y))]) == t_highway
                        or int(tiles[idx(int(x), int(y) + 1)]) == t_highway
                        or int(tiles[idx(int(x), int(y) - 1)]) == t_highway
                    ):
                        continue
                    tdef = self.state._TILES.get(int(t))
                    if tdef is not None and bool(getattr(tdef, "solid", False)):
                        continue
                    bill_cand.append((int(x), int(y)))

            if bill_cand and rng.random() < 0.55:
                rng.shuffle(bill_cand)
                want = 1 if len(bill_cand) >= 18 else 0
                for x, y in bill_cand:
                    if want <= 0:
                        break
                    wx = int(base_tx + int(x))
                    wy = int(base_ty + int(y))
                    px = (float(wx) + 0.5) * float(self.state.TILE_SIZE)
                    py = (float(wy) + 0.5) * float(self.state.TILE_SIZE)
                    if not far_from_existing(px, py, min_d=26.0):
                        continue
                    props.append(
                        HardcoreSurvivalState._WorldProp(
                            pos=pygame.Vector2(px, py),
                            prop_id="billboard",
                            variant=int(rng.randint(0, 7)),
                            dir=str(rng.choice(["left", "right", "up", "down"])),
                        )
                    )
                    want -= 1

        def _stamp_city_decor(
            self,
            tiles: list[int],
            props: list["HardcoreSurvivalState._WorldProp"],
            cx: int,
            cy: int,
            *,
            rng: random.Random,
        ) -> None:
            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)

            # Billboards near streets (non-blocking decoration).
            bill_cand: list[tuple[int, int]] = []
            for y in range(2, int(self.state.CHUNK_SIZE) - 2):
                for x in range(2, int(self.state.CHUNK_SIZE) - 2):
                    t = int(tiles[idx(int(x), int(y))])
                    if t not in (
                        int(self.state.T_SIDEWALK),
                        int(self.state.T_PAVEMENT),
                        int(self.state.T_CONCRETE),
                        int(self.state.T_BRICK),
                        int(self.state.T_PARKING),
                    ):
                        continue
                    near_road = False
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if int(tiles[idx(int(x + dx), int(y + dy))]) == int(self.state.T_ROAD):
                            near_road = True
                            break
                    if not near_road:
                        continue
                    bill_cand.append((int(x), int(y)))
            rng.shuffle(bill_cand)

            want_bill = 0
            if bill_cand and rng.random() < 0.55:
                want_bill = 1 if rng.random() < 0.75 else 2

            def far_from_existing(px: float, py: float, *, min_d: float) -> bool:
                md2 = float(min_d) * float(min_d)
                for pr in props:
                    dx = float(pr.pos.x) - float(px)
                    dy = float(pr.pos.y) - float(py)
                    if dx * dx + dy * dy < md2:
                        return False
                return True

            for x, y in bill_cand:
                if want_bill <= 0:
                    break
                wx = int(base_tx + int(x))
                wy = int(base_ty + int(y))
                px = (float(wx) + 0.5) * float(self.state.TILE_SIZE)
                py = (float(wy) + 0.5) * float(self.state.TILE_SIZE)
                if not far_from_existing(px, py, min_d=26.0):
                    continue
                props.append(
                    HardcoreSurvivalState._WorldProp(
                        pos=pygame.Vector2(px, py),
                        prop_id="billboard",
                        variant=int(rng.randint(0, 7)),
                        dir=str(rng.choice(["left", "right", "up", "down"])),
                    )
                )
                want_bill -= 1

            # Park toys: only inside 3x3 grass/forest clusters.
            park_cand: list[tuple[int, int]] = []
            for y in range(3, int(self.state.CHUNK_SIZE) - 3):
                for x in range(3, int(self.state.CHUNK_SIZE) - 3):
                    t0 = int(tiles[idx(int(x), int(y))])
                    if t0 not in (int(self.state.T_GRASS), int(self.state.T_FOREST)):
                        continue
                    ok = True
                    for yy in range(int(y) - 1, int(y) + 2):
                        for xx in range(int(x) - 1, int(x) + 2):
                            tt = int(tiles[idx(int(xx), int(yy))])
                            if tt not in (int(self.state.T_GRASS), int(self.state.T_FOREST)):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        park_cand.append((int(x), int(y)))
            rng.shuffle(park_cand)

            want_toys = 0
            if len(park_cand) >= 70:
                want_toys = int(rng.randint(1, 3))
            elif len(park_cand) >= 36:
                want_toys = int(rng.randint(0, 2))
            elif len(park_cand) >= 20:
                want_toys = int(rng.randint(0, 1))

            toy_pool = ["toy_seesaw", "toy_carousel", "toy_swing"]
            for x, y in park_cand:
                if want_toys <= 0:
                    break
                wx = int(base_tx + int(x))
                wy = int(base_ty + int(y))
                px = (float(wx) + 0.5) * float(self.state.TILE_SIZE)
                py = (float(wy) + 0.5) * float(self.state.TILE_SIZE)
                if not far_from_existing(px, py, min_d=30.0):
                    continue
                props.append(
                    HardcoreSurvivalState._WorldProp(
                        pos=pygame.Vector2(px, py),
                        prop_id=str(rng.choice(toy_pool)),
                        variant=int(rng.randint(0, 7)),
                        dir=str(rng.choice(["left", "right", "up", "down"])),
                    )
                )
                want_toys -= 1

        def _stamp_buildings(
            self,
            tiles: list[int],
            items: list["HardcoreSurvivalState._WorldItem"],
            buildings: list[tuple[int, int, int, int, int, int]],
            special_buildings: list["HardcoreSurvivalState._SpecialBuilding"],  
            props: list["HardcoreSurvivalState._WorldProp"],
            cx: int,
            cy: int,
            rng: random.Random,
            town_kind: str,
            reserved: set[tuple[int, int]] | None = None,
        ) -> None:
            tries = 0
            placed = 0
            want = 1
            max_tries = 120 if town_kind in ("大型超市", "大型监狱", "高层住宅大") else 60

            def idx(x: int, y: int) -> int:
                return y * self.state.CHUNK_SIZE + x

            base_tx = int(cx) * int(self.state.CHUNK_SIZE)
            base_ty = int(cy) * int(self.state.CHUNK_SIZE)
            allow_overwrite_city_street = town_kind in ("大型超市", "大型监狱", "高层住宅大")

            while placed < want and tries < max_tries:
                tries += 1
                building_floors = 0
                if town_kind == "大型超市":
                    w = rng.randint(20, 28)
                    h = rng.randint(14, 22)
                elif town_kind == "超市":
                    w = rng.randint(13, 19)
                    h = rng.randint(10, 16)
                elif town_kind == "新华书店":
                    w = rng.randint(12, 18)
                    h = rng.randint(10, 15)
                elif town_kind == "医院":
                    w = rng.randint(14, 20)
                    h = rng.randint(12, 18)
                elif town_kind == "学校":
                    w = rng.randint(16, 23)
                    h = rng.randint(13, 19)
                elif town_kind == "高层住宅大":
                    w = rng.randint(16, 24)
                    h = rng.randint(16, 24)
                elif town_kind == "高层住宅":
                    w = rng.randint(10, 15)
                    h = rng.randint(10, 16)
                elif town_kind == "大型监狱":
                    w = rng.randint(22, 28)
                    h = rng.randint(20, 28)
                elif town_kind == "监狱":
                    w = rng.randint(14, 20)
                    h = rng.randint(13, 19)
                elif town_kind == "中式建筑":
                    w = rng.randint(9, 15)
                    h = rng.randint(8, 13)
                elif town_kind == "住宅":
                    w = rng.randint(6, 10)
                    h = rng.randint(6, 10)
                else:
                    w = rng.randint(8, 12)
                    h = rng.randint(7, 12)
                x0 = rng.randint(2, self.state.CHUNK_SIZE - w - 2)
                y0 = rng.randint(2, self.state.CHUNK_SIZE - h - 2)

                blocked = False
                for y in range(y0, y0 + h):
                    for x in range(x0, x0 + w):
                        if reserved is not None and (int(x), int(y)) in reserved:
                            blocked = True
                            break
                        t = tiles[idx(x, y)]
                        if int(t) == int(self.state.T_ROAD):
                            wx = int(base_tx + int(x))
                            wy = int(base_ty + int(y))
                            is_main = bool(self.is_road(wx, wy))
                            is_city_minor = bool(self.is_city_street(wx, wy)) and not is_main
                            if allow_overwrite_city_street and is_city_minor:
                                continue
                            blocked = True
                            break
                        if t not in (
                            self.state.T_GRASS,
                            self.state.T_FOREST,
                            self.state.T_WATER,
                            self.state.T_PAVEMENT,
                            self.state.T_PARKING,
                            self.state.T_COURT,
                            self.state.T_SIDEWALK,
                            self.state.T_BRICK,
                            self.state.T_CONCRETE,
                        ):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue

                # Always use south-facing entrances so the single "front facade"
                # (bottom/S) matches the actual doorway + collision.     
                door_side = "S"
                door_tiles: list[tuple[int, int]] = []
                if door_side == "N":
                    dx = rng.randint(x0 + 1, x0 + w - 3)
                    dy = y0
                    door_tiles = [(dx, dy), (dx + 1, dy)]
                elif door_side == "S":
                    dy = y0 + h - 1

                    def approach_clear(ddx0: int, ddy0: int) -> bool:
                        for ddx, ddy in ((ddx0, ddy0), (ddx0 + 1, ddy0)):
                            for step in range(1, 9):
                                px = int(ddx)
                                py = int(ddy + step)
                                if not (0 <= px < int(self.state.CHUNK_SIZE) and 0 <= py < int(self.state.CHUNK_SIZE)):
                                    break
                                tcur = int(tiles[idx(int(px), int(py))])
                                if tcur in (
                                    int(self.state.T_WALL),
                                    int(self.state.T_FLOOR),
                                    int(self.state.T_DOOR),
                                ):
                                    return False
                        return True

                    door_x_candidates = [int(x) for x in range(int(x0 + 1), int(x0 + w - 2))]
                    rng.shuffle(door_x_candidates)
                    door_x_candidates = [int(x) for x in door_x_candidates if approach_clear(int(x), int(dy))]
                    if not door_x_candidates:
                        continue
                    dx = int(rng.choice(door_x_candidates))
                    door_tiles = [(dx, dy), (dx + 1, dy)]
                elif door_side == "W":
                    dx = x0
                    dy = rng.randint(y0 + 1, y0 + h - 3)
                    door_tiles = [(dx, dy), (dx, dy + 1)]
                else:
                    dx = x0 + w - 1
                    dy = rng.randint(y0 + 1, y0 + h - 3)
                    door_tiles = [(dx, dy), (dx, dy + 1)]

                for y in range(y0, y0 + h):
                    for x in range(x0, x0 + w):
                        border = x in (x0, x0 + w - 1) or y in (y0, y0 + h - 1)
                        tiles[idx(x, y)] = self.state.T_WALL if border else self.state.T_FLOOR

                for dx, dy in door_tiles:
                    tiles[idx(dx, dy)] = self.state.T_DOOR

                # Carve a small approach path outside the door so buildings are
                # always reachable even if terrain noise produced water nearby.
                if door_side == "N":
                    ox, oy = 0, -1
                elif door_side == "S":
                    ox, oy = 0, 1
                elif door_side == "W":
                    ox, oy = -1, 0
                else:
                    ox, oy = 1, 0
                path_tile = int(self.state.T_SIDEWALK) if bool(self._is_city_chunk(int(cx), int(cy))) else int(self.state.T_ROAD)
                for dx, dy in door_tiles:
                    for step in range(1, 9):
                        px = dx + ox * step
                        py = dy + oy * step
                        if not (0 <= px < self.state.CHUNK_SIZE and 0 <= py < self.state.CHUNK_SIZE):
                            break
                        tcur = int(tiles[idx(int(px), int(py))])
                        if tcur in (int(self.state.T_WALL), int(self.state.T_FLOOR), int(self.state.T_DOOR)):
                            break
                        tiles[idx(px, py)] = path_tile
                        if reserved is not None:
                            reserved.add((int(px), int(py)))

                # Exterior signage so building type reads from outside.
                sign_id = "sign_home"
                if town_kind == "医院":
                    sign_id = "sign_hospital"
                elif town_kind == "超市":
                    sign_id = "sign_shop"
                elif town_kind == "大型超市":
                    sign_id = "sign_shop_big"
                elif town_kind == "新华书店":
                    sign_id = "sign_bookstore"
                elif town_kind == "学校":
                    sign_id = "sign_school"
                elif town_kind in ("监狱", "大型监狱"):
                    sign_id = "sign_prison"
                elif town_kind in ("高层住宅", "高层住宅大"):
                    sign_id = "sign_highrise"
                elif town_kind == "中式建筑":
                    sign_id = "sign_chinese"

                door_mx = int(round(sum(int(p[0]) for p in door_tiles) / max(1, len(door_tiles))))
                door_my = int(round(sum(int(p[1]) for p in door_tiles) / max(1, len(door_tiles))))
                out_x = door_mx + int(ox)
                out_y = door_my + int(oy)
                # Avoid placing a sign directly in front of the doorway (looks like it blocks the entrance).
                sign_candidates: list[tuple[int, int]] = []
                xs0 = [p[0] for p in door_tiles]
                ys0 = [p[1] for p in door_tiles]
                dx0 = int(min(xs0)) if xs0 else int(door_mx)
                dx1 = int(max(xs0)) if xs0 else int(door_mx)
                dy0 = int(max(ys0)) if ys0 else int(door_my)

                # Prefer placing signs on the ground outside the entrance (never on wall tiles),
                # so they don't look like they're floating on the facade.

                if door_side in ("N", "S"):
                    sign_candidates += [
                        (int(out_x) - 1, int(out_y)),
                        (int(out_x) + 1, int(out_y)),
                        (int(out_x) - 2, int(out_y)),
                        (int(out_x) + 2, int(out_y)),
                    ]
                else:
                    sign_candidates += [
                        (int(out_x), int(out_y) - 1),
                        (int(out_x), int(out_y) + 1),
                        (int(out_x), int(out_y) - 2),
                        (int(out_x), int(out_y) + 2),
                    ]

                for sx, sy in sign_candidates:
                    if not (0 <= int(sx) < int(self.state.CHUNK_SIZE) and 0 <= int(sy) < int(self.state.CHUNK_SIZE)):
                        continue
                    tdef = self.state._TILES.get(int(tiles[idx(int(sx), int(sy))]))
                    if tdef is not None and bool(getattr(tdef, "solid", False)):
                        continue
                    wx = int(base_tx + int(sx))
                    wy = int(base_ty + int(sy))
                    px = (float(wx) + 0.5) * float(self.state.TILE_SIZE)
                    py = (float(wy) + 0.5) * float(self.state.TILE_SIZE)
                    props.append(
                        HardcoreSurvivalState._WorldProp(
                            pos=pygame.Vector2(px, py),
                            prop_id=str(sign_id),
                            variant=int(rng.randint(0, 7)),
                            dir=str({"N": "up", "S": "down", "W": "left", "E": "right"}.get(str(door_side), "down")),
                        )
                    )
                    break

                def near_door(x: int, y: int) -> bool:
                    for ddx, ddy in door_tiles:
                        if abs(int(x) - int(ddx)) <= 1 and abs(int(y) - int(ddy)) <= 1:
                            return True
                    return False

                # Interior layout (rooms + props). Keep a clear corridor so the
                # entrance always feels navigable.
                in_left = x0 + 1
                in_right = x0 + w - 2
                in_top = y0 + 1
                in_bottom = y0 + h - 2

                door_cx = sum(int(p[0]) for p in door_tiles) / max(1, len(door_tiles))
                door_cy = sum(int(p[1]) for p in door_tiles) / max(1, len(door_tiles))

                corridor: set[tuple[int, int]] = set()

                def mark_corridor(x: int, y: int) -> None:
                    if in_left <= int(x) <= in_right and in_top <= int(y) <= in_bottom:
                        corridor.add((int(x), int(y)))

                def set_interior(x: int, y: int, tile_id: int) -> None:
                    if not (in_left <= int(x) <= in_right and in_top <= int(y) <= in_bottom):
                        return
                    if near_door(int(x), int(y)):
                        return
                    if (int(x), int(y)) in corridor:
                        return
                    tiles[idx(int(x), int(y))] = int(tile_id)

                def clear_interior(x: int, y: int) -> None:
                    if not (in_left <= int(x) <= in_right and in_top <= int(y) <= in_bottom):
                        return
                    tiles[idx(int(x), int(y))] = self.state.T_FLOOR

                # Main corridor based on entrance side.
                if door_side in ("N", "S"):
                    cx_line = int(clamp(int(round(door_cx)), in_left, in_right))
                    for y in range(in_top, in_bottom + 1):
                        mark_corridor(cx_line, y)
                        mark_corridor(cx_line + 1, y)
                else:
                    cy_line = int(clamp(int(round(door_cy)), in_top, in_bottom))
                    for x in range(in_left, in_right + 1):
                        mark_corridor(x, cy_line)
                        mark_corridor(x, cy_line + 1)

                # Ensure the immediate area behind the door is clear.
                if door_side == "N":
                    for dx, _dy in door_tiles:
                        mark_corridor(dx, y0 + 1)
                        mark_corridor(dx, y0 + 2)
                elif door_side == "S":
                    for dx, _dy in door_tiles:
                        mark_corridor(dx, y0 + h - 2)
                        mark_corridor(dx, y0 + h - 3)
                elif door_side == "W":
                    for _dx, dy in door_tiles:
                        mark_corridor(x0 + 1, dy)
                        mark_corridor(x0 + 2, dy)
                else:
                    for _dx, dy in door_tiles:
                        mark_corridor(x0 + w - 2, dy)
                        mark_corridor(x0 + w - 3, dy)

                def build_room(rx: int, ry: int, rw: int, rh: int) -> tuple[int, int] | None:
                    # rx/ry are interior-floor top-left; walls wrap around.
                    rw = int(max(2, rw))
                    rh = int(max(2, rh))
                    rx = int(clamp(rx, in_left + 1, in_right - rw))
                    ry = int(clamp(ry, in_top + 1, in_bottom - rh))

                    for y in range(ry, ry + rh):
                        for x in range(rx, rx + rw):
                            clear_interior(x, y)

                    # Walls (only inside the building border).
                    for x in range(rx - 1, rx + rw + 1):
                        set_interior(x, ry - 1, self.state.T_WALL)
                        set_interior(x, ry + rh, self.state.T_WALL)
                    for y in range(ry - 1, ry + rh + 1):
                        set_interior(rx - 1, y, self.state.T_WALL)
                        set_interior(rx + rw, y, self.state.T_WALL)

                    # Door on the side facing the center.
                    cx = x0 + w / 2.0
                    cy = y0 + h / 2.0
                    side: str
                    if abs((rx + rw / 2.0) - cx) >= abs((ry + rh / 2.0) - cy):
                        side = "W" if (rx + rw / 2.0) > cx else "E"
                    else:
                        side = "N" if (ry + rh / 2.0) > cy else "S"

                    if side == "N":
                        dx = rx + rw // 2
                        dy = ry - 1
                    elif side == "S":
                        dx = rx + rw // 2
                        dy = ry + rh
                    elif side == "W":
                        dx = rx - 1
                        dy = ry + rh // 2
                    else:
                        dx = rx + rw
                        dy = ry + rh // 2

                    if in_left <= dx <= in_right and in_top <= dy <= in_bottom and not near_door(dx, dy):
                        tiles[idx(dx, dy)] = self.state.T_DOOR
                        # Clear a tiny landing outside the door.
                        if side == "N":
                            clear_interior(dx, dy + 1)
                        elif side == "S":
                            clear_interior(dx, dy - 1)
                        elif side == "W":
                            clear_interior(dx + 1, dy)
                        else:
                            clear_interior(dx - 1, dy)
                        mark_corridor(dx, dy)
                        return dx, dy
                    return None

                if town_kind == "新华书店":
                    # Bookstore: shelves in aisles + a small reading area.
                    backroom: tuple[int, int, int, int] | None = None
                    if w >= 12 and h >= 9 and rng.random() < 0.75:
                        rw = min(4, int(w) - 6)
                        rh = min(3, int(h) - 5)
                        rx = x0 + 2 if door_side in ("N", "S") else (x0 + w - rw - 2)
                        ry = y0 + h - rh - 2 if door_side == "N" else (y0 + 2)
                        backroom = (int(rx), int(ry), int(rw), int(rh))
                        build_room(int(rx), int(ry), int(rw), int(rh))

                    # Shelf islands (leave walkable aisles).
                    for y in range(y0 + 3, y0 + h - 3, 3):
                        for x in range(x0 + 2, x0 + w - 3, 4):
                            if backroom is not None:
                                bx, by, bw, bh = backroom
                                if (bx - 1) <= int(x) <= (bx + bw) and (by - 1) <= int(y) <= (by + bh):
                                    continue
                            if tiles[idx(int(x), int(y))] == self.state.T_FLOOR:
                                set_interior(int(x), int(y), self.state.T_SHELF)
                            if tiles[idx(int(x) + 1, int(y))] == self.state.T_FLOOR:
                                set_interior(int(x) + 1, int(y), self.state.T_SHELF)

                    # Reading tables nearer the entrance.
                    if door_side in ("N", "S"):
                        y = y0 + 3 if door_side == "N" else y0 + h - 4
                        for x in range(x0 + 3, x0 + w - 3, 5):
                            if tiles[idx(int(x), int(y))] == self.state.T_FLOOR:
                                set_interior(int(x), int(y), self.state.T_TABLE)

                    # Checkout counter near entrance.
                    if door_side in ("N", "S"):
                        y = y0 + 2 if door_side == "N" else y0 + h - 3
                        x0c = int(round(door_cx)) - 1
                        x1c = int(round(door_cx)) + 1
                        for x in range(x0c, x1c + 1):
                            if (int(x), int(y)) in corridor or near_door(int(x), int(y)):
                                continue
                            set_interior(int(x), int(y), self.state.T_TABLE)
                    else:
                        x = x0 + 2 if door_side == "W" else x0 + w - 3
                        y0c = int(round(door_cy)) - 1
                        y1c = int(round(door_cy)) + 1
                        for y in range(y0c, y1c + 1):
                            if (int(x), int(y)) in corridor or near_door(int(x), int(y)):
                                continue
                            set_interior(int(x), int(y), self.state.T_TABLE)

                    # Backroom storage shelves.
                    if backroom is not None:
                        bx, by, bw, bh = backroom
                        for x in range(bx, bx + bw):
                            if (x - bx) % 2 == 0:
                                for y in range(by, by + bh):
                                    if tiles[idx(x, y)] == self.state.T_FLOOR:
                                        tiles[idx(x, y)] = self.state.T_SHELF

                elif town_kind in ("超市", "大型超市"):
                    # Optional back room.
                    backroom: tuple[int, int, int, int] | None = None     
                    if w >= 12 and h >= 9:
                        rw = min(5, int(w) - 6)
                        rh = min(4, int(h) - 5)
                        rx = x0 + 2 if door_side in ("N", "S") else (x0 + w - rw - 2)
                        ry = y0 + h - rh - 2 if door_side == "N" else (y0 + 2)
                        backroom = (int(rx), int(ry), int(rw), int(rh))
                        build_room(int(rx), int(ry), int(rw), int(rh))

                    cross_y = int(clamp(int(y0 + h // 2), in_top + 1, in_bottom - 1))
                    for x in range(x0 + 2, x0 + w - 2, 3):
                        for y in range(y0 + 2, y0 + h - 2):
                            if (int(x), int(y)) in corridor:
                                continue
                            if abs(int(y) - int(cross_y)) <= 0:
                                continue
                            if near_door(int(x), int(y)):
                                continue
                            if backroom is not None:
                                bx, by, bw, bh = backroom
                                if (bx - 1) <= int(x) <= (bx + bw) and (by - 1) <= int(y) <= (by + bh):
                                    continue
                            tiles[idx(int(x), int(y))] = self.state.T_SHELF

                    # Checkout counter near entrance.
                    if door_side in ("N", "S"):
                        y = y0 + 2 if door_side == "N" else y0 + h - 3
                        x0c = int(round(door_cx)) - 2
                        x1c = int(round(door_cx)) + 2
                        for x in range(x0c, x1c + 1):
                            if (int(x), int(y)) in corridor or near_door(int(x), int(y)):
                                continue
                            set_interior(int(x), int(y), self.state.T_TABLE)
                    else:
                        x = x0 + 2 if door_side == "W" else x0 + w - 3
                        y0c = int(round(door_cy)) - 2
                        y1c = int(round(door_cy)) + 2
                        for y in range(y0c, y1c + 1):
                            if (int(x), int(y)) in corridor or near_door(int(x), int(y)):
                                continue
                            set_interior(int(x), int(y), self.state.T_TABLE)

                    # Backroom storage shelves.
                    if backroom is not None:
                        bx, by, bw, bh = backroom
                        for x in range(bx, bx + bw):
                            if (x - bx) % 2 == 0:
                                for y in range(by, by + bh):
                                    if tiles[idx(x, y)] == self.state.T_FLOOR:
                                        tiles[idx(x, y)] = self.state.T_SHELF

                elif town_kind == "中式建筑":
                    # Chinese-style small house: bed + table + side storage.
                    bx = in_left + 1
                    if door_side in ("N", "S"):
                        if int(round(door_cx)) <= int((in_left + in_right) // 2):
                            bx = in_right - 2
                        else:
                            bx = in_left + 1
                    by = in_top + 1
                    if tiles[idx(bx, by)] == self.state.T_FLOOR and tiles[idx(bx + 1, by)] == self.state.T_FLOOR:
                        tiles[idx(bx, by)] = self.state.T_BED
                        tiles[idx(bx + 1, by)] = self.state.T_BED

                    tx0c = int(clamp(int(round((in_left + in_right) / 2.0)), in_left + 1, in_right - 1))
                    ty0c = int(clamp(int(round((in_top + in_bottom) / 2.0)), in_top + 1, in_bottom - 1))
                    if tiles[idx(tx0c, ty0c)] == self.state.T_FLOOR:
                        tiles[idx(tx0c, ty0c)] = self.state.T_TABLE

                    shelf_x = in_left if bx > int((in_left + in_right) // 2) else in_right
                    for y in range(in_top + 2, in_bottom - 1, 2):
                        set_interior(int(shelf_x), int(y), self.state.T_SHELF)

                elif town_kind == "医院":
                    # Reception + ward split.
                    if door_side in ("N", "S"):
                        wy = y0 + 4 if door_side == "N" else y0 + h - 5   
                        if in_top + 1 <= wy <= in_bottom - 1:
                            for x in range(in_left, in_right + 1):
                                if not near_door(x, wy):
                                    tiles[idx(x, wy)] = self.state.T_WALL
                            dx = int(clamp(int(round(door_cx)), in_left + 1, in_right - 1))
                            tiles[idx(dx, wy)] = self.state.T_DOOR
                            mark_corridor(dx, wy)

                        # Reception desks.
                        ry0 = in_top if door_side == "N" else (wy + 1)
                        ry1 = wy - 1 if door_side == "N" else in_bottom
                        desk_y = ry0 + 1 if door_side == "N" else ry1 - 1
                        for x in range(in_left + 1, in_right):
                            if (x, desk_y) in corridor or near_door(x, desk_y):
                                continue
                            if (x - (in_left + 1)) % 2 == 0:
                                set_interior(x, desk_y, self.state.T_TABLE)
                        # Ward beds (opposite side).
                        wy0 = wy + 1 if door_side == "N" else in_top
                        wy1 = in_bottom if door_side == "N" else wy - 1
                        for y in range(wy0 + 1, wy1, 2):
                            for x in (in_left + 1, in_right - 2):
                                if (x, y) in corridor or near_door(x, y):
                                    continue
                                if tiles[idx(x, y)] == self.state.T_FLOOR and tiles[idx(x + 1, y)] == self.state.T_FLOOR:
                                    tiles[idx(x, y)] = self.state.T_BED
                                    tiles[idx(x + 1, y)] = self.state.T_BED
                        # Supply closet.
                        if w >= 11 and h >= 9:
                            cxr = x0 + w - 6 if door_side in ("N", "S") else x0 + 2
                            cyr = y0 + h - 5 if door_side == "N" else (y0 + 2)
                            room = build_room(int(cxr), int(cyr), 3, 2)
                            if room is not None:
                                rx, ry, rw, rh = int(cxr), int(cyr), 3, 2
                                for x in range(rx, rx + rw):
                                    for y in range(ry, ry + rh):
                                        if tiles[idx(x, y)] == self.state.T_FLOOR:
                                            tiles[idx(x, y)] = self.state.T_SHELF
                    else:
                        wx = x0 + 4 if door_side == "W" else x0 + w - 5
                        if in_left + 1 <= wx <= in_right - 1:
                            for y in range(in_top, in_bottom + 1):
                                if not near_door(wx, y):
                                    tiles[idx(wx, y)] = self.state.T_WALL
                            dy = int(clamp(int(round(door_cy)), in_top + 1, in_bottom - 1))
                            tiles[idx(wx, dy)] = self.state.T_DOOR
                            mark_corridor(wx, dy)

                        # Reception desks.
                        rx0 = in_left if door_side == "W" else (wx + 1)
                        rx1 = wx - 1 if door_side == "W" else in_right
                        desk_x = rx0 + 1 if door_side == "W" else rx1 - 1
                        for y in range(in_top + 1, in_bottom):
                            if (desk_x, y) in corridor or near_door(desk_x, y):
                                continue
                            if (y - (in_top + 1)) % 2 == 0:
                                set_interior(desk_x, y, self.state.T_TABLE)

                        # Ward beds.
                        wx0 = wx + 1 if door_side == "W" else in_left
                        wx1 = in_right if door_side == "W" else wx - 1
                        for x in range(wx0 + 1, wx1, 3):
                            for y in (in_top + 1, in_bottom - 2):
                                if (x, y) in corridor or near_door(x, y):  
                                    continue
                                if tiles[idx(x, y)] == self.state.T_FLOOR and tiles[idx(x + 1, y)] == self.state.T_FLOOR:
                                    tiles[idx(x, y)] = self.state.T_BED    
                                    tiles[idx(x + 1, y)] = self.state.T_BED

                elif town_kind == "学校":
                    # School: some schools are multi-floor and become a portal to a
                    # separate interior scene (world-map interior is sealed).
                    is_city = bool(self._is_city_chunk(int(cx), int(cy)))
                    floors = int(rng.randint(3, 6)) if is_city else int(rng.randint(2, 4))
                    if (not is_city) and rng.random() < 0.35:
                        floors = 1

                    if floors > 1:
                        for y in range(y0 + 1, y0 + h - 1):
                            for x in range(x0 + 1, x0 + w - 1):
                                if tiles[idx(x, y)] != self.state.T_DOOR:
                                    tiles[idx(x, y)] = self.state.T_WALL

                        tx0w = cx * self.state.CHUNK_SIZE + x0
                        ty0w = cy * self.state.CHUNK_SIZE + y0
                        doors_w = tuple(
                            (cx * self.state.CHUNK_SIZE + int(dx), cy * self.state.CHUNK_SIZE + int(dy))
                            for dx, dy in door_tiles
                        )
                        special_buildings.append(
                            HardcoreSurvivalState._SpecialBuilding(
                                kind="school",
                                name="学校",
                                tx0=int(tx0w),
                                ty0=int(ty0w),
                                w=int(w),
                                h=int(h),
                                door_tiles=doors_w,
                                floors=int(floors),
                            )
                        )
                    else:
                        # Single-floor school: keep corridor clear, add desks + lockers.
                        # Lockers along one side.
                        if door_side in ("N", "S"):
                            lx = in_left + 1
                            for y in range(in_top + 1, in_bottom):
                                if (lx, y) in corridor or near_door(lx, y):
                                    continue
                                if (y - (in_top + 1)) % 2 == 0:
                                    set_interior(lx, y, self.state.T_SHELF)
                        else:
                            ly = in_top + 1
                            for x in range(in_left + 1, in_right):
                                if (x, ly) in corridor or near_door(x, ly):
                                    continue
                                if (x - (in_left + 1)) % 2 == 0:
                                    set_interior(x, ly, self.state.T_SHELF)

                        # Desks in rows (reads as classrooms).
                        for y in range(in_top + 2, in_bottom, 2):
                            for x in range(in_left + 2, in_right, 3):
                                set_interior(x, y, self.state.T_TABLE)

                        # Small office/storage room.
                        if w >= 12 and h >= 9:
                            rx = x0 + w - 6 if door_side in ("N", "S") else x0 + 2
                            ry = y0 + 2 if door_side in ("N", "S") else y0 + h - 5
                            room = build_room(int(rx), int(ry), 3, 2)
                            if room is not None:
                                for x in range(int(rx), int(rx) + 3):
                                    for y in range(int(ry), int(ry) + 2):
                                        if tiles[idx(x, y)] == self.state.T_FLOOR and ((x + y) % 2 == 0):
                                            tiles[idx(x, y)] = self.state.T_SHELF

                elif town_kind in ("高层住宅", "高层住宅大"):
                    # High-rise: act as a portal to a separate interior (world-map interior is sealed).
                    for y in range(y0 + 1, y0 + h - 1):
                        for x in range(x0 + 1, x0 + w - 1):
                            if tiles[idx(x, y)] != self.state.T_DOOR:
                                tiles[idx(x, y)] = self.state.T_WALL

                    # Leave a small lobby behind the entrance so you can walk
                    # inside instead of hitting a wall at the door.
                    if False and door_side == "S" and door_tiles:
                        door_xs = [int(p[0]) for p in door_tiles]
                        door_cx = int(round(sum(door_xs) / max(1, len(door_xs))))
                        lobby_w = int(clamp(8 if int(w) >= 14 else 6, 4, int(w) - 4))
                        lobby_h = int(clamp(6 if int(h) >= 14 else 5, 4, int(h) - 4))
                        lobby_x0 = int(
                            clamp(
                                int(door_cx - lobby_w // 2),
                                int(x0 + 2),
                                int(x0 + w - 2 - lobby_w),
                            )
                        )
                        lobby_y1 = int(y0 + h - 2)  # just inside the south wall
                        lobby_y0 = int(clamp(int(lobby_y1 - lobby_h + 1), int(y0 + 2), int(y0 + h - 2)))
                        for yy in range(int(lobby_y0), int(lobby_y1) + 1):
                            for xx in range(int(lobby_x0), int(lobby_x0 + lobby_w)):
                                tiles[idx(int(xx), int(yy))] = int(self.state.T_FLOOR)

                        # Ensure the tiles immediately behind the door are floor.
                        for ddx, ddy in door_tiles:
                            ix = int(ddx)
                            iy = int(ddy) - 1
                            if int(x0 + 1) <= ix <= int(x0 + w - 2) and int(y0 + 1) <= iy <= int(y0 + h - 2):
                                tiles[idx(int(ix), int(iy))] = int(self.state.T_FLOOR)

                    tx0w = cx * self.state.CHUNK_SIZE + x0
                    ty0w = cy * self.state.CHUNK_SIZE + y0
                    doors_w = tuple((cx * self.state.CHUNK_SIZE + int(dx), cy * self.state.CHUNK_SIZE + int(dy)) for dx, dy in door_tiles)
                    if town_kind == "高层住宅大":
                        floors = int(rng.randint(12, 20))
                    else:
                        floors = int(rng.randint(6, 12))
                    building_floors = int(floors)
                    special_buildings.append(
                        HardcoreSurvivalState._SpecialBuilding(
                            kind="highrise",
                            name="高层住宅",
                            tx0=int(tx0w),
                            ty0=int(ty0w),
                            w=int(w),
                            h=int(h),
                            door_tiles=doors_w,
                            floors=floors,
                        )
                    )

                else:
                    # Prison: split into cell block + common area.
                    if w >= h:
                        split_x = int(clamp(int(x0 + w // 2), in_left + 2, in_right - 2))
                        for y in range(in_top, in_bottom + 1):
                            if near_door(split_x, y):
                                continue
                            tiles[idx(split_x, y)] = self.state.T_WALL

                        cell_n = 3 if (in_bottom - in_top + 1) >= 7 else 2
                        total_h = int(in_bottom - in_top + 1)
                        # Horizontal separators for cells.
                        for i in range(1, cell_n):
                            by = in_top + (i * total_h) // cell_n
                            if not (in_top + 1 <= by <= in_bottom - 1):
                                continue
                            for x in range(in_left, split_x + 1):
                                if near_door(x, by):
                                    continue
                                tiles[idx(x, by)] = self.state.T_WALL

                        for i in range(cell_n):
                            y0c = in_top + (i * total_h) // cell_n
                            y1c = in_top + ((i + 1) * total_h) // cell_n - 1
                            dy = int(clamp((y0c + y1c) // 2, in_top + 1, in_bottom - 1))
                            if not near_door(split_x, dy):
                                tiles[idx(split_x, dy)] = self.state.T_DOOR
                                clear_interior(split_x + 1, dy)
                                mark_corridor(split_x, dy)

                            bx = in_left + 1
                            by = int(clamp(dy, in_top + 1, in_bottom - 1))
                            if tiles[idx(bx, by)] == self.state.T_FLOOR and tiles[idx(bx + 1, by)] == self.state.T_FLOOR:
                                tiles[idx(bx, by)] = self.state.T_BED
                                tiles[idx(bx + 1, by)] = self.state.T_BED
                            if rng.random() < 0.55:
                                tx = in_left + 1
                                ty = int(clamp(by - 1, in_top + 1, in_bottom - 1))
                                if tiles[idx(tx, ty)] == self.state.T_FLOOR:
                                    tiles[idx(tx, ty)] = self.state.T_TABLE

                        # Common tables on the right side.
                        for y in range(in_top + 1, in_bottom, 2):
                            for x in range(split_x + 2, in_right - 1, 3):
                                if (x, y) in corridor or near_door(x, y):
                                    continue
                                if tiles[idx(x, y)] == self.state.T_FLOOR:
                                    tiles[idx(x, y)] = self.state.T_TABLE
                    else:
                        split_y = int(clamp(int(y0 + h // 2), in_top + 2, in_bottom - 2))
                        for x in range(in_left, in_right + 1):
                            if near_door(x, split_y):
                                continue
                            tiles[idx(x, split_y)] = self.state.T_WALL

                        cell_n = 3 if (in_right - in_left + 1) >= 9 else 2
                        total_w = int(in_right - in_left + 1)
                        for i in range(1, cell_n):
                            bx = in_left + (i * total_w) // cell_n
                            if not (in_left + 1 <= bx <= in_right - 1):
                                continue
                            for y in range(in_top, split_y + 1):
                                if near_door(bx, y):
                                    continue
                                tiles[idx(bx, y)] = self.state.T_WALL

                        for i in range(cell_n):
                            x0c = in_left + (i * total_w) // cell_n
                            x1c = in_left + ((i + 1) * total_w) // cell_n - 1
                            dx = int(clamp((x0c + x1c) // 2, in_left + 1, in_right - 1))
                            if not near_door(dx, split_y):
                                tiles[idx(dx, split_y)] = self.state.T_DOOR
                                clear_interior(dx, split_y + 1)
                                mark_corridor(dx, split_y)

                            bx = int(clamp(dx, in_left + 1, in_right - 2))
                            by = in_top + 1
                            if tiles[idx(bx, by)] == self.state.T_FLOOR and tiles[idx(bx + 1, by)] == self.state.T_FLOOR:
                                tiles[idx(bx, by)] = self.state.T_BED
                                tiles[idx(bx + 1, by)] = self.state.T_BED
                            if rng.random() < 0.55:
                                tx = int(clamp(bx - 1, in_left + 1, in_right - 1))
                                ty = in_top + 1
                                if tiles[idx(tx, ty)] == self.state.T_FLOOR:
                                    tiles[idx(tx, ty)] = self.state.T_TABLE

                        for x in range(in_left + 1, in_right, 2):
                            for y in range(split_y + 2, in_bottom - 1, 3):
                                if (x, y) in corridor or near_door(x, y):
                                    continue
                                if tiles[idx(x, y)] == self.state.T_FLOOR:
                                    tiles[idx(x, y)] = self.state.T_TABLE

                # Prevent "see it but cannot enter" interior gaps: ensure all passable
                # tiles (floor/door) inside the building are reachable from the entrance.
                entrance_set: set[tuple[int, int]] = {(int(dx), int(dy)) for dx, dy in door_tiles}

                def passable(t: int) -> bool:
                    return int(t) in (int(self.state.T_FLOOR), int(self.state.T_DOOR))

                # Start tiles: door tiles + the tile just inside the entrance.
                starts: list[tuple[int, int]] = []
                starts.extend([(int(dx), int(dy)) for dx, dy in door_tiles])
                if door_side == "N":
                    starts.extend([(int(dx), int(dy) + 1) for dx, dy in door_tiles])
                elif door_side == "S":
                    starts.extend([(int(dx), int(dy) - 1) for dx, dy in door_tiles])
                elif door_side == "W":
                    starts.extend([(int(dx) + 1, int(dy)) for dx, dy in door_tiles])
                else:
                    starts.extend([(int(dx) - 1, int(dy)) for dx, dy in door_tiles])
                starts = [
                    (sx, sy)
                    for (sx, sy) in starts
                    if (int(x0) <= int(sx) < int(x0 + w) and int(y0) <= int(sy) < int(y0 + h) and passable(tiles[idx(int(sx), int(sy))]))
                ]

                if starts:
                    passable_tiles: set[tuple[int, int]] = set()
                    for py in range(int(y0), int(y0 + h)):
                        for px in range(int(x0), int(x0 + w)):
                            if passable(tiles[idx(px, py)]):
                                passable_tiles.add((int(px), int(py)))

                    def flood(start_list: list[tuple[int, int]]) -> set[tuple[int, int]]:
                        seen: set[tuple[int, int]] = set()
                        stack: list[tuple[int, int]] = list(start_list)
                        while stack:
                            x, y = stack.pop()
                            x = int(x)
                            y = int(y)
                            if (x, y) in seen:
                                continue
                            if not (int(x0) <= x < int(x0 + w) and int(y0) <= y < int(y0 + h)):
                                continue
                            if not passable(tiles[idx(x, y)]):
                                continue
                            seen.add((x, y))
                            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                        return seen

                    solids = {
                        int(self.state.T_WALL),
                        int(self.state.T_TABLE),
                        int(self.state.T_SHELF),
                        int(self.state.T_BED),
                    }

                    # Try to connect unreachable pockets by opening a wall (door) or clearing a blocking prop.
                    for _ in range(40):
                        reachable = flood(starts)
                        unreachable = passable_tiles.difference(reachable)
                        if not unreachable:
                            break

                        opened = False
                        for py in range(int(in_top), int(in_bottom) + 1):
                            for px in range(int(in_left), int(in_right) + 1):
                                t = int(tiles[idx(px, py)])
                                if t not in solids:
                                    continue

                                has_reach = False
                                has_unreach = False
                                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                                    if not (int(x0) <= nx < int(x0 + w) and int(y0) <= ny < int(y0 + h)):
                                        continue
                                    if not passable(tiles[idx(nx, ny)]):
                                        continue
                                    if (int(nx), int(ny)) in reachable:
                                        has_reach = True
                                    elif (int(nx), int(ny)) in unreachable:
                                        has_unreach = True
                                if has_reach and has_unreach:
                                    tiles[idx(px, py)] = self.state.T_DOOR if t == int(self.state.T_WALL) else self.state.T_FLOOR
                                    passable_tiles.add((int(px), int(py)))
                                    opened = True
                                    break
                            if opened:
                                break

                        if not opened:
                            # Fallback: seal unreachable walkable tiles so players never see fake rooms.
                            for ux, uy in unreachable:
                                if (int(ux), int(uy)) in entrance_set:
                                    continue
                                tiles[idx(int(ux), int(uy))] = self.state.T_WALL
                            break

                    # Remove dead-end internal doors (doors must connect >=2 passable neighbors).
                    for py in range(int(in_top), int(in_bottom) + 1):
                        for px in range(int(in_left), int(in_right) + 1):
                            if int(tiles[idx(px, py)]) != int(self.state.T_DOOR):
                                continue
                            if (int(px), int(py)) in entrance_set:
                                continue
                            n_pass = 0
                            for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                                if not (int(x0) <= nx < int(x0 + w) and int(y0) <= ny < int(y0 + h)):
                                    continue
                                if passable(tiles[idx(nx, ny)]):
                                    n_pass += 1
                            if n_pass < 2:
                                tiles[idx(px, py)] = self.state.T_WALL

                loot_choices: list[str]
                if town_kind == "医院":
                    loot_choices = ["bandage", "medkit", "water"]
                elif town_kind == "新华书店":
                    loot_choices = ["food_can", "water", "cola", "cola", "bandage", "scrap", "wood"]
                elif town_kind == "监狱":
                    loot_choices = ["pistol", "ammo_9mm", "ammo_9mm", "scrap", "bandage"]
                elif town_kind == "学校":
                    loot_choices = ["food_can", "water", "cola", "cola", "bandage", "scrap", "wood"]
                elif town_kind == "高层住宅":
                    loot_choices = ["food_can", "water", "cola", "bandage", "scrap", "ammo_9mm"]
                else:
                    loot_choices = ["food_can", "water", "cola", "cola", "ammo_9mm", "scrap", "wood"]

                loot_n = rng.randint(2, 4) if town_kind in ("超市", "大型超市", "新华书店", "医院") else rng.randint(2, 3)

                # Place loot only on tiles reachable from the entrance, so we
                # don't create "see the item but cannot enter" situations.
                start: tuple[int, int] | None = None
                entrance_candidates: list[tuple[int, int]] = []
                if door_side == "N":
                    entrance_candidates = [(int(dx), int(dy) + 1) for dx, dy in door_tiles] + list(door_tiles)
                elif door_side == "S":
                    entrance_candidates = [(int(dx), int(dy) - 1) for dx, dy in door_tiles] + list(door_tiles)
                elif door_side == "W":
                    entrance_candidates = [(int(dx) + 1, int(dy)) for dx, dy in door_tiles] + list(door_tiles)
                else:
                    entrance_candidates = [(int(dx) - 1, int(dy)) for dx, dy in door_tiles] + list(door_tiles)

                for sx, sy in entrance_candidates:
                    if not (int(x0) <= int(sx) < int(x0 + w) and int(y0) <= int(sy) < int(y0 + h)):
                        continue
                    if tiles[idx(int(sx), int(sy))] in (self.state.T_FLOOR, self.state.T_DOOR):
                        start = (int(sx), int(sy))
                        break

                reachable_floor: list[tuple[int, int]] = []
                if start is not None:
                    seen: set[tuple[int, int]] = set()
                    stack: list[tuple[int, int]] = [start]
                    while stack:
                        x, y = stack.pop()
                        x = int(x)
                        y = int(y)
                        if (x, y) in seen:
                            continue
                        if not (int(x0) <= x < int(x0 + w) and int(y0) <= y < int(y0 + h)):
                            continue
                        t = tiles[idx(x, y)]
                        if t not in (self.state.T_FLOOR, self.state.T_DOOR):
                            continue
                        seen.add((x, y))
                        if t == self.state.T_FLOOR and not near_door(x, y):
                            reachable_floor.append((x, y))
                        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

                if reachable_floor:
                    rng.shuffle(reachable_floor)

                spot_i = 0
                for _ in range(loot_n):
                    if spot_i >= len(reachable_floor):
                        break
                    ix, iy = reachable_floor[spot_i]
                    spot_i += 1

                    item_id = rng.choice(loot_choices)
                    idef = self.state._ITEMS.get(item_id)
                    if idef is None:
                        continue
                    if idef.kind == "ammo":
                        qty = rng.randint(12, 28)
                    elif idef.kind == "mat":
                        qty = rng.randint(2, 6)
                    else:
                        qty = 1
                    qty = int(clamp(qty, 1, idef.stack))

                    tx = cx * self.state.CHUNK_SIZE + ix
                    ty = cy * self.state.CHUNK_SIZE + iy
                    px = (tx + 0.5) * self.state.TILE_SIZE
                    py = (ty + 0.5) * self.state.TILE_SIZE
                    items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(px, py), item_id=item_id, qty=qty))

                if town_kind in ("监狱", "大型监狱") and spot_i < len(reachable_floor):
                    ix, iy = reachable_floor[spot_i]
                    tx = cx * self.state.CHUNK_SIZE + ix
                    ty = cy * self.state.CHUNK_SIZE + iy
                    px = (tx + 0.5) * self.state.TILE_SIZE
                    py = (ty + 0.5) * self.state.TILE_SIZE
                    items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(px, py), item_id="pistol", qty=1))
                    items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(px, py + 6), item_id="ammo_9mm", qty=rng.randint(18, 34)))

                roof_style = 0
                if town_kind == "住宅":
                    roof_style = 1
                elif town_kind in ("超市", "大型超市"):
                    roof_style = 2
                elif town_kind == "新华书店":
                    roof_style = 7
                elif town_kind == "医院":
                    roof_style = 3
                elif town_kind in ("监狱", "大型监狱"):
                    roof_style = 4
                elif town_kind == "学校":
                    roof_style = 5
                elif town_kind in ("高层住宅", "高层住宅大"):
                    roof_style = 6
                elif town_kind == "中式建筑":
                    roof_style = 8
                roof_var = int(rng.randrange(0, 256))
                roof_kind = (int(roof_style) << 8) | int(roof_var)
                buildings.append(
                    (
                        cx * self.state.CHUNK_SIZE + x0,
                        cy * self.state.CHUNK_SIZE + y0,
                        w,
                        h,
                        int(roof_kind),
                        int(building_floors),
                    )
                )
                placed += 1

    @dataclass
    class _Player:
        pos: pygame.Vector2
        vel: pygame.Vector2
        facing: pygame.Vector2
        w: int = 8
        h: int = 12
        hp: int = 100
        hunger: float = 100.0
        thirst: float = 100.0
        condition: float = 100.0
        morale: float = 100.0
        stamina: float = 100.0
        walk_phase: float = 0.0
        dir: str = "down"

        def rect_at(self, pos: pygame.Vector2 | None = None) -> pygame.Rect:
            p = self.pos if pos is None else pos
            return pygame.Rect(
                iround(float(p.x) - float(self.w) / 2.0),
                iround(float(p.y) - float(self.h) / 2.0),
                int(self.w),
                int(self.h),
            )

    def on_enter(self) -> None:
        self.seed = int(time.time()) & 0xFFFFFFFF
        self.world = HardcoreSurvivalState._World(self, seed=self.seed)
        spawn_tx, spawn_ty = self.world.spawn_tile()
        spawn_px = (spawn_tx + 0.5) * self.TILE_SIZE
        spawn_py = (spawn_ty + 0.5) * self.TILE_SIZE
        self.avatar = getattr(self, "avatar", None) or SurvivalAvatar()
        self.avatar.clamp_all()
        self.player = HardcoreSurvivalState._Player(
            pos=pygame.Vector2(spawn_px, spawn_py),
            vel=pygame.Vector2(0, 0),
            facing=pygame.Vector2(1, 0),
        )
        # Apply appearance + body size.
        self.player_frames = HardcoreSurvivalState.build_avatar_player_frames(self.avatar, run=False)
        self.player_frames_run = HardcoreSurvivalState.build_avatar_player_frames(self.avatar, run=True)
        self.cyclist_frames = HardcoreSurvivalState.build_avatar_cyclist_frames(self.avatar)
        self.player_sprinting = False
        gender = int(self.avatar.gender) % len(SURVIVAL_GENDER_OPTIONS)
        height = int(self.avatar.height) % len(SURVIVAL_HEIGHT_OPTIONS)
        self.player.w = 8 if gender == 0 else 7
        self.player.h = (11, 12, 13)[height]
        self.rv_int_player_w = (10 if gender == 0 else 9)
        self.rv_int_player_h = int(self._RV_INT_PLAYER_H) + (height - 1)
        self.inventory = HardcoreSurvivalState._Inventory(slots=[None] * 16)
        self.inv_open = False
        self.inv_index = 0
        self.rv_storage = HardcoreSurvivalState._Inventory(slots=[None] * 24, cols=6)
        self.rv_ui_open = False
        self.rv_ui_focus: str = "map"  # map | player | storage
        self.rv_ui_map = (0, 0)
        self.rv_ui_player_index = 0
        self.rv_ui_storage_index = 0
        self.rv_ui_status = ""
        self.rv_ui_status_left = 0.0
        # Apartment home storage (openable cabinets).
        self.home_storage = HardcoreSurvivalState._Inventory(slots=[None] * 30, cols=6)
        self.fridge_storage = HardcoreSurvivalState._Inventory(slots=[None] * 12, cols=4)
        self.home_ui_open = False
        self.home_ui_focus: str = "storage"  # player | storage
        self.home_ui_storage_kind: str = "cabinet"  # cabinet | fridge
        self.home_ui_player_index = 0
        self.home_ui_storage_index = 0
        self.home_ui_status = ""
        self.home_ui_status_left = 0.0
        self.home_ui_open_block: tuple[int, int, int, int] | None = None
        # Some starter supplies at home.
        self.home_storage.add("food_can", 4, self._ITEMS)
        self.home_storage.add("bandage", 6, self._ITEMS)
        self.fridge_storage.add("water", 2, self._ITEMS)
        self.fridge_storage.add("cola", 2, self._ITEMS)
        # Simple pose / action state (sit / sleep) used in interiors.
        self.player_pose: str | None = None  # sit | sleep
        self.player_pose_space: str = ""  # hr | rv
        self.player_pose_left = 0.0
        self.player_pose_anchor: tuple[float, float] | None = None
        self.player_pose_phase = 0.0
        self.hint_text = ""
        self.hint_left = 0.0
        self.gun: HardcoreSurvivalState._Gun | None = None
        self.bullets: list[HardcoreSurvivalState._Bullet] = []
        self.zombies: list[HardcoreSurvivalState._Zombie] = []
        self.spawn_left = 8.0
        self.zombie_cap = 8
        self.zombie_frozen = False
        self.dead_left = 0.0
        self.starve_accum = 0.0
        self.mount: str | None = None
        self.rv = HardcoreSurvivalState._RV(
            pos=pygame.Vector2(self.player.pos) + pygame.Vector2(34, 0),
            vel=pygame.Vector2(0, 0),
            model_id="rv",
            fuel=100.0,
        )
        self._apply_rv_model()
        self.rv_dir = "right"
        self.rv_anim = 0.0
        self.bike = HardcoreSurvivalState._Bike(
            pos=pygame.Vector2(self.player.pos) + pygame.Vector2(-28, 10),
            vel=pygame.Vector2(0, 0),
            model_id="bike_mountain",
        )
        self._apply_bike_model()
        self.bike_dir = "right"
        self.bike_anim = 0.0
        self.aim_dir = pygame.Vector2(1, 0)
        self.cam_x = 0
        self.cam_y = 0
        self._reload_lock_dir: pygame.Vector2 | None = None
        self.rv_interior = False
        self.rv_int_pos = pygame.Vector2(0, 0)
        self.rv_int_vel = pygame.Vector2(0, 0)
        self.rv_int_facing = pygame.Vector2(0, 1)
        self.rv_int_walk_phase = 0.0

        # High-rise interior (multi-floor apartment).
        self.hr_interior = False
        self.hr_int_pos = pygame.Vector2(0, 0)
        self.hr_int_vel = pygame.Vector2(0, 0)
        self.hr_int_facing = pygame.Vector2(0, 1)
        self.hr_int_walk_phase = 0.0
        self.hr_mode = "lobby"  # lobby | hall | home
        self.hr_floor = 0
        self.hr_layout: list[str] = list(self._HR_INT_LOBBY_LAYOUT)
        self.hr_building: HardcoreSurvivalState._SpecialBuilding | None = None
        self.hr_world_return = pygame.Vector2(self.player.pos)
        self.hr_auto_walk_to_elevator = False
        self.hr_auto_walk_delay = 0.0

        self.home_highrise_door: tuple[int, int] | None = None
        self.home_highrise_floor = 0
        self.home_highrise_door_index = 0
        self.home_highrise_unit = 0
        self.home_highrise_room = ""

        self.hr_elevator_ui_open = False
        self.hr_elevator_sel = 0
        self.hr_elevator_cols = 4
        self.hr_max_floors = int(self._HR_INT_MAX_FLOORS_DEFAULT)        

        # School interior (multi-floor).
        self.sch_interior = False
        self.sch_int_pos = pygame.Vector2(0, 0)
        self.sch_int_vel = pygame.Vector2(0, 0)
        self.sch_int_facing = pygame.Vector2(0, 1)
        self.sch_int_walk_phase = 0.0
        self.sch_floor = 1
        self.sch_layout: list[str] = list(self._SCH_INT_LOBBY_LAYOUT)
        self.sch_building: HardcoreSurvivalState._SpecialBuilding | None = None
        self.sch_world_return = pygame.Vector2(self.player.pos)
        self.sch_elevator_ui_open = False
        self.sch_elevator_sel = 0
        self.sch_elevator_cols = 4
        self.sch_max_floors = int(self._SCH_INT_MAX_FLOORS_DEFAULT)
        self.sch_elevator_input = ""

        tx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
        ty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
        spawn_chunk = self.world.get_chunk(tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE)
        # Pick a reliable home entrance near spawn (guarantees the map marker exists).
        spawn_cx = int(spawn_tx) // self.CHUNK_SIZE
        spawn_cy = int(spawn_ty) // self.CHUNK_SIZE
        best_home: tuple[int, int, int, int] | None = None  # (d2, tx, ty, floors)
        # Search a few chunks around spawn so it also finds the nearby compound towers.
        search_r = 4
        for cy2 in range(int(spawn_cy) - int(search_r), int(spawn_cy) + int(search_r) + 1):
            for cx2 in range(int(spawn_cx) - int(search_r), int(spawn_cx) + int(search_r) + 1):
                chunk2 = self.world.get_chunk(int(cx2), int(cy2))
                for sb in getattr(chunk2, "special_buildings", []):
                    if getattr(sb, "kind", "") != "highrise":
                        continue
                    floors2 = int(getattr(sb, "floors", 0) or 0)
                    if floors2 <= 0:
                        floors2 = int(self._HR_INT_MAX_FLOORS_DEFAULT)
                    for dtile in tuple(getattr(sb, "door_tiles", ()) or ()):
                        hx, hy = int(dtile[0]), int(dtile[1])
                        d2 = (hx - int(spawn_tx)) ** 2 + (hy - int(spawn_ty)) ** 2
                        if best_home is None or d2 < int(best_home[0]):
                            best_home = (int(d2), int(hx), int(hy), int(floors2))

        floors = int(self._HR_INT_MAX_FLOORS_DEFAULT)
        if best_home is not None:
            _d2, hx, hy, floors = best_home
            self.home_highrise_door = (int(hx), int(hy))
        else:
            # Fallback: still show a marker instead of having "no home".
            self.home_highrise_door = (int(spawn_tx), int(spawn_ty))

        # Fixed "Chinese community" home for now: 6F room 604.
        self.home_highrise_floor = int(clamp(6, 2, int(floors)))
        # Unit numbering follows the door list order: 01..N
        self.home_highrise_unit = 4
        door_count = int(max(1, len(self._HR_INT_APT_DOORS)))
        self.home_highrise_unit = int(clamp(int(self.home_highrise_unit), 1, door_count))
        self.home_highrise_door_index = int(self.home_highrise_unit - 1)
        self.home_highrise_room = f"{int(self.home_highrise_floor)}{int(self.home_highrise_unit):02d}"
        self._set_hint(f"你的家：{self.home_highrise_room}", seconds=2.0)

        # Ensure the starter vehicles don't spawn "inside" a building.
        def vehicle_clear(p: pygame.Vector2, w: int, h: int) -> bool:
            r = pygame.Rect(int(round(p.x - w / 2)), int(round(p.y - h / 2)), int(w), int(h))
            return len(self._collide_rect_world_vehicle(r)) == 0

        def find_clear_vehicle_pos(desired: pygame.Vector2, *, w: int, h: int, max_r_tiles: int = 18) -> pygame.Vector2:
            base_tx = int(math.floor(float(desired.x) / float(self.TILE_SIZE)))
            base_ty = int(math.floor(float(desired.y) / float(self.TILE_SIZE)))
            if vehicle_clear(desired, int(w), int(h)):
                return pygame.Vector2(desired)
            for r in range(1, int(max_r_tiles) + 1):
                for dy2 in (-r, r):
                    for dx2 in range(-r, r + 1):
                        tx2 = int(base_tx + dx2)
                        ty2 = int(base_ty + dy2)
                        px = (float(tx2) + 0.5) * float(self.TILE_SIZE)
                        py = (float(ty2) + 0.5) * float(self.TILE_SIZE)
                        cand = pygame.Vector2(px, py)
                        if vehicle_clear(cand, int(w), int(h)):
                            return cand
                for dx2 in (-r, r):
                    for dy2 in range(-r + 1, r):
                        tx2 = int(base_tx + dx2)
                        ty2 = int(base_ty + dy2)
                        px = (float(tx2) + 0.5) * float(self.TILE_SIZE)
                        py = (float(ty2) + 0.5) * float(self.TILE_SIZE)
                        cand = pygame.Vector2(px, py)
                        if vehicle_clear(cand, int(w), int(h)):
                            return cand
            return pygame.Vector2(desired)

        self.rv.pos = find_clear_vehicle_pos(pygame.Vector2(self.rv.pos), w=int(self.rv.w), h=int(self.rv.h))
        self.bike.pos = find_clear_vehicle_pos(pygame.Vector2(self.bike.pos), w=int(self.bike.w), h=int(self.bike.h))
        spawn_chunk.items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(self.player.pos) + pygame.Vector2(18, 0), item_id="pistol", qty=1))
        spawn_chunk.items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(self.player.pos) + pygame.Vector2(18, 10), item_id="ammo_9mm", qty=24))
        # Default start: spring morning (season is derived from day index).
        self.world_time_s = float(self.DAY_LENGTH_S) * float(self.START_DAY_FRACTION)
        self._debug = False
        self._gallery_open = False
        self._gallery_page = 0  # 0=cars, 1=bike
        self._roof_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self._sprite_outline_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self._hover_tooltip: tuple[list[str], tuple[int, int]] | None = None
        self.muzzle_flash_left = 0.0
        self.noise_left = 0.0
        self.noise_radius = 0.0

        # Weather (visual-first prototype).
        self.weather_kind = "clear"  # clear | cloudy | rain | storm | snow
        self.weather_intensity = 0.0  # 0..1
        self.weather_target_kind = "clear"
        self.weather_target_intensity = 0.0
        self.weather_day = 0
        self.weather_wind = 0.0  # px/s (screen-space drift)
        self.weather_wind_target = 0.0
        self.weather_flash_left = 0.0
        self.weather_rng = random.Random(self.seed ^ 0xA53C9F17)
        self._rain_drops: list[list[float]] = []  # [x, y, speed]
        self._snow_flakes: list[list[float]] = []  # [x, y, speed, drift]
        self._weather_layer = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        self._set_weather_targets_for_day(int(self.world_time_s / self.DAY_LENGTH_S) + 1)
        self.weather_kind = str(self.weather_target_kind)
        self.weather_intensity = float(self.weather_target_intensity)
        self.weather_wind = float(self.weather_wind_target)
        self.weather_day = int(self.world_time_s / self.DAY_LENGTH_S) + 1       
        self._sync_weather_particles()

        # Map UI.
        self.minimap_rect: pygame.Rect | None = None
        self.world_map_open = False
        self.world_map_scale = 3  # px per tile (big map)
        self.world_map_legend_open = True
        self.world_map_center = self._player_tile()
        self.world_map_markers: list[HardcoreSurvivalState._MapMarker] = []
        self.world_map_marker_next = 0
        self._world_map_cache_key: tuple[int, int, int, int, int, int] | None = None
        self._world_map_cache_scaled: pygame.Surface | None = None
        self._world_map_draw_rect: pygame.Rect | None = None
        self._world_map_start_tile: tuple[int, int] = (0, 0)
        self.world_map_dragging = False
        self.world_map_drag_start: tuple[int, int] = (0, 0)
        self.world_map_drag_center: tuple[int, int] = (0, 0)

    def handle_event(self, event: pygame.event.Event) -> None:
        if getattr(self, "world_map_open", False):
            if event.type == pygame.KEYDOWN:
                self._handle_world_map_key(int(event.key))
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
                self._handle_world_map_mouse(event)
                return

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
            if internal is not None and isinstance(getattr(self, "minimap_rect", None), pygame.Rect):
                if self.minimap_rect.collidepoint(internal):
                    self._toggle_world_map(open=True)
                    return

        if getattr(self, "home_ui_open", False):
            if event.type == pygame.KEYDOWN:
                self._handle_home_ui_key(int(event.key))
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
                self._handle_home_ui_mouse(event)
                return

        if getattr(self, "sch_elevator_ui_open", False):
            if event.type == pygame.KEYDOWN:
                self._handle_sch_elevator_ui_key(int(event.key))        
                return
            if event.type == pygame.MOUSEWHEEL:
                self._handle_sch_elevator_ui_wheel(event)
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                self._handle_sch_elevator_ui_mouse(event)
                return
            return

        if getattr(self, "hr_elevator_ui_open", False):
            if event.type == pygame.KEYDOWN:
                self._handle_hr_elevator_ui_key(int(event.key))
                return
            if event.type == pygame.MOUSEWHEEL:
                self._handle_hr_elevator_ui_wheel(event)
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                self._handle_hr_elevator_ui_mouse(event)
                return
            return

        if getattr(self, "rv_interior", False) and bool(getattr(self, "rv_edit_mode", False)):
            if self._rv_edit_handle_mouse(event):
                return

        if getattr(self, "hr_interior", False) and bool(getattr(self, "hr_edit_mode", False)):
            if self._hr_edit_handle_mouse(event):
                return

        if event.type == pygame.KEYDOWN:
            if getattr(self, "rv_ui_open", False):
                self._handle_rv_ui_key(int(event.key))
                return
            if getattr(self, "_gallery_open", False):
                if event.key in (pygame.K_ESCAPE, pygame.K_F2):
                    self._gallery_open = False
                    return
                if event.key in (pygame.K_TAB, pygame.K_RIGHT, pygame.K_d):
                    self._gallery_page = (int(self._gallery_page) + 1) % 2
                    return
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self._gallery_page = (int(self._gallery_page) - 1) % 2
                    return
                return
            if event.key in (pygame.K_m,):
                self._toggle_world_map()
                return
            if getattr(self, "sch_interior", False):
                if event.key in (pygame.K_ESCAPE,):
                    self._sch_interior_exit()
                    return
                if event.key in (pygame.K_e, pygame.K_RETURN, pygame.K_SPACE):
                    self._sch_interior_interact()
                    return
                if event.key in (pygame.K_f, pygame.K_v, pygame.K_h):
                    return
            if getattr(self, "hr_interior", False):
                if event.key in (pygame.K_ESCAPE,):
                    if str(getattr(self, "hr_mode", "lobby")) == "home":        
                        self._hr_leave_home()
                    else:
                        self._hr_interior_exit()
                    return
                if event.key in (pygame.K_r,) and str(getattr(self, "hr_mode", "lobby")) == "home":
                    self._hr_toggle_edit_mode()
                    return
                if getattr(self, "hr_edit_mode", False):
                    if event.key in (pygame.K_TAB,):
                        self._hr_edit_cycle(1)
                        return
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        self._hr_edit_move(-1, 0)
                        return
                    if event.key in (pygame.K_RIGHT, pygame.K_d):
                        self._hr_edit_move(1, 0)
                        return
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self._hr_edit_move(0, -1)
                        return
                    if event.key in (pygame.K_DOWN, pygame.K_s):
                        self._hr_edit_move(0, 1)
                        return
                    if event.key in (pygame.K_e, pygame.K_RETURN, pygame.K_SPACE):
                        return
                if event.key in (pygame.K_e, pygame.K_RETURN, pygame.K_SPACE):  
                    self._hr_interior_interact()
                    return
                if event.key in (pygame.K_f, pygame.K_v, pygame.K_h):
                    return
            if getattr(self, "rv_interior", False):
                if event.key in (pygame.K_ESCAPE, pygame.K_h):
                    self._rv_interior_exit()
                    return
                if event.key in (pygame.K_r,):
                    self._rv_toggle_edit_mode()
                    return
                if getattr(self, "rv_edit_mode", False):
                    if event.key in (pygame.K_TAB,):
                        self._rv_edit_cycle(1)
                        return
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        self._rv_edit_move(-1, 0)
                        return
                    if event.key in (pygame.K_RIGHT, pygame.K_d):
                        self._rv_edit_move(1, 0)
                        return
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self._rv_edit_move(0, -1)
                        return
                    if event.key in (pygame.K_DOWN, pygame.K_s):
                        self._rv_edit_move(0, 1)
                        return
                if event.key in (pygame.K_e, pygame.K_RETURN, pygame.K_SPACE):  
                    self._rv_interior_interact()
                    return
                if event.key in (pygame.K_f, pygame.K_v):
                    return
            if event.key in (pygame.K_ESCAPE,):
                if self.inv_open:
                    self.inv_open = False
                else:
                    self.app.set_state(MainMenuState(self.app))
                return
            if event.key in (pygame.K_TAB,):
                self.inv_open = not self.inv_open
                return
            if event.key in (pygame.K_F2,):
                self.inv_open = False
                self._gallery_page = 0
                self._gallery_open = not bool(getattr(self, "_gallery_open", False))
                return
            if event.key in (pygame.K_e,):
                self._interact_primary()
                return
            if not self.inv_open and event.key in (pygame.K_h,):
                self._rv_interior_toggle()
                return
            if not self.inv_open and event.key in (pygame.K_f,):
                self._toggle_vehicle()
                return
            if not self.inv_open and event.key in (pygame.K_r,):
                self._start_reload()
                return
            if not self.inv_open and event.key in (pygame.K_v,):
                self._cycle_rv_model()
                return
            if self.inv_open and event.key in (pygame.K_q,):
                self._drop_selected()
                return
            if self.inv_open and event.key in (pygame.K_LEFT, pygame.K_a):
                self.inv_index = (self.inv_index - 1) % len(self.inventory.slots)
                return
            if self.inv_open and event.key in (pygame.K_RIGHT, pygame.K_d):
                self.inv_index = (self.inv_index + 1) % len(self.inventory.slots)
                return
            if self.inv_open and event.key in (pygame.K_UP, pygame.K_w):
                self.inv_index = (self.inv_index - self.inventory.cols) % len(self.inventory.slots)
                return
            if self.inv_open and event.key in (pygame.K_DOWN, pygame.K_s):
                self.inv_index = (self.inv_index + self.inventory.cols) % len(self.inventory.slots)
                return
            if self.inv_open and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._equip_selected()
                return
            if event.key in (pygame.K_F3,):
                self._debug = not self._debug
                return

    def _apply_rv_model(self) -> None:
        mid = getattr(self.rv, "model_id", "schoolbus")
        model = self._CAR_MODELS.get(str(mid))
        if model is None:
            model = self._CAR_MODELS["schoolbus"]
            self.rv.model_id = model.id
        self.rv.w = int(model.collider[0])
        self.rv.h = int(model.collider[1])

    def _two_wheel_name(self, model_id: str) -> str:
        model_id = str(model_id)
        name_map = {
            "bike": "自行车",
            "bike_lady": "女士自行车",
            "bike_mountain": "山地车",
            "bike_auto": "自动自行车",
            "moto": "摩托车",
            "moto_lux": "豪华摩托车",
            "moto_long": "加长摩托车",
        }
        return str(name_map.get(model_id, model_id))

    def _two_wheel_collider_px(self, model_id: str) -> tuple[int, int]:
        mid = str(model_id)
        collider_map: dict[str, tuple[int, int]] = {
            "bike": (14, 10),
            "bike_lady": (14, 10),
            "bike_mountain": (14, 10),
            "bike_auto": (14, 10),
            "moto": (16, 12),
            "moto_lux": (16, 12),
            "moto_long": (18, 12),
        }
        return collider_map.get(mid, (14, 10))

    def _apply_bike_model(self) -> None:
        if not hasattr(self, "bike") or self.bike is None:
            return
        mid = str(getattr(self.bike, "model_id", "bike"))
        if mid not in self._TWO_WHEEL_FRAMES:
            mid = "bike"
            self.bike.model_id = mid

        w, h = self._two_wheel_collider_px(mid)
        self.bike.w = int(w)
        self.bike.h = int(h)

    def _bike_base_speed(self) -> float:
        mid = str(getattr(getattr(self, "bike", None), "model_id", "bike"))
        speed_map: dict[str, float] = {
            "bike": 92.0,
            "bike_lady": 90.0,
            "bike_mountain": 96.0,
            "bike_auto": 108.0,
            "moto": 148.0,
            "moto_lux": 158.0,
            "moto_long": 152.0,
        }
        return float(speed_map.get(mid, 92.0))

    def _bike_uses_stamina(self) -> bool:
        mid = str(getattr(getattr(self, "bike", None), "model_id", "bike"))
        if mid.startswith("moto"):
            return False
        if mid == "bike_auto":
            return False
        return True

    def _sprite_baseline_y(self, spr: pygame.Surface) -> int:
        cache = getattr(self, "_sprite_baseline_cache", None)
        if cache is None:
            cache = {}
            self._sprite_baseline_cache = cache
        key = int(id(spr))
        by = cache.get(key)
        if by is not None:
            return int(by)

        w, h = spr.get_size()
        baseline = int(h - 1)
        for y in range(int(h) - 1, -1, -1):
            for x in range(int(w)):
                if int(spr.get_at((int(x), int(y))).a) > 0:
                    baseline = int(y)
                    cache[key] = int(baseline)
                    return int(baseline)
        cache[key] = int(baseline)
        return int(baseline)

    def _two_wheel_shadow_rect(
        self,
        spr_rect: pygame.Rect,
        d: str,
        *,
        ground_y: int | None = None,
    ) -> pygame.Rect:
        d = str(d)
        if d in ("left", "right"):
            shadow_w = max(12, int(spr_rect.w) - 3)
        else:
            shadow_w = max(10, int(spr_rect.w) - 6)
        shadow_h = 4
        shadow = pygame.Rect(0, 0, int(shadow_w), int(shadow_h))
        cy = int(spr_rect.bottom - 2) if ground_y is None else int(ground_y - 2)
        shadow.center = (int(spr_rect.centerx), int(cy))
        return shadow

    def _cycle_rv_model(self) -> None:
        if not hasattr(self, "rv") or self.rv is None:
            return
        rv_d2 = float((self.player.pos - self.rv.pos).length_squared())
        can_change = self.mount == "rv" or rv_d2 <= (36.0 * 36.0)
        if not can_change:
            return

        ids = list(self._CAR_MODELS.keys())
        if not ids:
            return
        cur = str(getattr(self.rv, "model_id", ids[0]))
        try:
            idx = ids.index(cur)
        except ValueError:
            idx = 0
        self.rv.model_id = ids[(idx + 1) % len(ids)]
        self._apply_rv_model()
        model = self._CAR_MODELS.get(str(self.rv.model_id))
        name = model.name if model is not None else str(self.rv.model_id)
        self._set_hint(f"房车型号：{name}", seconds=1.5)

    def _set_rv_ui_status(self, text: str, *, seconds: float = 1.6) -> None:
        self.rv_ui_status = str(text)
        self.rv_ui_status_left = float(max(0.0, seconds))

    def _set_home_ui_status(self, text: str, *, seconds: float = 1.6) -> None:
        self.home_ui_status = str(text)
        self.home_ui_status_left = float(max(0.0, seconds))

    def _clear_player_pose(self) -> None:
        self.player_pose = None
        self.player_pose_space = ""
        self.player_pose_left = 0.0
        self.player_pose_anchor = None
        self.player_pose_phase = 0.0

    def _set_player_pose(
        self,
        pose: str,
        *,
        space: str,
        anchor: tuple[float, float],
        seconds: float = 0.0,
    ) -> None:
        self.player_pose = str(pose)
        self.player_pose_space = str(space)
        self.player_pose_anchor = (float(anchor[0]), float(anchor[1]))
        self.player_pose_left = float(max(0.0, seconds))
        self.player_pose_phase = 0.0

    def _get_pose_sprite(self, pose: str, *, direction: str = "down", frame: int = 0) -> pygame.Surface:
        av = getattr(self, "avatar", None) or SurvivalAvatar()
        av.clamp_all()
        direction = str(direction)
        if direction not in ("down", "up", "right", "left"):
            direction = "down"
        key = (
            str(pose),
            str(direction),
            int(frame) % 2,
            int(av.gender),
            int(av.height),
            int(av.face),
            int(av.eyes),
            int(av.hair),
            int(av.nose),
            int(av.outfit),
        )
        cache = getattr(self, "_pose_sprite_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._pose_sprite_cache = cache
        spr = cache.get(key)
        if isinstance(spr, pygame.Surface):
            return spr
        if str(pose) == "sleep":
            spr = HardcoreSurvivalState._make_avatar_pose_sleep_sprite(int(frame), avatar=av)
        else:
            spr = HardcoreSurvivalState._make_avatar_pose_sit_sprite(str(direction), avatar=av)
        cache[key] = spr
        return spr

    def _can_access_rv(self) -> bool:
        if self.mount == "rv":
            return True
        return (self.player.pos - self.rv.pos).length_squared() <= (22.0 * 22.0)

    def _rv_int_find(self, ch: str) -> tuple[int, int] | None:
        ch = str(ch)[:1]
        layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)
        for y, row in enumerate(layout):
            for x in range(min(len(row), int(self._RV_INT_W))):
                if row[x] == ch:
                    return int(x), int(y)
        return None

    def _rv_interior_toggle(self) -> None:
        if getattr(self, "rv_interior", False):
            self._rv_interior_exit()
            return
        if not self._can_access_rv():
            self._set_hint("靠近房车按 H 进入房车内部", seconds=1.2)
            return
        if self.mount == "rv" and abs(float(getattr(self.rv, "speed", 0.0))) > 1.0:
            self._set_hint("先停车再进房车内部", seconds=1.2)
            return
        self._rv_interior_enter()

    def _rv_interior_enter(self) -> None:
        self.inv_open = False
        self.rv_ui_open = False
        self._gallery_open = False
        self.rv_interior = True
        self.mount = None
        self.player.vel.update(0, 0)
        self.player.walk_phase *= 0.85
        self.rv.vel.update(0, 0)
        self.bike.vel.update(0, 0)

        if not isinstance(getattr(self, "rv_int_layout", None), list):
            self.rv_int_layout = [list(row) for row in self._RV_INT_LAYOUT]
        self.rv_edit_mode = False
        self.rv_edit_dragging = False
        self.rv_edit_blocks: list[tuple[str, list[tuple[int, int]]]] = []
        self.rv_edit_index = 0

        door = self._rv_int_find("D")
        if door is None:
            door = (int(self._RV_INT_W) - 2, 1)
        dx, dy = int(door[0]), int(door[1])
        tile = int(self._RV_INT_TILE_SIZE)

        # Spawn just inside the doorway so movement isn't immediately blocked.
        spawn_candidates: list[tuple[int, int]] = []
        if dx <= 0:
            spawn_candidates.append((dx + 1, dy))
        elif dx >= int(self._RV_INT_W) - 1:
            spawn_candidates.append((dx - 1, dy))
        if dy <= 0:
            spawn_candidates.append((dx, dy + 1))
        elif dy >= int(self._RV_INT_H) - 1:
            spawn_candidates.append((dx, dy - 1))
        spawn_candidates.extend([(dx, dy), (dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)])

        sx, sy = dx, dy
        for cx, cy in spawn_candidates:
            ch = self._rv_int_char_at(cx, cy)
            if ch == "W":
                continue
            if self._rv_int_solid_tile(ch):
                continue
            sx, sy = int(cx), int(cy)
            if ch != "D":
                break

        self.rv_int_pos = pygame.Vector2((sx + 0.5) * tile, (sy + 0.5) * tile)
        self.rv_int_vel = pygame.Vector2(0, 0)
        if dx >= int(self._RV_INT_W) - 1:
            self.rv_int_facing = pygame.Vector2(-1, 0)
        elif dx <= 0:
            self.rv_int_facing = pygame.Vector2(1, 0)
        elif dy >= int(self._RV_INT_H) - 1:
            self.rv_int_facing = pygame.Vector2(0, -1)
        elif dy <= 0:
            self.rv_int_facing = pygame.Vector2(0, 1)
        else:
            self.rv_int_facing = pygame.Vector2(0, 1)
        self.rv_int_walk_phase = 0.0
        self.player.pos.update(self.rv.pos)
        self._set_hint("房车内部：WASD 走动  E 互动  H/ESC 退出", seconds=1.6)

    def _rv_interior_exit(self) -> None:
        if not getattr(self, "rv_interior", False):
            return
        self._clear_player_pose()
        self.rv_interior = False
        self.rv_ui_open = False
        self.rv_edit_mode = False
        self.rv_edit_dragging = False

        base = pygame.Vector2(self.rv.pos)
        fwd = pygame.Vector2(math.cos(float(self.rv.heading)), math.sin(float(self.rv.heading)))
        if fwd.length_squared() <= 0.001:
            fwd = pygame.Vector2(1, 0)
        fwd = fwd.normalize()
        candidates = [
            fwd * 26.0,
            fwd.rotate(90) * 26.0,
            fwd.rotate(-90) * 26.0,
            -fwd * 26.0,
            pygame.Vector2(26, 0),
            pygame.Vector2(-26, 0),
            pygame.Vector2(0, 26),
            pygame.Vector2(0, -26),
        ]
        placed = False
        for off in candidates:
            pos = base + off
            rect = pygame.Rect(
                iround(float(pos.x) - float(self.player.w) / 2.0),
                iround(float(pos.y) - float(self.player.h) / 2.0),
                int(self.player.w),
                int(self.player.h),
            )
            if self._collide_rect_world(rect):
                continue
            self.player.pos.update(pos)
            placed = True
            break
        if not placed:
            self.player.pos.update(base)

        self._set_hint("离开房车", seconds=0.9)

    def _sch_int_set_layout(self, layout: list[str]) -> None:
        self.sch_layout = [str(r) for r in layout]

    def _sch_int_find(self, ch: str, *, layout: list[str] | None = None) -> tuple[int, int] | None:
        ch = str(ch)[:1]
        rows = self.sch_layout if layout is None else layout
        for y, row in enumerate(rows):
            x = str(row).find(ch)
            if x >= 0:
                return int(x), int(y)
        return None

    def _sch_int_char_at(self, x: int, y: int) -> str:
        x = int(x)
        y = int(y)
        if not (0 <= x < int(self._SCH_INT_W) and 0 <= y < int(self._SCH_INT_H)):
            return "W"
        row = self.sch_layout[y] if 0 <= y < len(self.sch_layout) else ""
        if 0 <= x < len(row):
            return row[x]
        return "W"

    def _sch_int_solid_tile(self, ch: str) -> bool:
        ch = str(ch)[:1]
        return ch not in (".", "D", "E", "^", "v")

    def _sch_int_move_box(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float, *, w: int, h: int) -> pygame.Vector2:
        tile = int(self._SCH_INT_TILE_SIZE)
        rect = pygame.Rect(int(round(pos.x - w / 2)), int(round(pos.y - h / 2)), int(w), int(h))
        dx = float(vel.x * dt)
        dy = float(vel.y * dt)

        def collide(r: pygame.Rect) -> list[pygame.Rect]:
            left = int(math.floor(r.left / tile))
            right = int(math.floor((r.right - 1) / tile))
            top = int(math.floor(r.top / tile))
            bottom = int(math.floor((r.bottom - 1) / tile))
            hits: list[pygame.Rect] = []
            for ty in range(top, bottom + 1):
                for tx in range(left, right + 1):
                    ch = self._sch_int_char_at(tx, ty)
                    if not self._sch_int_solid_tile(ch):
                        continue
                    tr = pygame.Rect(tx * tile, ty * tile, tile, tile)
                    if r.colliderect(tr):
                        hits.append(tr)
            return hits

        if dx != 0.0:
            rect.x += int(round(dx))
            for hit in collide(rect):
                if dx > 0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right

        if dy != 0.0:
            rect.y += int(round(dy))
            for hit in collide(rect):
                if dy > 0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom

        pad = 3
        half_w = float(w) / 2.0
        half_h = float(h) / 2.0
        min_x = tile + half_w + pad
        min_y = tile + half_h + pad
        max_x = int(self._SCH_INT_W) * tile - tile - half_w - pad
        max_y = int(self._SCH_INT_H) * tile - tile - half_h - pad
        rect.centerx = int(clamp(rect.centerx, min_x, max_x))
        rect.centery = int(clamp(rect.centery, min_y, max_y))

        return pygame.Vector2(rect.centerx, rect.centery)

    def _sch_int_player_tile(self) -> tuple[int, int]:
        tile = int(self._SCH_INT_TILE_SIZE)
        tx = int(math.floor(float(self.sch_int_pos.x) / tile))
        ty = int(math.floor(float(self.sch_int_pos.y) / tile))
        return tx, ty

    def _sch_set_floor(self, floor: int, *, spawn_at: str = "elevator") -> None:
        max_f = int(max(1, int(getattr(self, "sch_max_floors", int(self._SCH_INT_MAX_FLOORS_DEFAULT)))))
        floor = int(clamp(int(floor), 1, max_f))
        self.sch_floor = int(floor)
        if floor <= 1:
            self._sch_int_set_layout(self._SCH_INT_LOBBY_LAYOUT)
        else:
            self._sch_int_set_layout(self._SCH_INT_FLOOR_LAYOUT)

        tile = int(self._SCH_INT_TILE_SIZE)
        target = "E"
        if spawn_at == "stairs_up":
            target = "^"
        elif spawn_at == "stairs_down":
            target = "v"
        elif spawn_at == "door_inside":
            target = "D"

        pt = self._sch_int_find(target)
        if pt is None:
            self.sch_int_pos = pygame.Vector2((self._SCH_INT_W / 2.0) * tile, (self._SCH_INT_H / 2.0) * tile)
        else:
            sx, sy = pt
            if spawn_at == "door_inside":
                spawn = (sx - 1, sy) if sx >= int(self._SCH_INT_W) - 1 else (sx + 1, sy)
                px, py = int(spawn[0]), int(spawn[1])
                if self._sch_int_solid_tile(self._sch_int_char_at(px, py)):
                    px, py = int(sx), int(sy)
                self.sch_int_pos = pygame.Vector2((px + 0.5) * tile, (py + 0.5) * tile)
            else:
                self.sch_int_pos = pygame.Vector2((sx + 0.5) * tile, (sy + 0.5) * tile)
        self.sch_int_vel = pygame.Vector2(0, 0)
        self.sch_int_facing = pygame.Vector2(0, 1)
        self.sch_int_walk_phase = 0.0

    def _sch_elevator_open(self) -> None:
        if not getattr(self, "sch_interior", False):
            return
        max_f = int(max(1, int(getattr(self, "sch_max_floors", int(self._SCH_INT_MAX_FLOORS_DEFAULT)))))
        options = list(range(max_f, 0, -1))
        self.sch_elevator_options = options
        cur = int(getattr(self, "sch_floor", 1))
        try:
            self.sch_elevator_sel = int(options.index(cur))
        except ValueError:
            self.sch_elevator_sel = 0
        self.sch_elevator_input = ""
        self.sch_elevator_ui_open = True
        self._set_hint("学校电梯：方向键选择 | PgUp/PgDn翻页 | 数字直达 | Enter确认", seconds=1.8)

    def _handle_sch_elevator_ui_wheel(self, event: pygame.event.Event) -> None:
        if not getattr(self, "sch_elevator_ui_open", False):
            return
        options: list[int] = list(getattr(self, "sch_elevator_options", []))
        if not options:
            options = [1]
            self.sch_elevator_options = options
        n = len(options)
        cols = max(1, int(getattr(self, "sch_elevator_cols", 4)))
        rows_per_page = 4
        page_size = max(1, cols * rows_per_page)
        step = cols
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            step = page_size
        dy = int(getattr(event, "y", 0))
        if dy == 0:
            return
        sel = int(getattr(self, "sch_elevator_sel", 0)) % n
        sel = (sel - step) % n if dy > 0 else (sel + step) % n
        self.sch_elevator_sel = int(sel)
        self.sch_elevator_input = ""

    def _handle_sch_elevator_ui_mouse(self, event: pygame.event.Event) -> None:
        if not getattr(self, "sch_elevator_ui_open", False):
            return

        et = int(getattr(event, "type", 0))
        button = int(getattr(event, "button", 0))
        if et == int(pygame.MOUSEBUTTONDOWN) and button == 3:
            self.sch_elevator_ui_open = False
            self.sch_elevator_input = ""
            return

        if et not in (int(pygame.MOUSEMOTION), int(pygame.MOUSEBUTTONDOWN)):
            return
        if et == int(pygame.MOUSEBUTTONDOWN) and button != 1:
            return

        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return
        mx, my = int(internal[0]), int(internal[1])

        options: list[int] = list(getattr(self, "sch_elevator_options", []))
        if not options:
            options = [1]
            self.sch_elevator_options = options
        cols = max(1, int(getattr(self, "sch_elevator_cols", 4)))
        rows_per_page = 4
        n = len(options)
        page_size = max(1, cols * rows_per_page)
        sel_i = int(getattr(self, "sch_elevator_sel", 0)) % max(1, n)
        page = int(sel_i // page_size)
        start = int(page * page_size)
        end = int(min(int(n), int(start + page_size)))
        page_opts = list(options[start:end])
        rows = int(max(1, int(math.ceil(float(len(page_opts)) / float(cols)))))

        btn_w = 56
        btn_h = 28
        gap = 8
        grid_w = cols * btn_w + (cols - 1) * gap
        grid_h = rows * btn_h + (rows - 1) * gap
        panel = pygame.Rect(0, 0, grid_w + 40, grid_h + 98)
        panel.center = (INTERNAL_W // 2, INTERNAL_H // 2)

        x0 = panel.centerx - grid_w // 2
        y0 = panel.top + 66
        hit_i: int | None = None
        hit_floor: int | None = None
        for local_i, f in enumerate(page_opts):
            r = int(local_i) // cols
            c = int(local_i) % cols
            x = x0 + c * (btn_w + gap)
            y = y0 + r * (btn_h + gap)
            br = pygame.Rect(x, y, btn_w, btn_h)
            if br.collidepoint(mx, my):
                hit_i = int(start) + int(local_i)
                hit_floor = int(f)
                break

        if hit_i is None or hit_floor is None:
            return

        self.sch_elevator_sel = int(hit_i)
        self.sch_elevator_input = ""
        if et == int(pygame.MOUSEBUTTONDOWN) and button == 1:
            self.sch_elevator_ui_open = False
            self._sch_set_floor(int(hit_floor), spawn_at="elevator")

    def _handle_sch_elevator_ui_key(self, key: int) -> None:
        if not getattr(self, "sch_elevator_ui_open", False):
            return
        options: list[int] = list(getattr(self, "sch_elevator_options", []))
        if not options:
            options = [1]
            self.sch_elevator_options = options
        n = len(options)
        cols = max(1, int(getattr(self, "sch_elevator_cols", 4)))
        rows_per_page = 4
        page_size = max(1, cols * rows_per_page)
        sel = int(getattr(self, "sch_elevator_sel", 0)) % n

        digit = self._key_digit(int(key))
        if digit is not None:
            cur = str(getattr(self, "sch_elevator_input", ""))
            if len(cur) < 3:
                self.sch_elevator_input = cur + digit
            return
        if key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            cur = str(getattr(self, "sch_elevator_input", ""))
            self.sch_elevator_input = cur[:-1]
            return

        if key in (pygame.K_ESCAPE,):
            self.sch_elevator_ui_open = False
            self.sch_elevator_input = ""
            return
        if key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_e):
            self.sch_elevator_ui_open = False
            typed = str(getattr(self, "sch_elevator_input", "")).strip()
            self.sch_elevator_input = ""
            if typed:
                try:
                    target = int(typed)
                except ValueError:
                    target = int(options[sel])
                self._sch_set_floor(int(target), spawn_at="elevator")
            else:
                self._sch_set_floor(int(options[sel]), spawn_at="elevator")
            return

        if key in (pygame.K_PAGEUP,):
            self.sch_elevator_sel = int((sel - page_size) % n)
            self.sch_elevator_input = ""
            return
        if key in (pygame.K_PAGEDOWN,):
            self.sch_elevator_sel = int((sel + page_size) % n)
            self.sch_elevator_input = ""
            return
        if key in (pygame.K_HOME,):
            self.sch_elevator_sel = 0
            self.sch_elevator_input = ""
            return
        if key in (pygame.K_END,):
            self.sch_elevator_sel = int(n - 1)
            self.sch_elevator_input = ""
            return

        if key in (pygame.K_LEFT, pygame.K_a):
            self.sch_elevator_sel = int((sel - 1) % n)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.sch_elevator_sel = int((sel + 1) % n)
        elif key in (pygame.K_UP, pygame.K_w):
            self.sch_elevator_sel = int((sel - cols) % n)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.sch_elevator_sel = int((sel + cols) % n)
        self.sch_elevator_input = ""

    def _sch_interior_enter(self, sb: HardcoreSurvivalState._SpecialBuilding) -> None:
        self.inv_open = False
        self.rv_ui_open = False
        self.hr_elevator_ui_open = False
        self._gallery_open = False
        self.sch_interior = True
        self.sch_elevator_ui_open = False
        self.mount = None
        self.player.vel.update(0, 0)
        self.player.walk_phase *= 0.85
        self.rv.vel.update(0, 0)
        self.bike.vel.update(0, 0)

        self.sch_building = sb
        self.sch_max_floors = int(getattr(sb, "floors", 0) or int(self._SCH_INT_MAX_FLOORS_DEFAULT))
        if self.sch_max_floors <= 0:
            self.sch_max_floors = int(self._SCH_INT_MAX_FLOORS_DEFAULT)
        self.sch_world_return = pygame.Vector2(self.player.pos)
        self._sch_set_floor(1, spawn_at="door_inside")
        self._set_hint("学校：E互动楼梯/电梯 | Esc退出", seconds=1.6)

    def _sch_interior_exit(self) -> None:
        if not getattr(self, "sch_interior", False):
            return
        self.sch_interior = False
        self.sch_elevator_ui_open = False
        self.sch_building = None
        self.sch_floor = 1
        self._sch_int_set_layout(self._SCH_INT_LOBBY_LAYOUT)
        self.player.pos.update(self.sch_world_return)
        self._set_hint("离开学校", seconds=1.0)

    def _sch_interior_interact(self) -> None:
        if not getattr(self, "sch_interior", False):
            return
        tx, ty = self._sch_int_player_tile()
        candidates = [(tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]
        chosen: tuple[int, int, str] | None = None
        for cx, cy in candidates:
            ch = self._sch_int_char_at(cx, cy)
            if ch in ("D", "E", "^", "v"):
                chosen = (int(cx), int(cy), str(ch))
                break
        if chosen is None:
            self._set_hint("这里没有可互动", seconds=1.0)
            return
        _cx, _cy, ch = chosen
        if ch == "E":
            self._sch_elevator_open()
            return
        if ch == "D":
            self._sch_interior_exit()
            return

        floor = int(getattr(self, "sch_floor", 1))
        max_f = int(max(1, int(getattr(self, "sch_max_floors", int(self._SCH_INT_MAX_FLOORS_DEFAULT)))))
        if ch == "^":
            if floor >= max_f:
                self._set_hint("已经是顶层", seconds=1.2)
                return
            self._sch_set_floor(int(floor + 1), spawn_at="stairs_down")
            self._set_hint(f"上楼：{int(floor + 1)}F", seconds=0.9)
            return
        if ch == "v":
            if floor <= 1:
                self._set_hint("已经是一楼", seconds=1.2)
                return
            self._sch_set_floor(int(floor - 1), spawn_at="stairs_up")
            self._set_hint(f"下楼：{int(floor - 1)}F", seconds=0.9)
            return
        self._set_hint("这里没有可互动", seconds=1.0)

    def _hr_int_set_layout(self, layout: list[str]) -> None:
        self.hr_layout = [str(r) for r in layout]

    def _hr_int_find(self, ch: str, *, layout: list[str] | None = None) -> tuple[int, int] | None:
        ch = str(ch)[:1]
        rows = self.hr_layout if layout is None else layout
        for y, row in enumerate(rows):
            x = str(row).find(ch)
            if x >= 0:
                return int(x), int(y)
        return None

    def _hr_int_char_at(self, x: int, y: int) -> str:
        x = int(x)
        y = int(y)
        if not (0 <= x < int(self._HR_INT_W) and 0 <= y < int(self._HR_INT_H)):
            return "W"
        row = self.hr_layout[y] if 0 <= y < len(self.hr_layout) else ""
        if 0 <= x < len(row):
            return row[x]
        return "W"

    def _hr_int_solid_tile(self, ch: str) -> bool:
        ch = str(ch)[:1]
        return ch not in (".", "D", "E", "A", "H", "B")

    def _hr_int_move_box(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float, *, w: int, h: int) -> pygame.Vector2:
        tile = int(self._HR_INT_TILE_SIZE)
        rect = pygame.Rect(int(round(pos.x - w / 2)), int(round(pos.y - h / 2)), int(w), int(h))
        dx = float(vel.x * dt)
        dy = float(vel.y * dt)

        def collide(r: pygame.Rect) -> list[pygame.Rect]:
            left = int(math.floor(r.left / tile))
            right = int(math.floor((r.right - 1) / tile))
            top = int(math.floor(r.top / tile))
            bottom = int(math.floor((r.bottom - 1) / tile))
            hits: list[pygame.Rect] = []
            for ty in range(top, bottom + 1):
                for tx in range(left, right + 1):
                    ch = self._hr_int_char_at(tx, ty)
                    if not self._hr_int_solid_tile(ch):
                        continue
                    tr = pygame.Rect(tx * tile, ty * tile, tile, tile)
                    if r.colliderect(tr):
                        hits.append(tr)
            return hits

        if dx != 0.0:
            rect.x += int(round(dx))
            for hit in collide(rect):
                if dx > 0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right

        if dy != 0.0:
            rect.y += int(round(dy))
            for hit in collide(rect):
                if dy > 0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom

        pad = 3
        half_w = float(w) / 2.0
        half_h = float(h) / 2.0
        min_x = tile + half_w + pad
        min_y = tile + half_h + pad
        max_x = int(self._HR_INT_W) * tile - tile - half_w - pad
        max_y = int(self._HR_INT_H) * tile - tile - half_h - pad
        rect.centerx = int(clamp(rect.centerx, min_x, max_x))
        rect.centery = int(clamp(rect.centery, min_y, max_y))

        return pygame.Vector2(rect.centerx, rect.centery)

    def _hr_int_player_tile(self) -> tuple[int, int]:
        tile = int(self._HR_INT_TILE_SIZE)
        tx = int(math.floor(float(self.hr_int_pos.x) / tile))
        ty = int(math.floor(float(self.hr_int_pos.y) / tile))
        return tx, ty

    def _hr_edit_rebuild_blocks(self) -> None:
        if not getattr(self, "hr_interior", False):
            self.hr_edit_blocks = []
            return
        if str(getattr(self, "hr_mode", "lobby")) != "home":
            self.hr_edit_blocks = []
            return
        layout = getattr(self, "hr_layout", None)
        if not isinstance(layout, list):
            self.hr_edit_blocks = []
            return

        movable = {"B", "S", "F", "K", "T", "C"}
        w = int(self._HR_INT_W)
        h = int(self._HR_INT_H)
        seen: set[tuple[int, int]] = set()
        blocks: list[tuple[str, list[tuple[int, int]]]] = []

        for y in range(h):
            for x in range(w):
                ch = str(self._hr_int_char_at(x, y))[:1]
                if ch not in movable:
                    continue
                if (x, y) in seen:
                    continue
                cells: list[tuple[int, int]] = []
                stack = [(int(x), int(y))]
                while stack:
                    cx, cy = stack.pop()
                    cx = int(cx)
                    cy = int(cy)
                    if (cx, cy) in seen:
                        continue
                    if not (0 <= cx < w and 0 <= cy < h):
                        continue
                    if str(self._hr_int_char_at(cx, cy))[:1] != ch:
                        continue
                    seen.add((cx, cy))
                    cells.append((cx, cy))
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                if cells:
                    blocks.append((ch, cells))

        def sort_key(b: tuple[str, list[tuple[int, int]]]) -> tuple:
            ch, cells = b
            miny = min(int(p[1]) for p in cells)
            minx = min(int(p[0]) for p in cells)
            return (ch, miny, minx, len(cells))

        blocks.sort(key=sort_key)
        self.hr_edit_blocks = blocks
        self.hr_edit_index = int(clamp(int(getattr(self, "hr_edit_index", 0)), 0, max(0, len(blocks) - 1)))

    def _hr_toggle_edit_mode(self) -> None:
        if not getattr(self, "hr_interior", False):
            return
        if str(getattr(self, "hr_mode", "lobby")) != "home":
            return
        self.hr_edit_mode = not bool(getattr(self, "hr_edit_mode", False))
        self.hr_edit_dragging = False
        if self.hr_edit_mode:
            self._hr_edit_rebuild_blocks()
            if not getattr(self, "hr_edit_blocks", None):
                self.hr_edit_mode = False
                self._set_hint("没有可挪动的家具", seconds=1.0)
                return
            self.hr_edit_index = int(
                clamp(int(getattr(self, "hr_edit_index", 0)), 0, len(self.hr_edit_blocks) - 1)
            )
            self._set_hint("摆放模式：Tab 选中 / 方向键移动 / 左键拖拽 / R 退出", seconds=1.8)
        else:
            self._set_hint("退出摆放模式", seconds=1.0)

    def _hr_edit_cycle(self, delta: int = 1) -> None:
        blocks = getattr(self, "hr_edit_blocks", None)
        if not isinstance(blocks, list) or not blocks:
            return
        self.hr_edit_index = (int(getattr(self, "hr_edit_index", 0)) + int(delta)) % len(blocks)

    def _hr_edit_move(self, dx: int, dy: int) -> None:
        if not bool(getattr(self, "hr_edit_mode", False)):
            return
        if str(getattr(self, "hr_mode", "lobby")) != "home":
            return
        blocks = getattr(self, "hr_edit_blocks", None)
        if not isinstance(blocks, list) or not blocks:
            return
        idx = int(getattr(self, "hr_edit_index", 0)) % len(blocks)
        ch, cells = blocks[idx]
        w = int(self._HR_INT_W)
        h = int(self._HR_INT_H)
        dx = int(dx)
        dy = int(dy)
        if dx == 0 and dy == 0:
            return

        cell_set = set((int(x), int(y)) for x, y in cells)
        new_cells = [(int(x) + dx, int(y) + dy) for x, y in cell_set]

        for nx, ny in new_cells:
            if not (0 <= int(nx) < w and 0 <= int(ny) < h):
                return
            if (int(nx), int(ny)) in cell_set:
                continue
            dst = str(self._hr_int_char_at(nx, ny))[:1]
            if dst != ".":
                return

        rows = [list(str(r)) for r in getattr(self, "hr_layout", [])]
        if len(rows) < h:
            rows.extend([list("." * w) for _ in range(h - len(rows))])
        for oy in range(min(len(rows), h)):
            if len(rows[oy]) < w:
                rows[oy].extend(list("." * (w - len(rows[oy]))))

        for ox, oy in cell_set:
            if 0 <= int(oy) < h and 0 <= int(ox) < w:
                rows[int(oy)][int(ox)] = "."
        for nx, ny in new_cells:
            rows[int(ny)][int(nx)] = str(ch)[:1]

        self.hr_layout = ["".join(r[:w]) for r in rows[:h]]
        self.hr_home_layout = list(self.hr_layout)
        blocks[idx] = (str(ch)[:1], [(int(x), int(y)) for x, y in new_cells])

    def _hr_edit_handle_mouse(self, event: pygame.event.Event) -> bool:
        if not bool(getattr(self, "hr_edit_mode", False)):
            return False
        if not getattr(self, "hr_interior", False):
            return False
        if str(getattr(self, "hr_mode", "lobby")) != "home":
            return False
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        if not hasattr(event, "pos"):
            return False
        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return False

        tile = int(self._HR_INT_TILE_SIZE)
        map_w = int(self._HR_INT_W) * tile
        map_h = int(self._HR_INT_H) * tile
        map_x = (INTERNAL_W - map_w) // 2
        map_y = 38
        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)

        movable = {"B", "S", "F", "K", "T", "C"}

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            if not map_rect.collidepoint(internal):
                return False
            ix = int((int(internal[0]) - int(map_x)) // max(1, int(tile)))
            iy = int((int(internal[1]) - int(map_y)) // max(1, int(tile)))
            ch = str(self._hr_int_char_at(ix, iy))[:1]
            if ch not in movable:
                return False
            blocks = getattr(self, "hr_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                self._hr_edit_rebuild_blocks()
                blocks = getattr(self, "hr_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                return False

            hit_i: int | None = None
            for i, (bch, cells) in enumerate(blocks):
                if str(bch)[:1] != ch:
                    continue
                for cx, cy in cells:
                    if int(cx) == int(ix) and int(cy) == int(iy):
                        hit_i = int(i)
                        break
                if hit_i is not None:
                    break
            if hit_i is None:
                return False

            self.hr_edit_index = int(hit_i)
            _bch, cells = blocks[hit_i]
            min_x = min(int(p[0]) for p in cells)
            min_y = min(int(p[1]) for p in cells)
            self.hr_edit_dragging = True
            self.hr_edit_drag_offset = (int(ix - min_x), int(iy - min_y))
            return True

        if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", 0) == 1:
            if bool(getattr(self, "hr_edit_dragging", False)):
                self.hr_edit_dragging = False
                return True
            return False

        if event.type == pygame.MOUSEMOTION:
            if not bool(getattr(self, "hr_edit_dragging", False)):
                return False
            if not map_rect.collidepoint(internal):
                return True
            ix = int((int(internal[0]) - int(map_x)) // max(1, int(tile)))
            iy = int((int(internal[1]) - int(map_y)) // max(1, int(tile)))
            off = getattr(self, "hr_edit_drag_offset", (0, 0))
            offx, offy = int(off[0]), int(off[1])
            target_min_x = int(ix - offx)
            target_min_y = int(iy - offy)

            blocks = getattr(self, "hr_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                return True
            idx = int(getattr(self, "hr_edit_index", 0)) % len(blocks)
            _ch, cells = blocks[idx]
            cur_min_x = min(int(p[0]) for p in cells)
            cur_min_y = min(int(p[1]) for p in cells)
            dx = int(target_min_x - cur_min_x)
            dy = int(target_min_y - cur_min_y)
            if dx != 0 or dy != 0:
                self._hr_edit_move(dx, dy)
            return True

        return False

    def _hr_make_hall_layout(self, floor: int) -> list[str]:
        base = [str(r) for r in self._HR_INT_HALL_BASE]
        home_floor = int(getattr(self, "hr_home_floor", -1))
        if int(floor) != int(home_floor):
            return base
        doors = list(getattr(self, "_HR_INT_APT_DOORS", []))
        if not doors:
            return base
        idx = int(getattr(self, "hr_home_door_index", 0)) % len(doors)
        hx, hy = doors[idx]
        rows = [list(r) for r in base]
        if 0 <= int(hy) < len(rows) and 0 <= int(hx) < len(rows[int(hy)]):
            rows[int(hy)][int(hx)] = "H"
        return ["".join(r) for r in rows]

    def _hr_set_floor(self, floor: int) -> None:
        max_f = int(max(1, int(getattr(self, "hr_max_floors", int(self._HR_INT_MAX_FLOORS_DEFAULT)))))
        floor = int(clamp(int(floor), 0, max_f))
        self.hr_floor = int(floor)
        if floor <= 0:
            self.hr_mode = "lobby"
            self._hr_int_set_layout(self._HR_INT_LOBBY_LAYOUT)
        else:
            self.hr_mode = "hall"
            self._hr_int_set_layout(self._hr_make_hall_layout(floor))

        elev = self._hr_int_find("E")
        tile = int(self._HR_INT_TILE_SIZE)
        if elev is None:
            self.hr_int_pos = pygame.Vector2((self._HR_INT_W / 2.0) * tile, (self._HR_INT_H / 2.0) * tile)
        else:
            ex, ey = elev
            self.hr_int_pos = pygame.Vector2((ex + 0.5) * tile, (ey + 0.5) * tile)
        self.hr_int_vel = pygame.Vector2(0, 0)
        self.hr_int_facing = pygame.Vector2(0, 1)
        self.hr_int_walk_phase = 0.0

    def _hr_elevator_open(self) -> None:
        if not getattr(self, "hr_interior", False):
            return
        max_f = int(max(1, int(getattr(self, "hr_max_floors", int(self._HR_INT_MAX_FLOORS_DEFAULT)))))
        options = list(range(max_f, 0, -1)) + [0]
        self.hr_elevator_options = options
        cur = int(getattr(self, "hr_floor", 0))
        try:
            self.hr_elevator_sel = int(options.index(cur))
        except ValueError:
            self.hr_elevator_sel = 0
        self.hr_elevator_input = ""
        self.hr_elevator_ui_open = True
        self._set_hint("电梯：方向键选择 | PgUp/PgDn翻页 | 数字直达 | Enter确认", seconds=1.8)

    def _key_digit(self, key: int) -> str | None:
        key = int(key)
        if int(pygame.K_0) <= key <= int(pygame.K_9):
            return chr(key)
        if int(pygame.K_KP0) <= key <= int(pygame.K_KP9):
            return str(int(key) - int(pygame.K_KP0))
        return None

    def _handle_hr_elevator_ui_wheel(self, event: pygame.event.Event) -> None:
        if not getattr(self, "hr_elevator_ui_open", False):
            return
        options: list[int] = list(getattr(self, "hr_elevator_options", []))
        if not options:
            options = [0]
            self.hr_elevator_options = options
        n = len(options)
        cols = max(1, int(getattr(self, "hr_elevator_cols", 4)))
        rows_per_page = 4
        page_size = max(1, cols * rows_per_page)
        step = cols
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            step = page_size
        dy = int(getattr(event, "y", 0))
        if dy == 0:
            return
        sel = int(getattr(self, "hr_elevator_sel", 0)) % n
        sel = (sel - step) % n if dy > 0 else (sel + step) % n
        self.hr_elevator_sel = int(sel)
        self.hr_elevator_input = ""

    def _handle_hr_elevator_ui_mouse(self, event: pygame.event.Event) -> None:
        if not getattr(self, "hr_elevator_ui_open", False):
            return

        et = int(getattr(event, "type", 0))
        button = int(getattr(event, "button", 0))
        if et == int(pygame.MOUSEBUTTONDOWN) and button == 3:
            self.hr_elevator_ui_open = False
            self.hr_elevator_input = ""
            return

        if et not in (int(pygame.MOUSEMOTION), int(pygame.MOUSEBUTTONDOWN)):
            return
        if et == int(pygame.MOUSEBUTTONDOWN) and button != 1:
            return

        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return
        mx, my = int(internal[0]), int(internal[1])

        options: list[int] = list(getattr(self, "hr_elevator_options", []))
        if not options:
            options = [0]
            self.hr_elevator_options = options
        cols = max(1, int(getattr(self, "hr_elevator_cols", 4)))
        rows_per_page = 4
        n = len(options)
        page_size = max(1, cols * rows_per_page)
        sel_i = int(getattr(self, "hr_elevator_sel", 0)) % max(1, n)
        page = int(sel_i // page_size)
        start = int(page * page_size)
        end = int(min(int(n), int(start + page_size)))
        page_opts = list(options[start:end])
        rows = int(max(1, int(math.ceil(float(len(page_opts)) / float(cols)))))

        btn_w = 56
        btn_h = 28
        gap = 8
        grid_w = cols * btn_w + (cols - 1) * gap
        grid_h = rows * btn_h + (rows - 1) * gap
        panel = pygame.Rect(0, 0, grid_w + 40, grid_h + 98)
        panel.center = (INTERNAL_W // 2, INTERNAL_H // 2)

        x0 = panel.centerx - grid_w // 2
        y0 = panel.top + 66
        hit_i: int | None = None
        hit_floor: int | None = None
        for local_i, f in enumerate(page_opts):
            r = int(local_i) // cols
            c = int(local_i) % cols
            x = x0 + c * (btn_w + gap)
            y = y0 + r * (btn_h + gap)
            br = pygame.Rect(x, y, btn_w, btn_h)
            if br.collidepoint(mx, my):
                hit_i = int(start) + int(local_i)
                hit_floor = int(f)
                break

        if hit_i is None or hit_floor is None:
            return

        self.hr_elevator_sel = int(hit_i)
        self.hr_elevator_input = ""
        if et == int(pygame.MOUSEBUTTONDOWN) and button == 1:
            self.hr_elevator_ui_open = False
            self._hr_set_floor(int(hit_floor))

    def _handle_hr_elevator_ui_key(self, key: int) -> None:
        if not getattr(self, "hr_elevator_ui_open", False):
            return
        options: list[int] = list(getattr(self, "hr_elevator_options", []))
        if not options:
            options = [0]
            self.hr_elevator_options = options
        n = len(options)
        cols = max(1, int(getattr(self, "hr_elevator_cols", 4)))
        rows_per_page = 4
        page_size = max(1, cols * rows_per_page)
        sel = int(getattr(self, "hr_elevator_sel", 0)) % n

        digit = self._key_digit(int(key))
        if digit is not None:
            cur = str(getattr(self, "hr_elevator_input", ""))
            if len(cur) < 3:
                self.hr_elevator_input = cur + digit
            return
        if key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            cur = str(getattr(self, "hr_elevator_input", ""))
            self.hr_elevator_input = cur[:-1]
            return

        if key in (pygame.K_ESCAPE,):
            self.hr_elevator_ui_open = False
            self.hr_elevator_input = ""
            return
        if key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_e):
            self.hr_elevator_ui_open = False
            typed = str(getattr(self, "hr_elevator_input", "")).strip()
            self.hr_elevator_input = ""
            if typed:
                try:
                    target = int(typed)
                except ValueError:
                    target = int(options[sel])
                self._hr_set_floor(int(target))
            else:
                self._hr_set_floor(int(options[sel]))
            return

        if key in (pygame.K_PAGEUP,):
            self.hr_elevator_sel = int((sel - page_size) % n)
            self.hr_elevator_input = ""
            return
        if key in (pygame.K_PAGEDOWN,):
            self.hr_elevator_sel = int((sel + page_size) % n)
            self.hr_elevator_input = ""
            return
        if key in (pygame.K_HOME,):
            self.hr_elevator_sel = 0
            self.hr_elevator_input = ""
            return
        if key in (pygame.K_END,):
            self.hr_elevator_sel = int(n - 1)
            self.hr_elevator_input = ""
            return

        if key in (pygame.K_LEFT, pygame.K_a):
            self.hr_elevator_sel = int((sel - 1) % n)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.hr_elevator_sel = int((sel + 1) % n)
        elif key in (pygame.K_UP, pygame.K_w):
            self.hr_elevator_sel = int((sel - cols) % n)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.hr_elevator_sel = int((sel + cols) % n)
        self.hr_elevator_input = ""

    def _hr_enter_home(self) -> None:
        self.hr_hall_pos_before_home = pygame.Vector2(self.hr_int_pos)
        self.hr_mode = "home"
        self.hr_edit_mode = False
        self.hr_edit_dragging = False
        self.home_ui_open = False
        self.home_ui_open_block = None
        if not isinstance(getattr(self, "hr_home_layout", None), list):
            self.hr_home_layout = [str(r) for r in self._HR_INT_HOME_LAYOUT]
        self._hr_int_set_layout(getattr(self, "hr_home_layout", self._HR_INT_HOME_LAYOUT))

        door = self._hr_int_find("D")
        tile = int(self._HR_INT_TILE_SIZE)
        if door is None:
            self.hr_int_pos = pygame.Vector2((self._HR_INT_W / 2.0) * tile, (self._HR_INT_H / 2.0) * tile)
        else:
            dx, dy = door
            spawn = (dx - 1, dy) if dx >= int(self._HR_INT_W) - 1 else (dx + 1, dy)
            sx, sy = spawn
            if self._hr_int_solid_tile(self._hr_int_char_at(sx, sy)):
                sx, sy = int(dx), int(dy)
            self.hr_int_pos = pygame.Vector2((sx + 0.5) * tile, (sy + 0.5) * tile)
        self.hr_int_vel = pygame.Vector2(0, 0)
        self.hr_int_walk_phase = 0.0
        self._set_hint("进入你的家：E互动床恢复体力/心态", seconds=1.6)

    def _hr_leave_home(self) -> None:
        floor = int(getattr(self, "hr_floor", 1))
        self._clear_player_pose()
        self.hr_mode = "hall"
        self.hr_edit_mode = False
        self.hr_edit_dragging = False
        self.home_ui_open = False
        self.home_ui_open_block = None
        self._hr_int_set_layout(self._hr_make_hall_layout(floor))
        saved = getattr(self, "hr_hall_pos_before_home", None)
        if isinstance(saved, pygame.Vector2):
            self.hr_int_pos = pygame.Vector2(saved)
        self.hr_int_vel = pygame.Vector2(0, 0)
        self.hr_int_walk_phase = 0.0
        self._set_hint("离开房间", seconds=0.9)

    def _hr_interior_enter(self, sb: HardcoreSurvivalState._SpecialBuilding) -> None:
        self.inv_open = False
        self.rv_ui_open = False
        self._gallery_open = False
        self.hr_interior = True
        self.hr_elevator_ui_open = False
        self.hr_edit_mode = False
        self.hr_edit_dragging = False
        self.home_ui_open = False
        self.home_ui_open_block = None
        self.hr_edit_blocks: list[tuple[str, list[tuple[int, int]]]] = []
        self.hr_edit_index = 0
        self.mount = None
        self.player.vel.update(0, 0)
        self.player.walk_phase *= 0.85
        self.rv.vel.update(0, 0)
        self.bike.vel.update(0, 0)

        self.hr_building = sb
        self.hr_max_floors = int(getattr(sb, "floors", 0) or int(self._HR_INT_MAX_FLOORS_DEFAULT))
        if self.hr_max_floors <= 0:
            self.hr_max_floors = int(self._HR_INT_MAX_FLOORS_DEFAULT)

        home_door = getattr(self, "home_highrise_door", None)
        is_home = bool(home_door is not None and home_door in set(getattr(sb, "door_tiles", ())))
        if is_home:
            self.hr_home_floor = int(getattr(self, "home_highrise_floor", 0))
            self.hr_home_door_index = int(getattr(self, "home_highrise_door_index", 0))
        else:
            self.hr_home_floor = -1
            self.hr_home_door_index = 0

        self.hr_world_return = pygame.Vector2(self.player.pos)
        self._hr_set_floor(0)
        # Walk into the lobby, then auto-walk to the elevator and open the floor UI.
        self.hr_auto_walk_to_elevator = True
        self.hr_auto_walk_delay = 0.12

        door = self._hr_int_find("D")
        if door is not None:
            dx, dy = door
            tile = int(self._HR_INT_TILE_SIZE)
            spawn = (dx - 1, dy) if dx >= int(self._HR_INT_W) - 1 else (dx + 1, dy)
            sx, sy = spawn
            if self._hr_int_solid_tile(self._hr_int_char_at(sx, sy)):
                sx, sy = int(dx), int(dy)
            self.hr_int_pos = pygame.Vector2((sx + 0.5) * tile, (sy + 0.5) * tile)

        if is_home and int(self.hr_home_floor) > 0:
            home_room = str(getattr(self, "home_highrise_room", "")).strip()
            if home_room:
                self._set_hint(f"高层住宅：你的家 {home_room}", seconds=2.0)
            else:
                self._set_hint(f"高层住宅：你的家在 {int(self.hr_home_floor)} 层", seconds=2.0)
        else:
            self._set_hint("高层住宅：E进入电梯选择楼层", seconds=1.6)   

    def _hr_interior_exit(self) -> None:
        if not getattr(self, "hr_interior", False):
            return
        self._clear_player_pose()
        self.hr_interior = False
        self.hr_elevator_ui_open = False
        self.hr_edit_mode = False
        self.hr_edit_dragging = False
        self.home_ui_open = False
        self.home_ui_open_block = None
        self.hr_building = None
        self.hr_mode = "lobby"
        self._hr_int_set_layout(self._HR_INT_LOBBY_LAYOUT)
        self.player.pos.update(self.hr_world_return)
        self._set_hint("离开高层住宅", seconds=1.0)

    def _hr_interior_interact(self) -> None:
        if not getattr(self, "hr_interior", False):
            return
        mode = str(getattr(self, "hr_mode", "lobby"))
        tx, ty = self._hr_int_player_tile()
        candidates = [(tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]
        chosen: tuple[int, int, str] | None = None
        interact = {"D", "E", "H", "A", "B"}
        if mode == "home":
            interact.update({"S", "F", "C"})
        for cx, cy in candidates:
            ch = self._hr_int_char_at(cx, cy)
            if ch in interact:
                chosen = (cx, cy, ch)
                break
        if chosen is None:
            self._set_hint("这里没有可互动", seconds=1.0)
            return
        _cx, _cy, ch = chosen
        if ch == "E":
            self._hr_elevator_open()
            return
        if ch == "D":
            if mode == "home":
                self._hr_leave_home()
                return
            self._hr_interior_exit()
            return
        if ch == "S" and mode == "home":
            w = int(self._HR_INT_W)
            h = int(self._HR_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._hr_int_char_at(cx, cy))[:1] != "S":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                self._home_ui_open(open_block=(int(min_x), int(min_y), int(max_x), int(max_y)), storage_kind="cabinet")
            else:
                self._home_ui_open(open_block=(int(_cx), int(_cy), int(_cx), int(_cy)), storage_kind="cabinet")
            return
        if ch == "F" and mode == "home":
            w = int(self._HR_INT_W)
            h = int(self._HR_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._hr_int_char_at(cx, cy))[:1] != "F":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                self._home_ui_open(open_block=(int(min_x), int(min_y), int(max_x), int(max_y)), storage_kind="fridge")
            else:
                self._home_ui_open(open_block=(int(_cx), int(_cy), int(_cx), int(_cy)), storage_kind="fridge")
            return
        if ch == "C" and mode == "home":
            if str(getattr(self, "player_pose", "")) == "sit" and str(getattr(self, "player_pose_space", "")) == "hr":
                self._clear_player_pose()
                self._set_hint("起身", seconds=0.8)
                return
            w = int(self._HR_INT_W)
            h = int(self._HR_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._hr_int_char_at(cx, cy))[:1] != "C":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            tile = int(self._HR_INT_TILE_SIZE)
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                ax = (float(min_x + max_x + 1) * float(tile)) * 0.5
                ay = (float(min_y + max_y + 1) * float(tile)) * 0.5
            else:
                ax = (float(_cx) + 0.5) * float(tile)
                ay = (float(_cy) + 0.5) * float(tile)
            self._set_player_pose("sit", space="hr", anchor=(ax, ay), seconds=0.0)
            self._set_hint("坐下休息：移动键起身", seconds=1.4)
            return
        if ch == "B":
            self.player.stamina = 100.0
            self.player.morale = float(clamp(self.player.morale + 15.0, 0.0, 100.0))
            w = int(self._HR_INT_W)
            h = int(self._HR_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._hr_int_char_at(cx, cy))[:1] != "B":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            tile = int(self._HR_INT_TILE_SIZE)
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                ax = (float(min_x + max_x + 1) * float(tile)) * 0.5
                ay = (float(min_y + max_y + 1) * float(tile)) * 0.5
            else:
                ax = (float(_cx) + 0.5) * float(tile)
                ay = (float(_cy) + 0.5) * float(tile)
            self._set_player_pose("sleep", space="hr", anchor=(ax, ay), seconds=2.6)
            self._set_hint("躺床休息：体力恢复 / 心态提升", seconds=1.4)
            return
        if ch == "H":
            if mode == "hall" and int(getattr(self, "hr_floor", 0)) == int(getattr(self, "hr_home_floor", -1)):
                self._hr_enter_home()
                return
            self._set_hint("这不是你的家", seconds=1.2)
            return
        if ch == "A":
            self._set_hint("门锁住了", seconds=1.1)
            return
        self._set_hint("这里没有可互动", seconds=1.0)

    def _rv_int_char_at(self, x: int, y: int) -> str:
        x = int(x)
        y = int(y)
        if not (0 <= x < int(self._RV_INT_W) and 0 <= y < int(self._RV_INT_H)):
            return "W"
        layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)
        row = layout[y]
        if 0 <= x < len(row):
            return row[x]
        return "W"

    def _rv_int_solid_tile(self, ch: str) -> bool:
        ch = str(ch)[:1]
        return ch not in (".", "D", "B")

    def _rv_edit_rebuild_blocks(self) -> None:
        layout = getattr(self, "rv_int_layout", None)
        if not isinstance(layout, list):
            self.rv_edit_blocks = []
            return
        movable = {"B", "S", "K", "H", "T", "C"}
        w = int(self._RV_INT_W)
        h = int(self._RV_INT_H)
        seen: set[tuple[int, int]] = set()
        blocks: list[tuple[str, list[tuple[int, int]]]] = []

        for y in range(h):
            for x in range(w):
                ch = self._rv_int_char_at(x, y)
                if ch not in movable:
                    continue
                if (x, y) in seen:
                    continue
                cells: list[tuple[int, int]] = []
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    cx = int(cx)
                    cy = int(cy)
                    if (cx, cy) in seen:
                        continue
                    if not (0 <= cx < w and 0 <= cy < h):
                        continue
                    if self._rv_int_char_at(cx, cy) != ch:
                        continue
                    seen.add((cx, cy))
                    cells.append((cx, cy))
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                if cells:
                    blocks.append((ch, cells))

        def sort_key(b: tuple[str, list[tuple[int, int]]]) -> tuple:
            ch, cells = b
            miny = min(p[1] for p in cells)
            minx = min(p[0] for p in cells)
            return (ch, miny, minx, len(cells))

        blocks.sort(key=sort_key)
        self.rv_edit_blocks = blocks
        self.rv_edit_index = int(clamp(int(getattr(self, "rv_edit_index", 0)), 0, max(0, len(blocks) - 1)))

    def _rv_toggle_edit_mode(self) -> None:
        if not getattr(self, "rv_interior", False):
            return
        self.rv_edit_mode = not bool(getattr(self, "rv_edit_mode", False))
        self.rv_edit_dragging = False
        if self.rv_edit_mode:
            self._rv_edit_rebuild_blocks()
            if not getattr(self, "rv_edit_blocks", None):
                self.rv_edit_mode = False
                self._set_hint("没有可挪动的家具", seconds=1.0)
                return
            self.rv_edit_index = int(clamp(int(getattr(self, "rv_edit_index", 0)), 0, len(self.rv_edit_blocks) - 1))
            self._set_hint("摆放模式：Tab选中 方向键移动  R退出", seconds=1.6)
        else:
            self._set_hint("退出摆放模式", seconds=1.0)

    def _rv_edit_cycle(self, delta: int = 1) -> None:
        blocks = getattr(self, "rv_edit_blocks", None)
        if not isinstance(blocks, list) or not blocks:
            return
        self.rv_edit_index = (int(getattr(self, "rv_edit_index", 0)) + int(delta)) % len(blocks)

    def _rv_edit_move(self, dx: int, dy: int) -> None:
        if not bool(getattr(self, "rv_edit_mode", False)):
            return
        layout = getattr(self, "rv_int_layout", None)
        blocks = getattr(self, "rv_edit_blocks", None)
        if not isinstance(layout, list) or not isinstance(blocks, list) or not blocks:
            return
        idx = int(getattr(self, "rv_edit_index", 0)) % len(blocks)
        ch, cells = blocks[idx]
        w = int(self._RV_INT_W)
        h = int(self._RV_INT_H)
        dx = int(dx)
        dy = int(dy)
        if dx == 0 and dy == 0:
            return

        cell_set = set((int(x), int(y)) for x, y in cells)
        new_cells = [(int(x) + dx, int(y) + dy) for x, y in cell_set]

        # Validate.
        for nx, ny in new_cells:
            if not (0 <= int(nx) < w and 0 <= int(ny) < h):
                return
            if (int(nx), int(ny)) in cell_set:
                continue
            dst = self._rv_int_char_at(nx, ny)
            if dst != ".":
                return

        # Apply.
        for ox, oy in cell_set:
            layout[int(oy)][int(ox)] = "."
        for nx, ny in new_cells:
            layout[int(ny)][int(nx)] = ch

        blocks[idx] = (ch, [(int(x), int(y)) for x, y in new_cells])

    def _rv_edit_handle_mouse(self, event: pygame.event.Event) -> bool:
        if not bool(getattr(self, "rv_edit_mode", False)):
            return False
        if not getattr(self, "rv_interior", False):
            return False
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        if not hasattr(event, "pos"):
            return False
        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return False

        tile = int(self._RV_INT_TILE_SIZE)
        map_w = int(self._RV_INT_W) * tile
        map_h = int(self._RV_INT_H) * tile
        map_x = (INTERNAL_W - map_w) // 2
        map_y = 92
        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)

        movable = {"B", "S", "K", "H", "T", "C"}

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            if not map_rect.collidepoint(internal):
                return False
            ix = int((int(internal[0]) - int(map_x)) // max(1, int(tile)))
            iy = int((int(internal[1]) - int(map_y)) // max(1, int(tile)))
            ch = str(self._rv_int_char_at(ix, iy))[:1]
            if ch not in movable:
                return False
            blocks = getattr(self, "rv_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                self._rv_edit_rebuild_blocks()
                blocks = getattr(self, "rv_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                return False

            hit_i: int | None = None
            for i, (bch, cells) in enumerate(blocks):
                if str(bch)[:1] != ch:
                    continue
                for cx, cy in cells:
                    if int(cx) == int(ix) and int(cy) == int(iy):
                        hit_i = int(i)
                        break
                if hit_i is not None:
                    break
            if hit_i is None:
                return False

            self.rv_edit_index = int(hit_i)
            _bch, cells = blocks[hit_i]
            min_x = min(int(p[0]) for p in cells)
            min_y = min(int(p[1]) for p in cells)
            self.rv_edit_dragging = True
            self.rv_edit_drag_offset = (int(ix - min_x), int(iy - min_y))
            return True

        if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", 0) == 1:
            if bool(getattr(self, "rv_edit_dragging", False)):
                self.rv_edit_dragging = False
                return True
            return False

        if event.type == pygame.MOUSEMOTION:
            if not bool(getattr(self, "rv_edit_dragging", False)):
                return False
            if not map_rect.collidepoint(internal):
                return True
            ix = int((int(internal[0]) - int(map_x)) // max(1, int(tile)))
            iy = int((int(internal[1]) - int(map_y)) // max(1, int(tile)))
            off = getattr(self, "rv_edit_drag_offset", (0, 0))
            offx, offy = int(off[0]), int(off[1])
            target_min_x = int(ix - offx)
            target_min_y = int(iy - offy)

            blocks = getattr(self, "rv_edit_blocks", None)
            if not isinstance(blocks, list) or not blocks:
                return True
            idx = int(getattr(self, "rv_edit_index", 0)) % len(blocks)
            _ch, cells = blocks[idx]
            cur_min_x = min(int(p[0]) for p in cells)
            cur_min_y = min(int(p[1]) for p in cells)
            dx = int(target_min_x - cur_min_x)
            dy = int(target_min_y - cur_min_y)
            if dx != 0 or dy != 0:
                self._rv_edit_move(dx, dy)
            return True

        return False

    def _rv_int_move_box(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float, *, w: int, h: int) -> pygame.Vector2:
        tile = int(self._RV_INT_TILE_SIZE)
        rect = pygame.Rect(int(round(pos.x - w / 2)), int(round(pos.y - h / 2)), int(w), int(h))
        dx = float(vel.x * dt)
        dy = float(vel.y * dt)

        def collide(r: pygame.Rect) -> list[pygame.Rect]:
            left = int(math.floor(r.left / tile))
            right = int(math.floor((r.right - 1) / tile))
            top = int(math.floor(r.top / tile))
            bottom = int(math.floor((r.bottom - 1) / tile))
            hits: list[pygame.Rect] = []
            for ty in range(top, bottom + 1):
                for tx in range(left, right + 1):
                    ch = self._rv_int_char_at(tx, ty)
                    if not self._rv_int_solid_tile(ch):
                        continue
                    tr = pygame.Rect(tx * tile, ty * tile, tile, tile)
                    if r.colliderect(tr):
                        hits.append(tr)
            return hits

        if dx != 0.0:
            rect.x += int(round(dx))
            for hit in collide(rect):
                if dx > 0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right

        if dy != 0.0:
            rect.y += int(round(dy))
            for hit in collide(rect):
                if dy > 0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom

        pad = 3
        half_w = float(w) / 2.0
        half_h = float(h) / 2.0
        min_x = tile + half_w + pad
        min_y = tile + half_h + pad
        max_x = int(self._RV_INT_W) * tile - tile - half_w - pad
        max_y = int(self._RV_INT_H) * tile - tile - half_h - pad
        rect.centerx = int(clamp(rect.centerx, min_x, max_x))
        rect.centery = int(clamp(rect.centery, min_y, max_y))

        return pygame.Vector2(rect.centerx, rect.centery)

    def _rv_int_player_tile(self) -> tuple[int, int]:
        tile = int(self._RV_INT_TILE_SIZE)
        tx = int(math.floor(float(self.rv_int_pos.x) / tile))
        ty = int(math.floor(float(self.rv_int_pos.y) / tile))
        return tx, ty

    def _rv_interior_interact(self) -> None:
        if not getattr(self, "rv_interior", False):
            return
        tx, ty = self._rv_int_player_tile()
        candidates = [(tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]
        chosen: tuple[int, int, str] | None = None
        for cx, cy in candidates:
            ch = self._rv_int_char_at(cx, cy)
            if ch in ("D", "B", "S", "K", "H", "T", "C"):
                chosen = (cx, cy, ch)
                break
        if chosen is None:
            self._set_hint("这里没有可用互动", seconds=1.0)
            return
        _cx, _cy, ch = chosen
        if ch == "D":
            self._rv_interior_exit()
            return
        if ch == "B":
            self.player.stamina = 100.0
            self.player.morale = 100.0
            w = int(self._RV_INT_W)
            h = int(self._RV_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._rv_int_char_at(cx, cy))[:1] != "B":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            tile = int(self._RV_INT_TILE_SIZE)
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                ax = (float(min_x + max_x + 1) * float(tile)) * 0.5
                ay = (float(min_y + max_y + 1) * float(tile)) * 0.5
            else:
                ax = (float(_cx) + 0.5) * float(tile)
                ay = (float(_cy) + 0.5) * float(tile)
            self._set_player_pose("sleep", space="rv", anchor=(ax, ay), seconds=2.4)
            self._set_hint("躺床休息：体力/心态恢复", seconds=1.4)
            return
        if ch == "S":
            self._rv_ui_open()
            self.rv_ui_focus = "storage"
            self._set_rv_ui_status("储物柜：在背包/储物间按 Enter 转移", seconds=1.8)
            return
        if ch == "K":
            self._set_hint("厨房：后续加入烹饪/净水/热食加成", seconds=1.6)
            return
        if ch == "H":
            self._set_hint("工作台：后续加入制作/维修/枪械加工", seconds=1.6)
            return
        if ch == "C":
            if str(getattr(self, "player_pose", "")) == "sit" and str(getattr(self, "player_pose_space", "")) == "rv":
                self._clear_player_pose()
                self._set_hint("起身", seconds=0.8)
                return
            w = int(self._RV_INT_W)
            h = int(self._RV_INT_H)
            seen: set[tuple[int, int]] = set()
            stack = [(int(_cx), int(_cy))]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cx = int(cx)
                cy = int(cy)
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if str(self._rv_int_char_at(cx, cy))[:1] != "C":
                    continue
                seen.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            tile = int(self._RV_INT_TILE_SIZE)
            if cells:
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                ax = (float(min_x + max_x + 1) * float(tile)) * 0.5
                ay = (float(min_y + max_y + 1) * float(tile)) * 0.5
            else:
                ax = (float(_cx) + 0.5) * float(tile)
                ay = (float(_cy) + 0.5) * float(tile)
            self._set_player_pose("sit", space="rv", anchor=(ax, ay), seconds=0.0)
            self._set_hint("坐下休息：移动键起身", seconds=1.4)
            return
        if ch == "T":
            self._set_hint("工作台/驾驶舱：后续加入制作/维修/改装", seconds=1.5)
            return
        self._set_hint("这里没有可用互动", seconds=1.0)

    def _update_gun_timers(self, dt: float, *, allow_fire: bool) -> None:
        if self.gun is None:
            return
        self.gun.cooldown_left = max(0.0, float(self.gun.cooldown_left) - dt)
        if self.gun.reload_left > 0.0:
            self.gun.reload_left = max(0.0, float(self.gun.reload_left) - dt)
            if self.gun.reload_left <= 0.0:
                self._reload_lock_dir = None
                gun_def = self._GUNS.get(self.gun.gun_id)
                if gun_def is not None:
                    need = int(gun_def.mag_size) - int(self.gun.mag)
                    if need > 0:
                        got = self.inventory.remove(gun_def.ammo_item, need)
                        self.gun.mag = int(self.gun.mag) + int(got)
            return
        if allow_fire and self.mount is None and pygame.mouse.get_pressed()[0]:
            self._fire()

    def _interior_floor_surface(self, w: int, h: int, *, kind: str) -> pygame.Surface:
        w = int(w)
        h = int(h)
        kind = str(kind)
        key = (kind, w, h)

        cache = getattr(self, "_int_floor_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._int_floor_cache = cache
        cached = cache.get(key)
        if isinstance(cached, pygame.Surface):
            return cached

        def _tint(c: tuple[int, int, int], add: tuple[int, int, int]) -> tuple[int, int, int]:
            return (
                int(clamp(int(c[0]) + int(add[0]), 0, 255)),
                int(clamp(int(c[1]) + int(add[1]), 0, 255)),
                int(clamp(int(c[2]) + int(add[2]), 0, 255)),
            )

        surf = pygame.Surface((w, h))

        if kind in ("home", "rv", "wood"):
            base = (102, 90, 72) if kind == "home" else (92, 82, 68)     
            seam = _tint(base, (-20, -20, -22))
            seam2 = _tint(base, (-10, -10, -12))
            knot = _tint(base, (-28, -22, -18))
            hi = _tint(base, (18, 16, 14))
            board_h = 6
            surf.fill(base)

            for y0 in range(0, h, board_h):
                row = int(y0 // board_h)
                row_col = _tint(base, (2, 1, 0) if (row % 2 == 0) else (-2, -2, -3))
                surf.fill(row_col, pygame.Rect(0, y0, w, min(board_h, h - y0)))
                surf.fill(seam, pygame.Rect(0, y0, w, 1))

                # Staggered vertical seams so it reads like boards.
                off = (row * 17) % 29
                board_len = 46
                x = -off
                while x < w:
                    sx = int(x + board_len)
                    if 0 <= sx < w:
                        surf.fill(seam2, pygame.Rect(sx, y0 + 1, 1, min(board_h - 1, h - y0 - 1)))
                    x += board_len

                # Subtle grain / knots (deterministic).
                for gx in range((row * 13) % 31, w, 37):
                    gy = y0 + 2 + ((gx + row * 7) % 2)
                    if 0 <= gy < h:
                        surf.fill(knot, pygame.Rect(gx, gy, 2, 1))
                        if gx + 10 < w and row % 3 == 0:
                            surf.fill(hi, pygame.Rect(gx + 10, gy, 1, 1))

        elif kind in ("school", "linoleum"):
            # School: dirty linoleum tiles (checker + grout + specks).
            base1 = (74, 74, 82)
            base2 = (66, 66, 74)
            grout = (52, 52, 60)
            speck1 = (84, 84, 92)
            speck2 = (58, 58, 66)
            cell = 10
            for y0 in range(0, h, cell):
                for x0 in range(0, w, cell):
                    col = base1 if (((x0 // cell) + (y0 // cell)) % 2 == 0) else base2
                    surf.fill(col, pygame.Rect(x0, y0, min(cell, w - x0), min(cell, h - y0)))
            for x in range(0, w, cell):
                surf.fill(grout, pygame.Rect(x, 0, 1, h))
            for y in range(0, h, cell):
                surf.fill(grout, pygame.Rect(0, y, w, 1))
            for y in range(2, h, 11):
                for x in range((y * 7) % 13, w, 29):
                    surf.fill(speck1 if ((x + y) % 2 == 0) else speck2, pygame.Rect(x, y, 1, 1))
        else:
            # Lobby / hall: stone tile with subtle grid + speckles.      
            base = (70, 72, 78)
            grid = (56, 58, 64)
            speck1 = (82, 84, 90)
            speck2 = (60, 62, 68)
            surf.fill(base)
            cell = 20
            for x in range(0, w, cell):
                surf.fill(grid, pygame.Rect(x, 0, 1, h))
            for y in range(0, h, cell):
                surf.fill(grid, pygame.Rect(0, y, w, 1))
            for y in range(2, h, 9):
                for x in range((y * 7) % 11, w, 23):
                    surf.fill(speck1 if ((x + y) % 2 == 0) else speck2, pygame.Rect(x, y, 1, 1))

        cache[key] = surf
        return surf

    def _marble_surface(
        self,
        w: int,
        h: int,
        *,
        seed: int,
        base: tuple[int, int, int] = (220, 222, 228),
    ) -> pygame.Surface:
        w = int(max(1, w))
        h = int(max(1, h))
        seed = int(seed) & 0xFFFFFFFF
        base = (int(base[0]), int(base[1]), int(base[2]))
        key = (w, h, seed, base)

        cache = getattr(self, "_marble_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._marble_cache = cache
        cached = cache.get(key)
        if isinstance(cached, pygame.Surface):
            return cached

        surf = pygame.Surface((w, h))
        surf.fill(base)

        rnd = random.Random(int(seed) ^ (int(w) << 16) ^ int(h) ^ 0xBADC0FFE)

        # Specks (reads like stone).
        specks = int(clamp((w * h) // 45, 10, 160))
        for _ in range(specks):
            x = int(rnd.randrange(w))
            y = int(rnd.randrange(h))
            d = int(rnd.randrange(-16, 10))
            col = (
                int(clamp(int(base[0]) + d, 0, 255)),
                int(clamp(int(base[1]) + d, 0, 255)),
                int(clamp(int(base[2]) + d, 0, 255)),
            )
            surf.set_at((x, y), col)

        vein_dark = (
            int(clamp(int(base[0]) - 28, 0, 255)),
            int(clamp(int(base[1]) - 28, 0, 255)),
            int(clamp(int(base[2]) - 24, 0, 255)),
        )
        vein_light = (
            int(clamp(int(base[0]) - 12, 0, 255)),
            int(clamp(int(base[1]) - 12, 0, 255)),
            int(clamp(int(base[2]) - 10, 0, 255)),
        )

        # A few drifting veins (deterministic, cached so no flicker).
        vein_count = int(clamp((w + h) // 28, 2, 5))
        for i in range(vein_count):
            x0 = int(rnd.randrange(-w // 2, w))
            y0 = int(rnd.randrange(0, h))
            x1 = int(rnd.randrange(w, int(w * 2.0)))
            y1 = int(clamp(int(y0 + rnd.randrange(-h // 2, h // 2)), 0, h - 1))

            steps = int(max(3, w // 10))
            px, py = x0, y0
            for s in range(steps):
                t = float(s + 1) / float(steps)
                nx = int(round(float(x0) + (float(x1) - float(x0)) * t + rnd.randrange(-2, 3)))
                ny = int(round(float(y0) + (float(y1) - float(y0)) * t + rnd.randrange(-2, 3)))
                pygame.draw.line(surf, vein_light, (px, py), (nx, ny), 1)
                if i % 2 == 0:
                    pygame.draw.line(surf, vein_dark, (px + 1, py), (nx + 1, ny), 1)
                px, py = nx, ny

        cache[key] = surf
        return surf

    def _window_view_surface(self, w: int, h: int) -> pygame.Surface:
        w = int(w)
        h = int(h)
        daylight, tday = self._daylight_amount()
        season = self._season_index()
        wkind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))

        tbin = int(clamp(int(tday * 64.0), 0, 63))
        dbin = int(clamp(int(daylight * 16.0), 0, 16))
        ibin = int(clamp(int(inten * 10.0), 0, 10))
        key = (w, h, int(season), str(wkind), int(tbin), int(dbin), int(ibin))

        cache = getattr(self, "_window_view_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._window_view_cache = cache
        cached = cache.get(key)
        if isinstance(cached, pygame.Surface):
            return cached

        def _lerp(a: float, b: float, t: float) -> float:
            return float(a + (b - a) * t)

        def _lerp_c(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
            return (
                int(round(_lerp(float(a[0]), float(b[0]), t))),
                int(round(_lerp(float(a[1]), float(b[1]), t))),
                int(round(_lerp(float(a[2]), float(b[2]), t))),
            )

        def _mul_c(c: tuple[int, int, int], m: float) -> tuple[int, int, int]:
            return (
                int(clamp(int(round(float(c[0]) * m)), 0, 255)),
                int(clamp(int(round(float(c[1]) * m)), 0, 255)),
                int(clamp(int(round(float(c[2]) * m)), 0, 255)),
            )

        day_sky_top = (78, 146, 224)
        day_sky_bot = (166, 214, 252)
        night_sky_top = (8, 10, 18)
        night_sky_bot = (24, 28, 44)
        sky_top = _lerp_c(night_sky_top, day_sky_top, float(daylight))
        sky_bot = _lerp_c(night_sky_bot, day_sky_bot, float(daylight))

        gloom = 1.0
        if wkind == "cloudy":
            gloom = 1.0 - 0.22 * inten
        elif wkind == "rain":
            gloom = 1.0 - 0.30 * inten
        elif wkind == "storm":
            gloom = 1.0 - 0.44 * inten
        elif wkind == "snow":
            gloom = 1.0 - 0.18 * inten
        sky_top = _mul_c(sky_top, gloom)
        sky_bot = _mul_c(sky_bot, gloom)
        if wkind == "snow" and inten > 0.05:
            sky_top = _lerp_c(sky_top, (200, 210, 230), 0.18 * inten)
            sky_bot = _lerp_c(sky_bot, (230, 240, 250), 0.24 * inten)

        if season == 0:  # spring
            ground = (74, 150, 78)
        elif season == 1:  # summer
            ground = (58, 135, 66)
        elif season == 2:  # autumn
            ground = (122, 118, 70)
        else:  # winter
            ground = (160, 162, 176)
        ground = _mul_c(ground, 0.85 + 0.15 * daylight)
        if wkind == "snow" and inten > 0.15:
            ground = _lerp_c(ground, (210, 214, 224), 0.45 * inten)

        base = pygame.Surface((w, h))
        for yy in range(h):
            t = float(yy) / max(1.0, float(h - 1))
            base.fill(_lerp_c(sky_top, sky_bot, t), pygame.Rect(0, yy, w, 1))
        gh = max(3, int(h // 3))
        base.fill(ground, pygame.Rect(0, h - gh, w, gh))
        base.fill((16, 16, 22), pygame.Rect(0, h - gh, w, 1))

        # Dawn / dusk warm stripe near horizon.
        if 0.0 < daylight < 1.0:
            warm = (220, 150, 96) if tday < 0.5 else (226, 140, 86)
            warm_y = int(h - gh - 2)
            if 0 <= warm_y < h:
                base.fill(_mul_c(warm, 0.65), pygame.Rect(0, warm_y, w, 1))

        # Tiny silhouettes (reads as distant trees/buildings).
        if w >= 10 and h >= 8:
            base.fill((18, 18, 22), pygame.Rect(2, int(h - gh - 3), 2, 3))
            base.fill((18, 18, 22), pygame.Rect(int(w - 5), int(h - gh - 4), 3, 4))

        view = pygame.Surface((w, h), pygame.SRCALPHA)
        view.blit(base, (0, 0))

        # Glass tint + simple reflection lines.
        tint = pygame.Surface((w, h), pygame.SRCALPHA)
        tint.fill((60, 86, 120, 55))
        pygame.draw.line(tint, (240, 244, 255, 70), (1, 1), (w - 2, 1), 1)
        pygame.draw.line(tint, (210, 220, 240, 45), (1, 2), (w - 3, 2), 1)
        view.blit(tint, (0, 0))

        cache[key] = view
        return view

    def _draw_window_precip(self, surface: pygame.Surface, rect: pygame.Rect, *, kind: str, intensity: float) -> None:
        kind = str(kind)
        intensity = float(clamp(float(intensity), 0.0, 1.0))
        if intensity <= 0.05:
            return

        w = int(rect.w)
        h = int(rect.h)
        if w <= 0 or h <= 0:
            return

        t = float(getattr(self, "world_time_s", 0.0))
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)

        if kind in ("rain", "storm"):
            alpha = int(clamp(int(70 + 140 * intensity), 0, 220))
            col = (220, 232, 255, alpha)
            step = max(4, int(round(8 - 3 * intensity)))
            speed = 140.0 if kind == "rain" else 220.0
            off = int((t * speed) % step)
            dx = max(5, int(round(h * 0.55)))
            for x in range(-h, w + h, step):
                xx = int(x + off)
                pygame.draw.line(overlay, col, (xx, 0), (xx + dx, h), 1)
            # A few bigger drops.
            if kind == "storm" and intensity > 0.35:
                for i in range(0, w, 9):
                    yy = int((i * 7 + int(t * 60)) % max(1, h))
                    overlay.fill((240, 246, 255, alpha), pygame.Rect(i, yy, 1, 2))

        elif kind == "snow":
            alpha = int(clamp(int(60 + 120 * intensity), 0, 210))
            col = (240, 244, 250, alpha)
            for y in range(1, h, 5):
                for x in range((y * 7 + int(t * 30)) % 9, w, 11):
                    overlay.fill(col, pygame.Rect(x, y, 1, 1))

        else:
            return

        surface.blit(overlay, rect.topleft)

    def _draw_rv_interior_scene(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        tile = int(self._RV_INT_TILE_SIZE)
        map_w = int(self._RV_INT_W) * tile
        map_h = int(self._RV_INT_H) * tile
        map_x = (INTERNAL_W - map_w) // 2
        map_y = 92
        panel = pygame.Rect(map_x - 10, map_y - 10, map_w + 20, map_h + 20)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (70, 70, 86), panel, 2, border_radius=12)

        # Cozier interior rendering (less grid, more furniture detail) inspired by the
        # user's reference screenshot.
        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        surface.blit(self._interior_floor_surface(map_w, map_h, kind="rv"), map_rect.topleft)

        wall = (26, 26, 32)
        wall_hi = (56, 56, 70)
        wall_edge = (12, 12, 16)
        frame = (88, 80, 70)
        frame2 = (62, 56, 48)
        outline = (10, 10, 12)

        bed_matt = (186, 170, 156)
        bed_blank = (132, 154, 112)
        bed_blank2 = (108, 130, 92)
        pillow = (238, 238, 244)
        wood = (98, 78, 56)
        wood2 = (74, 58, 42)
        steel = (216, 220, 228)
        steel2 = (134, 140, 152)
        seat = (72, 86, 148)
        seat2 = (54, 64, 118)

        wkind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        glass_w = max(4, int(tile - 6))
        glass_h = max(4, int(tile - 8))
        win_view = self._window_view_surface(glass_w, glass_h)
        layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)

        # Rug under the table.
        t_tiles: list[tuple[int, int]] = []
        for yy, row in enumerate(layout):
            for xx, ch in enumerate(row[: int(self._RV_INT_W)]):
                if ch == "T":
                    t_tiles.append((int(xx), int(yy)))
        if t_tiles:
            min_tx = min(p[0] for p in t_tiles)
            max_tx = max(p[0] for p in t_tiles)
            min_ty = min(p[1] for p in t_tiles)
            max_ty = max(p[1] for p in t_tiles)
            rug = pygame.Rect(
                map_x + min_tx * tile - 2,
                map_y + min_ty * tile + 8,
                (max_tx - min_tx + 1) * tile + 4,
                (max_ty - min_ty + 1) * tile + 6,
            )
            rug.clamp_ip(map_rect.inflate(-4, -4))
            rug_s = pygame.Surface((rug.w, rug.h), pygame.SRCALPHA)
            rug_s.fill((160, 82, 92, 190))
            for y in range(2, rug.h, 5):
                pygame.draw.line(rug_s, (210, 150, 150, 60), (2, y), (rug.w - 3, y), 1)
            pygame.draw.rect(rug_s, (40, 18, 20, 220), rug_s.get_rect(), 2, border_radius=4)
            surface.blit(rug_s, rug.topleft)

        bed_done: set[tuple[int, int]] = set()
        for y, row in enumerate(layout):
            for x in range(int(self._RV_INT_W)):
                ch = row[x] if x < len(row) else "W"
                r = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)

                if ch in ("W", "V"):
                    pygame.draw.rect(surface, wall, r)
                    pygame.draw.rect(surface, wall_hi, pygame.Rect(r.x, r.y, r.w, 4))
                    pygame.draw.rect(surface, wall_edge, r, 1)
                    pygame.draw.rect(surface, frame2, pygame.Rect(r.x, r.bottom - 4, r.w, 2))
                    if ch == "V":
                        glass = pygame.Rect(r.x + 3, r.y + 4, r.w - 6, r.h - 8)
                        wv = win_view
                        if wv.get_width() != glass.w or wv.get_height() != glass.h:
                            wv = pygame.transform.scale(wv, (int(glass.w), int(glass.h)))
                        surface.blit(wv, glass.topleft)
                        pygame.draw.rect(surface, frame, glass.inflate(4, 4), 1, border_radius=2)
                        pygame.draw.rect(surface, outline, glass, 1)
                        pygame.draw.line(surface, (40, 40, 50), (glass.centerx, glass.top + 1), (glass.centerx, glass.bottom - 2), 1)
                    continue

                if ch == "D":
                    dr = pygame.Rect(r.x + 4, r.y + 3, r.w - 8, r.h - 6)
                    pygame.draw.rect(surface, wood, dr, border_radius=2)
                    pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                    pygame.draw.rect(surface, wood2, pygame.Rect(dr.x + 2, dr.y + 2, dr.w - 4, 4))
                    pygame.draw.circle(surface, outline, (dr.right - 4, dr.centery), 2)
                    continue

                if ch == "B":
                    if (int(x), int(y)) in bed_done:
                        continue

                    bw = 1
                    while self._rv_int_char_at(int(x) + int(bw), int(y)) == "B":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._rv_int_char_at(int(x), int(ny)) != "B":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._rv_int_char_at(int(x) + int(dx), int(ny)) != "B":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1

                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            bed_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    bed_r = br.inflate(-4, -8)
                    sh = pygame.Surface((bed_r.w, bed_r.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 90), sh.get_rect(), border_radius=6)
                    surface.blit(sh, (bed_r.x + 2, bed_r.y + 3))

                    pygame.draw.rect(surface, bed_matt, bed_r, border_radius=6)
                    pygame.draw.rect(surface, outline, bed_r, 1, border_radius=6)

                    hb = pygame.Rect(bed_r.x, bed_r.y, 8, bed_r.h)
                    pygame.draw.rect(surface, wood2, hb, border_radius=5)
                    pygame.draw.rect(surface, outline, hb, 1, border_radius=5)

                    p1 = pygame.Rect(bed_r.x + 10, bed_r.y + 4, max(10, bed_r.w // 5), 7)
                    p2 = pygame.Rect(p1.x, p1.y + 10, p1.w, 7)
                    pygame.draw.rect(surface, pillow, p1, border_radius=3)
                    pygame.draw.rect(surface, pillow, p2, border_radius=3)
                    pygame.draw.rect(surface, outline, p1, 1, border_radius=3)
                    pygame.draw.rect(surface, outline, p2, 1, border_radius=3)

                    bl = pygame.Rect(bed_r.x + 10, bed_r.y + 20, bed_r.w - 14, max(10, bed_r.h - 24))
                    pygame.draw.rect(surface, bed_blank, bl, border_radius=5)
                    pygame.draw.rect(surface, outline, bl, 1, border_radius=5)
                    for sx in range(bl.x + 6, bl.right - 4, 10):
                        pygame.draw.line(surface, bed_blank2, (sx, bl.y + 2), (sx, bl.bottom - 3), 1)
                    # Bed legs/frame (so it doesn't look like it floats).
                    floor_y = int(br.bottom - 3)
                    leg_y0 = int(bed_r.bottom - 2)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(wood2, add=(-12, -12, -12))
                        for lx in (int(bed_r.x + 6), int(bed_r.right - 8)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                            surface.fill(leg_col, pygame.Rect(int(lx - 1), int(floor_y), 4, 1))
                    continue

                if ch == "S":
                    cab = pygame.Rect(r.x + 2, r.y + 6, r.w - 4, r.h - 10)
                    sh = pygame.Surface((cab.w, cab.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 80), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (cab.x + 2, cab.y + 3))

                    pygame.draw.rect(surface, wood, cab, border_radius=3)
                    pygame.draw.rect(surface, outline, cab, 1, border_radius=3)
                    pygame.draw.rect(surface, wood2, pygame.Rect(cab.x + 1, cab.y + 1, cab.w - 2, 3))
                    pygame.draw.rect(surface, wood2, pygame.Rect(cab.x + 1, cab.bottom - 4, cab.w - 2, 3))

                    is_open = bool(getattr(self, "rv_ui_open", False)) and str(getattr(self, "rv_ui_focus", "map")) == "storage"
                    if is_open:
                        inner = cab.inflate(-4, -6)
                        pygame.draw.rect(surface, (20, 20, 24), inner, border_radius=2)
                        pygame.draw.rect(surface, outline, inner, 1, border_radius=2)
                        for sy in range(inner.y + 3, inner.bottom - 1, 5):
                            pygame.draw.line(surface, (44, 44, 52), (inner.x + 2, sy), (inner.right - 3, sy), 1)

                        door_w = max(4, int(cab.w // 4))
                        ld = pygame.Rect(cab.x + 1, cab.y + 3, door_w, cab.h - 6)
                        rd = pygame.Rect(cab.right - door_w - 1, cab.y + 3, door_w, cab.h - 6)
                        pygame.draw.rect(surface, wood, ld, border_radius=2)
                        pygame.draw.rect(surface, outline, ld, 1, border_radius=2)
                        pygame.draw.rect(surface, wood, rd, border_radius=2)
                        pygame.draw.rect(surface, outline, rd, 1, border_radius=2)
                        # Tiny visible supplies.
                        can = pygame.Rect(inner.x + 3, inner.y + 4, 3, 5)
                        surface.fill((180, 60, 70), can)
                        pygame.draw.rect(surface, outline, can, 1)
                        jar = pygame.Rect(inner.right - 7, inner.y + 5, 4, 4)
                        surface.fill((90, 140, 190), jar)
                        pygame.draw.rect(surface, outline, jar, 1)
                    else:
                        midx = int(cab.centerx)
                        pygame.draw.line(surface, wood2, (midx, cab.y + 4), (midx, cab.bottom - 5), 2)
                        pygame.draw.line(surface, outline, (midx, cab.y + 4), (midx, cab.bottom - 5), 1)
                        for hx in (midx - 4, midx + 3):
                            hrect = pygame.Rect(int(hx), cab.y + 8, 2, 5)
                            surface.fill((220, 220, 230), hrect)
                            pygame.draw.rect(surface, outline, hrect, 1)
                    continue

                if ch == "T":
                    top = pygame.Rect(r.x + 2, r.y + 7, r.w - 4, r.h - 12)
                    pygame.draw.rect(surface, wood, top, border_radius=2)
                    pygame.draw.rect(surface, outline, top, 1, border_radius=2)
                    pygame.draw.rect(surface, wood2, pygame.Rect(top.x + 1, top.y + 1, top.w - 2, 3))
                    # Table legs/frame (so it reads like real furniture).
                    floor_y = int(r.bottom - 3)
                    leg_y0 = int(top.bottom - 1)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0 and top.w >= 10:
                        leg_col = self._tint(wood2, add=(-10, -10, -10))
                        for lx in (int(top.x + 2), int(top.right - 4)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                            surface.fill(leg_col, pygame.Rect(int(lx - 1), int(floor_y), 4, 1))
                    if (x + y) % 3 == 0:
                        mug = pygame.Rect(top.centerx - 2, top.y + 4, 4, 4)
                        pygame.draw.rect(surface, (230, 230, 240), mug, border_radius=1)
                        pygame.draw.rect(surface, outline, mug, 1, border_radius=1)
                    elif (x + y) % 3 == 1:
                        screen = pygame.Rect(top.centerx - 5, top.y + 2, 10, 6)
                        pygame.draw.rect(surface, (36, 38, 44), screen, border_radius=2)
                        pygame.draw.rect(surface, steel2, screen, 1, border_radius=2)
                    continue

                if ch == "K":
                    if (x, y) in k_done:
                        continue

                    bw = 1
                    while self._rv_int_char_at(int(x) + int(bw), int(y)) == "K":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._rv_int_char_at(int(x), int(ny)) != "K":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._rv_int_char_at(int(x) + int(dx), int(ny)) != "K":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            k_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    obj = br.inflate(-6, -10)
                    sh = pygame.Surface((obj.w, obj.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 70), sh.get_rect(), border_radius=5)
                    surface.blit(sh, (obj.x + 2, obj.y + 3))

                    # Long counter: top (lighter) + front face (darker) to read as a continuous kitchen.
                    top_h = int(clamp(int(obj.h * 0.28), 7, 10))
                    top_r = pygame.Rect(obj.x, obj.y, obj.w, top_h)
                    front_r = pygame.Rect(obj.x, obj.y + top_h - 1, obj.w, obj.h - (top_h - 1))
                    pygame.draw.rect(surface, wood2, front_r, border_radius=6)
                    pygame.draw.rect(surface, outline, front_r, 1, border_radius=6)
                    pygame.draw.rect(surface, wood, top_r, border_radius=6)
                    pygame.draw.rect(surface, outline, top_r, 1, border_radius=6)
                    pygame.draw.line(surface, frame2, (top_r.x + 2, top_r.top + 1), (top_r.right - 3, top_r.top + 1), 1)
                    pygame.draw.line(surface, frame2, (top_r.x + 2, top_r.bottom - 2), (top_r.right - 3, top_r.bottom - 2), 1)

                    # Cabinet doors + handles on the front face (depth cues).
                    modules = int(max(2, int(bw)))
                    for i in range(1, modules):
                        xline = int(front_r.x + (front_r.w * i) / modules)
                        pygame.draw.line(surface, frame2, (xline, front_r.y + 4), (xline, front_r.bottom - 5), 2)
                        pygame.draw.line(surface, outline, (xline, front_r.y + 4), (xline, front_r.bottom - 5), 1)
                    for i in range(modules):
                        cx = int(front_r.x + (front_r.w * (i + 0.5)) / modules)
                        hrect = pygame.Rect(cx - 1, front_r.y + 8, 2, 7)
                        surface.fill((220, 220, 230), hrect)
                        pygame.draw.rect(surface, outline, hrect, 1)

                    # Sink + stove placed once across the whole block.
                    sink_w = int(clamp(int(top_r.w * 0.18), 12, 18))
                    sink_h = int(clamp(int(top_r.h - 3), 5, 7))
                    sink = pygame.Rect(top_r.x + int(top_r.w * 0.16), top_r.y + 2, sink_w, sink_h)
                    pygame.draw.rect(surface, steel, sink, border_radius=3)
                    pygame.draw.rect(surface, outline, sink, 1, border_radius=3)
                    pygame.draw.circle(surface, (60, 60, 70), (sink.right - 3, sink.y + 3), 1)
                    pygame.draw.line(surface, steel2, (sink.x + 2, sink.y + 1), (sink.x + 2, sink.y - 3), 1)
                    pygame.draw.line(surface, steel2, (sink.x + 2, sink.y - 3), (sink.x + 6, sink.y - 3), 1)

                    stove_w = int(clamp(int(top_r.w * 0.20), 12, 18))
                    stove_h = sink_h
                    stove = pygame.Rect(top_r.right - int(top_r.w * 0.18) - stove_w, top_r.y + 2, stove_w, stove_h)
                    pygame.draw.rect(surface, (40, 40, 50), stove, border_radius=3)
                    pygame.draw.rect(surface, outline, stove, 1, border_radius=3)
                    pygame.draw.circle(surface, (80, 80, 90), (stove.x + stove_w // 3, stove.centery), 2)
                    pygame.draw.circle(surface, (80, 80, 90), (stove.right - stove_w // 3, stove.centery), 2)

                    # Pots / bowls / plates on the countertop.
                    plate = pygame.Rect(top_r.right - 14, top_r.y + 2, 12, 5)
                    pygame.draw.ellipse(surface, (232, 232, 238), plate)
                    pygame.draw.ellipse(surface, outline, plate, 1)
                    bowl = pygame.Rect(top_r.x + 8, top_r.y + 2, 10, 5)
                    pygame.draw.ellipse(surface, (200, 210, 220), bowl)
                    pygame.draw.ellipse(surface, outline, bowl, 1)
                    pot = pygame.Rect(top_r.centerx - 8, top_r.y + 1, 16, 6)
                    pygame.draw.rect(surface, (40, 40, 50), pot, border_radius=3)
                    pygame.draw.rect(surface, outline, pot, 1, border_radius=3)
                    pygame.draw.line(surface, steel2, (pot.x + 2, pot.y + 2), (pot.right - 3, pot.y + 2), 1)
                    pygame.draw.line(surface, outline, (pot.x + 3, pot.y + 1), (pot.right - 4, pot.y + 1), 1)

                    # Small rack cue above the counter.
                    rack_y = int(top_r.y - 6)
                    if rack_y >= int(br.y) + 1:
                        pygame.draw.line(surface, outline, (top_r.x + 2, rack_y), (top_r.right - 3, rack_y), 1)
                        for ux in (top_r.x + 8, top_r.centerx, top_r.right - 10):
                            pygame.draw.line(surface, steel2, (int(ux), rack_y + 1), (int(ux), rack_y + 4), 1)
                    continue

                if ch == "H":
                    bench = pygame.Rect(r.x + 2, r.y + 7, r.w - 4, r.h - 11)
                    pygame.draw.rect(surface, wood2, bench, border_radius=2)
                    pygame.draw.rect(surface, outline, bench, 1, border_radius=2)
                    pygame.draw.rect(surface, wood, pygame.Rect(bench.x + 1, bench.y + 1, bench.w - 2, 3))
                    pygame.draw.line(surface, steel2, (bench.x + 2, bench.y), (bench.right - 3, bench.y + 5), 1)
                    pygame.draw.rect(surface, steel2, pygame.Rect(bench.x + 4, bench.y + 6, 5, 2), border_radius=1)
                    continue

                if ch == "C":
                    back = pygame.Rect(r.x + 4, r.y + 5, r.w - 8, 7)
                    base = pygame.Rect(r.x + 4, r.y + 11, r.w - 8, 6)
                    pygame.draw.rect(surface, seat2, back, border_radius=3)
                    pygame.draw.rect(surface, outline, back, 1, border_radius=3)
                    pygame.draw.rect(surface, seat, base, border_radius=3)
                    pygame.draw.rect(surface, outline, base, 1, border_radius=3)
                    pygame.draw.rect(surface, self._tint(seat, add=(16, 16, 16)), pygame.Rect(base.x + 1, base.y + 1, base.w - 2, 2), border_radius=2)

                    arm_l = pygame.Rect(base.x - 1, base.y - 1, 3, 3)
                    arm_r = pygame.Rect(base.right - 2, base.y - 1, 3, 3)
                    pygame.draw.rect(surface, seat2, arm_l, border_radius=2)
                    pygame.draw.rect(surface, seat2, arm_r, border_radius=2)
                    pygame.draw.rect(surface, outline, arm_l, 1, border_radius=2)
                    pygame.draw.rect(surface, outline, arm_r, 1, border_radius=2)

                    head = pygame.Rect(back.centerx - 3, back.y - 3, 6, 3)
                    pygame.draw.rect(surface, seat2, head, border_radius=2)
                    pygame.draw.rect(surface, outline, head, 1, border_radius=2)

                    floor_y = int(r.bottom - 3)
                    leg_y0 = int(base.bottom - 1)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(seat2, add=(-24, -24, -24))
                        for lx in (int(base.x + 2), int(base.right - 4)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                    continue

        if getattr(self, "rv_edit_mode", False):
            blocks = getattr(self, "rv_edit_blocks", None)
            if isinstance(blocks, list) and blocks:
                sel_i = int(getattr(self, "rv_edit_index", 0)) % len(blocks)
                _ch, cells = blocks[sel_i]
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                sel = pygame.Rect(
                    int(map_x + min_x * tile),
                    int(map_y + min_y * tile),
                    int((max_x - min_x + 1) * tile),
                    int((max_y - min_y + 1) * tile),
                )
                ov = pygame.Surface((sel.w, sel.h), pygame.SRCALPHA)
                ov.fill((255, 220, 140, 50))
                pygame.draw.rect(ov, (255, 220, 140, 120), ov.get_rect(), 2, border_radius=8)
                surface.blit(ov, sel.topleft)

        face = pygame.Vector2(self.rv_int_facing)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)
        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"
        pose = str(getattr(self, "player_pose", "")).strip()
        pose_space = str(getattr(self, "player_pose_space", "")).strip()
        pose_anchor = getattr(self, "player_pose_anchor", None)
        use_pose = bool(pose and pose_space == "rv" and pose_anchor is not None)
        if use_pose:
            p = pygame.Vector2(float(pose_anchor[0]), float(pose_anchor[1]))
            if pose == "sleep":
                frame = int(float(getattr(self, "player_pose_phase", 0.0)) * 2.0) % 2
                base = self._get_pose_sprite("sleep", frame=frame)
            else:
                base = self._get_pose_sprite("sit", direction=d, frame=0)
        else:
            p = pygame.Vector2(self.rv_int_pos)
            speed2 = float(self.rv_int_vel.length_squared())
            pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
            pf_run = getattr(self, "player_frames_run", None)
            is_run = bool(getattr(self, "player_sprinting", False))
            pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
            frames = pf.get(d, pf["down"])
            if speed2 <= 0.2 or len(frames) <= 1:
                base = frames[0]
            else:
                walk = frames[1:]
                phase = (float(self.rv_int_walk_phase) % math.tau) / math.tau
                idx = int(phase * len(walk)) % len(walk)
                base = walk[idx]

        scale = int(max(1, int(self._RV_INT_SPRITE_SCALE)))
        spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
        sx = int(round(map_x + p.x))
        sy = int(round(map_y + p.y))
        if use_pose and pose == "sleep":
            shadow = pygame.Rect(0, 0, 20, 8)
            shadow.center = (sx, sy + 4)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 90), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect(center=(sx, sy + 2))
            surface.blit(spr, rect)
        elif use_pose and pose == "sit":
            shadow = pygame.Rect(0, 0, 16, 7)
            shadow.center = (sx, sy + 6)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 110), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect(center=(sx, sy + 4))
            surface.blit(spr, rect)
        else:
            shadow = pygame.Rect(0, 0, 14, 6)
            shadow.center = (sx, sy + 8)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 120), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect()
            rect.midbottom = (sx, sy + 12)
            surface.blit(spr, rect)

        # Hover: furniture details + outline.
        mouse = None
        if (
            not self.inv_open
            and not bool(getattr(self, "rv_ui_open", False))
            and not bool(getattr(self, "world_map_open", False))
        ):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())
        if mouse is not None:
            mx, my = int(mouse[0]), int(mouse[1])
            if map_rect.collidepoint(mx, my):
                ix = int((mx - int(map_x)) // int(tile))
                iy = int((my - int(map_y)) // int(tile))
                ch = str(self._rv_int_char_at(ix, iy))[:1]
                movable = {"B", "S", "K", "H", "T", "C"}
                if ch in movable:
                    w = int(self._RV_INT_W)
                    h = int(self._RV_INT_H)
                    seen: set[tuple[int, int]] = set()
                    stack = [(int(ix), int(iy))]
                    cells: list[tuple[int, int]] = []
                    while stack:
                        cx, cy = stack.pop()
                        cx = int(cx)
                        cy = int(cy)
                        if (cx, cy) in seen:
                            continue
                        if not (0 <= cx < w and 0 <= cy < h):
                            continue
                        if str(self._rv_int_char_at(cx, cy))[:1] != ch:
                            continue
                        seen.add((cx, cy))
                        cells.append((cx, cy))
                        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                    if cells:
                        min_x = min(int(p[0]) for p in cells)
                        max_x = max(int(p[0]) for p in cells)
                        min_y = min(int(p[1]) for p in cells)
                        max_y = max(int(p[1]) for p in cells)
                        sel = pygame.Rect(
                            int(map_x + min_x * tile),
                            int(map_y + min_y * tile),
                            int((max_x - min_x + 1) * tile),
                            int((max_y - min_y + 1) * tile),
                        )
                        ov = pygame.Surface((sel.w, sel.h), pygame.SRCALPHA)
                        ov.fill((255, 220, 140, 20))
                        pygame.draw.rect(ov, (255, 220, 140, 150), ov.get_rect(), 2, border_radius=8)
                        surface.blit(ov, sel.topleft)

                        meta: dict[str, tuple[str, str]] = {
                            "B": ("床", "躺下可恢复体力与心情"),
                            "S": ("储物柜", "打开后可存取物资"),
                            "T": ("桌子", "临时放置/整理物品（规划）"),
                            "K": ("厨房", "台面上有锅碗瓢盆（烹饪/净水规划）"),
                            "H": ("工作台", "枪械加工/修理（规划）"),
                            "C": ("座椅", "乘坐休息（规划）"),
                        }
                        name, desc = meta.get(ch, ("家具", ""))
                        if bool(getattr(self, "rv_edit_mode", False)):
                            hint = "左键拖拽 | 方向键微调 | R 退出摆放"
                        else:
                            hint = "R 进入摆放模式 | 左键拖拽"
                        self._hover_tooltip = ([str(name), str(desc), str(hint)], (mx, my))
        return

        floor_a = (64, 62, 58)
        floor_b = (58, 56, 52)
        wall = (22, 22, 26)
        wall_hi = (38, 38, 46)
        bed = (170, 150, 140)
        bed2 = (128, 108, 100)
        shelf = (92, 78, 62)
        shelf2 = (66, 56, 46)
        table = (88, 72, 54)
        table2 = (60, 50, 38)
        seat = (56, 72, 140)
        seat2 = (40, 52, 112)
        kitchen = (86, 78, 68)
        kitchen2 = (60, 54, 46)
        steel = (220, 220, 230)
        work = (96, 84, 66)
        work2 = (62, 54, 42)
        tool = (150, 150, 160)
        door = (240, 220, 140)

        # Window "outside view" cache (small stylized sky/ground) so the RV
        # interior feels like a real space with windows.
        daylight, tday = self._daylight_amount()
        season = self._season_index()
        wkind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))

        def _lerp(a: float, b: float, t: float) -> float:
            return float(a + (b - a) * t)

        def _lerp_c(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
            return (
                int(round(_lerp(float(a[0]), float(b[0]), t))),
                int(round(_lerp(float(a[1]), float(b[1]), t))),
                int(round(_lerp(float(a[2]), float(b[2]), t))),
            )

        def _mul_c(c: tuple[int, int, int], m: float) -> tuple[int, int, int]:
            return (
                int(clamp(int(round(float(c[0]) * m)), 0, 255)),
                int(clamp(int(round(float(c[1]) * m)), 0, 255)),
                int(clamp(int(round(float(c[2]) * m)), 0, 255)),
            )

        glass_w = max(4, int(tile - 6))
        glass_h = max(4, int(tile - 8))
        tbin = int(clamp(int(tday * 64.0), 0, 63))
        dbin = int(clamp(int(daylight * 16.0), 0, 16))
        ibin = int(clamp(int(inten * 10.0), 0, 10))
        wkey = (glass_w, glass_h, int(season), str(wkind), int(tbin), int(dbin), int(ibin))

        win_view: pygame.Surface | None = getattr(self, "_rv_win_view_cache", None)
        if (
            win_view is None
            or getattr(self, "_rv_win_view_cache_key", None) != wkey
            or win_view.get_width() != glass_w
            or win_view.get_height() != glass_h
        ):
            day_sky_top = (78, 146, 224)
            day_sky_bot = (166, 214, 252)
            night_sky_top = (8, 10, 18)
            night_sky_bot = (24, 28, 44)
            sky_top = _lerp_c(night_sky_top, day_sky_top, float(daylight))
            sky_bot = _lerp_c(night_sky_bot, day_sky_bot, float(daylight))

            gloom = 1.0
            if wkind == "cloudy":
                gloom = 1.0 - 0.22 * inten
            elif wkind == "rain":
                gloom = 1.0 - 0.30 * inten
            elif wkind == "storm":
                gloom = 1.0 - 0.44 * inten
            elif wkind == "snow":
                gloom = 1.0 - 0.18 * inten
            sky_top = _mul_c(sky_top, gloom)
            sky_bot = _mul_c(sky_bot, gloom)
            if wkind == "snow" and inten > 0.05:
                sky_top = _lerp_c(sky_top, (200, 210, 230), 0.18 * inten)
                sky_bot = _lerp_c(sky_bot, (230, 240, 250), 0.24 * inten)

            if season == 0:  # spring
                ground = (74, 150, 78)
            elif season == 1:  # summer
                ground = (58, 135, 66)
            elif season == 2:  # autumn
                ground = (122, 118, 70)
            else:  # winter
                ground = (160, 162, 176)
            ground = _mul_c(ground, 0.85 + 0.15 * daylight)
            if wkind == "snow" and inten > 0.15:
                ground = _lerp_c(ground, (210, 214, 224), 0.45 * inten)

            base = pygame.Surface((glass_w, glass_h))
            for yy in range(int(glass_h)):
                t = float(yy) / max(1.0, float(glass_h - 1))
                base.fill(_lerp_c(sky_top, sky_bot, t), pygame.Rect(0, int(yy), int(glass_w), 1))
            gh = max(3, int(glass_h // 3))
            base.fill(ground, pygame.Rect(0, int(glass_h - gh), int(glass_w), int(gh)))
            base.fill((16, 16, 22), pygame.Rect(0, int(glass_h - gh), int(glass_w), 1))

            # Dawn / dusk warm stripe near horizon.
            if 0.0 < daylight < 1.0:
                warm = (220, 150, 96) if tday < 0.5 else (226, 140, 86)
                warm_y = int(glass_h - gh - 2)
                if 0 <= warm_y < glass_h:
                    base.fill(_mul_c(warm, 0.65), pygame.Rect(0, warm_y, int(glass_w), 1))

            # Tiny silhouettes (reads as distant trees/buildings).
            if glass_w >= 10 and glass_h >= 8:
                base.fill((18, 18, 22), pygame.Rect(2, int(glass_h - gh - 3), 2, 3))
                base.fill((18, 18, 22), pygame.Rect(int(glass_w - 5), int(glass_h - gh - 4), 3, 4))

            win_view = pygame.Surface((glass_w, glass_h), pygame.SRCALPHA)
            win_view.blit(base, (0, 0))
            tint = pygame.Surface((glass_w, glass_h), pygame.SRCALPHA)
            tint.fill((60, 86, 120, 55))
            pygame.draw.line(tint, (240, 244, 255, 70), (1, 1), (int(glass_w - 2), 1), 1)
            pygame.draw.line(tint, (210, 220, 240, 45), (1, 2), (int(glass_w - 3), 2), 1)
            win_view.blit(tint, (0, 0))
            self._rv_win_view_cache = win_view
            self._rv_win_view_cache_key = wkey

        bed_done: set[tuple[int, int]] = set()
        for y, row in enumerate(self._RV_INT_LAYOUT):
            for x in range(int(self._RV_INT_W)):
                ch = row[x] if x < len(row) else "W"
                r = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)
                if ch in ("W", "V"):
                    pygame.draw.rect(surface, wall, r)
                    pygame.draw.rect(surface, wall_hi, pygame.Rect(r.x, r.y, r.w, 3))
                    pygame.draw.rect(surface, (0, 0, 0), r, 1)
                    if ch == "V":
                        glass = pygame.Rect(r.x + 3, r.y + 4, r.w - 6, r.h - 8)
                        if win_view is not None and glass.w > 0 and glass.h > 0:
                            wv = win_view
                            if wv.get_width() != glass.w or wv.get_height() != glass.h:
                                wv = pygame.transform.scale(wv, (int(glass.w), int(glass.h)))
                            surface.blit(wv, glass.topleft)
                        pygame.draw.rect(surface, (10, 10, 12), glass, 1)
                        pygame.draw.line(
                            surface,
                            (40, 40, 50),
                            (glass.centerx, glass.top + 1),
                            (glass.centerx, glass.bottom - 2),
                            1,
                        )
                    continue

                col = floor_a if ((x + y) % 2 == 0) else floor_b
                pygame.draw.rect(surface, col, r)
                pygame.draw.rect(surface, (0, 0, 0), r, 1)

                if ch == "D":
                    pygame.draw.rect(surface, door, pygame.Rect(r.right - 4, r.y + 3, 3, r.h - 6))
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.right - 4, r.y + 3, 3, r.h - 6), 1)
                elif ch == "B":
                    if (int(x), int(y)) in bed_done:
                        continue

                    # Find the full bed block (contiguous 'B's) and draw as one object
                    # so it reads clearly as a single bed.
                    bw = 1
                    while self._rv_int_char_at(int(x) + int(bw), int(y)) == "B":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._rv_int_char_at(int(x), int(ny)) != "B":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._rv_int_char_at(int(x) + int(dx), int(ny)) != "B":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1

                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            bed_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    bed_r = pygame.Rect(br.x + 2, br.y + 3, br.w - 4, br.h - 6)
                    pygame.draw.rect(surface, bed, bed_r, border_radius=4)
                    pygame.draw.rect(surface, (10, 10, 12), bed_r, 1, border_radius=4)

                    # Pillow + blanket band.
                    pillow_w = int(clamp(int(bed_r.w // 3), 8, max(8, bed_r.w - 4)))
                    pillow = pygame.Rect(bed_r.x + 2, bed_r.y + 2, pillow_w, 6)
                    pygame.draw.rect(surface, (230, 230, 235), pillow, border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pillow, 1, border_radius=2)
                    pygame.draw.rect(surface, bed2, pygame.Rect(bed_r.x + 2, bed_r.y + 10, bed_r.w - 4, 6), border_radius=2)

                    for sx in range(bed_r.x + 6, bed_r.right - 4, 6):
                        pygame.draw.line(surface, (10, 10, 12), (sx, bed_r.y + 11), (sx, bed_r.y + 15), 1)
                    continue
                elif ch == "S":
                    pygame.draw.rect(surface, shelf, pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6), 1, border_radius=2)
                    pygame.draw.line(surface, shelf2, (r.x + 4, r.y + r.h // 2), (r.right - 5, r.y + r.h // 2), 2)
                elif ch == "T":
                    pygame.draw.rect(surface, table, pygame.Rect(r.x + 3, r.y + 6, r.w - 6, r.h - 10), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 3, r.y + 6, r.w - 6, r.h - 10), 1, border_radius=2)
                    pygame.draw.rect(surface, table2, pygame.Rect(r.x + 4, r.y + 8, r.w - 8, 3))
                elif ch == "K":
                    pygame.draw.rect(surface, kitchen, pygame.Rect(r.x + 2, r.y + 5, r.w - 4, r.h - 8), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 2, r.y + 5, r.w - 4, r.h - 8), 1, border_radius=2)
                    pygame.draw.rect(surface, kitchen2, pygame.Rect(r.x + 3, r.y + 7, r.w - 6, 3))
                    # Sink / stove hints.
                    if x % 2 == 0:
                        pygame.draw.rect(surface, steel, pygame.Rect(r.x + 6, r.y + 6, 6, 4), border_radius=1)
                        pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 6, r.y + 6, 6, 4), 1, border_radius=1)
                        pygame.draw.circle(surface, (40, 40, 50), (r.x + 9, r.y + 8), 1)
                    else:
                        pygame.draw.rect(surface, steel, pygame.Rect(r.x + 5, r.y + 6, 8, 4), border_radius=1)
                        pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 5, r.y + 6, 8, 4), 1, border_radius=1)
                        pygame.draw.circle(surface, (40, 40, 50), (r.x + 7, r.y + 8), 1)
                        pygame.draw.circle(surface, (40, 40, 50), (r.x + 11, r.y + 8), 1)
                elif ch == "H":
                    pygame.draw.rect(surface, work, pygame.Rect(r.x + 2, r.y + 6, r.w - 4, r.h - 9), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 2, r.y + 6, r.w - 4, r.h - 9), 1, border_radius=2)
                    pygame.draw.rect(surface, work2, pygame.Rect(r.x + 3, r.y + 8, r.w - 6, 3))
                    # Tools.
                    pygame.draw.line(surface, tool, (r.x + 5, r.y + 6), (r.x + 12, r.y + 10), 1)
                    pygame.draw.line(surface, (10, 10, 12), (r.x + 5, r.y + 6), (r.x + 12, r.y + 10), 1)
                    pygame.draw.rect(surface, tool, pygame.Rect(r.x + 5, r.y + 11, 5, 2), border_radius=1)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 5, r.y + 11, 5, 2), 1, border_radius=1)
                elif ch == "C":
                    pygame.draw.rect(surface, seat, pygame.Rect(r.x + 4, r.y + 4, r.w - 8, r.h - 8), border_radius=3)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 4, r.y + 4, r.w - 8, r.h - 8), 1, border_radius=3)
                    pygame.draw.rect(surface, seat2, pygame.Rect(r.x + 6, r.y + 6, r.w - 12, 4))

        p = pygame.Vector2(self.rv_int_pos)
        speed2 = float(self.rv_int_vel.length_squared())
        face = pygame.Vector2(self.rv_int_facing)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)
        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"
        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = pf.get(d, pf["down"])
        if speed2 <= 0.2 or len(frames) <= 1:
            base = frames[0]
        else:
            walk = frames[1:]
            phase = (float(self.rv_int_walk_phase) % math.tau) / math.tau
            idx = int(phase * len(walk)) % len(walk)
            base = walk[idx]

        scale = int(max(1, int(self._RV_INT_SPRITE_SCALE)))
        spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
        sx = int(round(map_x + p.x))
        sy = int(round(map_y + p.y))
        shadow = pygame.Rect(0, 0, 14, 6)
        shadow.center = (sx, sy + 8)
        pygame.draw.ellipse(surface, (0, 0, 0), shadow)
        rect = spr.get_rect()
        rect.midbottom = (sx, sy + 12)
        surface.blit(spr, rect)

    def _draw_sch_interior_scene(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        tile = int(self._SCH_INT_TILE_SIZE)
        map_w = int(self._SCH_INT_W) * tile
        map_h = int(self._SCH_INT_H) * tile
        map_x = (INTERNAL_W - map_w) // 2
        map_y = 38
        panel = pygame.Rect(map_x - 10, map_y - 10, map_w + 20, map_h + 20)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (70, 70, 86), panel, 2, border_radius=12)

        floor = int(getattr(self, "sch_floor", 1))
        subtitle = "大厅" if floor <= 1 else f"{floor}F"
        draw_text(
            surface,
            self.app.font_s,
            f"学校 - {subtitle}",
            (INTERNAL_W // 2, panel.top - 14),
            pygame.Color(240, 240, 240),
            anchor="center",
        )

        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        surface.blit(self._interior_floor_surface(map_w, map_h, kind="school"), map_rect.topleft)

        wall = (26, 26, 32)
        wall_hi = (54, 54, 68)
        edge = (12, 12, 16)
        locker = (78, 90, 96)
        locker_hi = (110, 128, 136)
        desk = (96, 78, 56)
        desk_hi = (140, 116, 86)
        elev = (62, 62, 72)
        elev_hi = (100, 100, 116)
        stairs = (56, 60, 74)
        stairs_hi = (92, 98, 120)
        door = (96, 70, 46)
        door_hi = (132, 100, 68)

        for y in range(int(self._SCH_INT_H)):
            for x in range(int(self._SCH_INT_W)):
                ch = self._sch_int_char_at(x, y)
                r = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)
                if ch == "W":
                    pygame.draw.rect(surface, wall, r)
                    pygame.draw.rect(surface, edge, r, 1)
                    pygame.draw.line(surface, wall_hi, (r.left + 1, r.top + 1), (r.right - 2, r.top + 1), 1)
                elif ch == "V":
                    view = self._window_view_surface(tile - 6, tile - 6)
                    surface.blit(view, (r.x + 3, r.y + 3))
                    pygame.draw.rect(surface, (20, 20, 26), pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), 1)
                    pygame.draw.rect(surface, (70, 70, 86), pygame.Rect(r.x + 1, r.y + 1, tile - 2, tile - 2), 1)
                elif ch == "L":
                    pygame.draw.rect(surface, locker, pygame.Rect(r.x + 3, r.y + 2, tile - 6, tile - 4), border_radius=2)
                    pygame.draw.rect(surface, edge, pygame.Rect(r.x + 3, r.y + 2, tile - 6, tile - 4), 1, border_radius=2)
                    pygame.draw.rect(surface, locker_hi, pygame.Rect(r.x + 4, r.y + 4, tile - 8, 3))
                    pygame.draw.line(surface, edge, (r.x + tile // 2, r.y + 3), (r.x + tile // 2, r.y + tile - 3), 1)
                elif ch == "T":
                    pygame.draw.rect(surface, desk, pygame.Rect(r.x + 2, r.y + 5, tile - 4, tile - 8), border_radius=2)
                    pygame.draw.rect(surface, edge, pygame.Rect(r.x + 2, r.y + 5, tile - 4, tile - 8), 1, border_radius=2)
                    pygame.draw.rect(surface, desk_hi, pygame.Rect(r.x + 3, r.y + 6, tile - 6, 3))
                elif ch == "E":
                    pygame.draw.rect(surface, elev, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), border_radius=2)
                    pygame.draw.rect(surface, edge, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), 1, border_radius=2)
                    pygame.draw.rect(surface, elev_hi, pygame.Rect(r.x + 3, r.y + 4, tile - 6, 3))
                    pygame.draw.line(surface, edge, (r.x + tile // 2, r.y + 4), (r.x + tile // 2, r.y + tile - 4), 1)
                elif ch in ("^", "v"):
                    pygame.draw.rect(surface, stairs, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), border_radius=2)
                    pygame.draw.rect(surface, edge, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), 1, border_radius=2)
                    pygame.draw.rect(surface, stairs_hi, pygame.Rect(r.x + 3, r.y + 4, tile - 6, 3))
                    cx = r.centerx
                    cy = r.centery + (1 if ch == "v" else -1)
                    if ch == "^":
                        pygame.draw.polygon(surface, (230, 230, 240), [(cx, cy - 4), (cx - 4, cy + 3), (cx + 4, cy + 3)])
                    else:
                        pygame.draw.polygon(surface, (230, 230, 240), [(cx, cy + 4), (cx - 4, cy - 3), (cx + 4, cy - 3)])
                elif ch == "D":
                    pygame.draw.rect(surface, door, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), border_radius=2)
                    pygame.draw.rect(surface, edge, pygame.Rect(r.x + 2, r.y + 2, tile - 4, tile - 4), 1, border_radius=2)
                    pygame.draw.rect(surface, door_hi, pygame.Rect(r.x + 3, r.y + 4, tile - 6, 3))
                    pygame.draw.circle(surface, (30, 30, 34), (r.right - 6, r.centery), 1)

        p = pygame.Vector2(self.sch_int_pos)
        speed2 = float(self.sch_int_vel.length_squared())
        face = pygame.Vector2(self.sch_int_facing)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)
        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"

        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        frames = pf_walk.get(d, pf_walk["down"])
        if speed2 <= 0.2 or len(frames) <= 1:
            base = frames[0]
        else:
            walk = frames[1:]
            phase = (float(self.sch_int_walk_phase) % math.tau) / math.tau
            idx = int(phase * len(walk)) % len(walk)
            base = walk[idx]

        scale = int(max(1, int(self._SCH_INT_SPRITE_SCALE)))
        spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
        sx = int(round(map_x + p.x))
        sy = int(round(map_y + p.y))
        shadow = pygame.Rect(0, 0, 14, 6)
        shadow.center = (sx, sy + 8)
        pygame.draw.ellipse(surface, (0, 0, 0), shadow)
        rect = spr.get_rect()
        rect.midbottom = (sx, sy + 12)
        surface.blit(spr, rect)

    def _draw_hr_interior_scene(self, surface: pygame.Surface) -> None:  
        surface.fill((10, 10, 14))
        tile = int(self._HR_INT_TILE_SIZE)
        map_w = int(self._HR_INT_W) * tile
        map_h = int(self._HR_INT_H) * tile
        map_x = (INTERNAL_W - map_w) // 2
        map_y = 38
        panel = pygame.Rect(map_x - 10, map_y - 10, map_w + 20, map_h + 20)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (70, 70, 86), panel, 2, border_radius=12)

        # Cozier indoor rendering (wood floors, furniture detail, window view).
        mode = str(getattr(self, "hr_mode", "lobby"))
        floor = int(getattr(self, "hr_floor", 0))

        title = "高层住宅"
        home_room = str(getattr(self, "home_highrise_room", "")).strip()
        home_floor = int(getattr(self, "hr_home_floor", -1))
        if mode == "home":
            subtitle = f"你的家 {home_room}" if home_room else "你的家"
        elif floor <= 0:
            subtitle = "大厅"
        else:
            subtitle = f"{floor} 层"
            if home_floor == floor:
                subtitle += f"（家 {home_room}）" if home_room else "（家）"
        draw_text(
            surface,
            self.app.font_s,
            f"{title} - {subtitle}",
            (INTERNAL_W // 2, panel.top - 14),
            pygame.Color(240, 240, 240),
            anchor="center",
        )

        # Persistent home guidance (so the player always knows where the room is).
        info = ""
        if home_floor > 0:
            if home_room:
                info = f"家：{home_room}（{home_floor}F）"
            else:
                info = f"家：{home_floor}F"
            if mode == "hall":
                if floor == home_floor:
                    info += "  本层：绿色门"
                else:
                    info += f"  现在：{floor}F"
        if info:
            draw_text(
                surface,
                self.app.font_s,
                info,
                (panel.right - 8, panel.top + 6),
                pygame.Color(200, 200, 210),
                anchor="topright",
            )

        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        floor_kind = "home" if mode == "home" else "stone"
        surface.blit(self._interior_floor_surface(map_w, map_h, kind=floor_kind), map_rect.topleft)

        # "Back wall" band (front-view cue) for the player's home, per reference.
        if mode == "home":
            wall_h = int(tile * 2)
            wall_h = int(clamp(wall_h, 18, max(18, map_h - 40)))
            wall_rect = pygame.Rect(int(map_x), int(map_y), int(map_w), int(wall_h))
            wall_surf = pygame.Surface((wall_rect.w, wall_rect.h))
            wall_base = (188, 186, 182)
            wall_surf.fill(wall_base)
            # Subtle horizontal noise lines.
            for yy in range(2, wall_rect.h - 6, 6):
                col = (182, 180, 176) if ((yy // 6) % 2) == 0 else (190, 188, 184)
                wall_surf.fill(col, pygame.Rect(0, int(yy), wall_rect.w, 1))
            # Baseboard.
            base_y = int(max(0, wall_rect.h - 5))
            wall_surf.fill((74, 58, 42), pygame.Rect(0, base_y, wall_rect.w, 4))
            wall_surf.fill((12, 12, 16), pygame.Rect(0, base_y, wall_rect.w, 1))
            surface.blit(wall_surf, wall_rect.topleft)

        wall = (26, 26, 32)
        wall_hi = (56, 56, 70)
        wall_edge = (12, 12, 16)
        frame = (88, 80, 70)
        frame2 = (62, 56, 48)
        outline = (10, 10, 12)

        wood = (98, 78, 56)
        wood2 = (74, 58, 42)
        steel = (216, 220, 228)
        steel2 = (134, 140, 152)
        bed_matt = (186, 170, 156)
        bed_blank = (142, 128, 170)
        bed_blank2 = (120, 106, 144)
        pillow = (238, 238, 244)
        seat = (72, 86, 148)
        seat2 = (54, 64, 118)

        wkind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        glass_w = max(4, int(tile - 6))
        glass_h = max(4, int(tile - 8))
        win_view = self._window_view_surface(glass_w, glass_h)

        # Rug under the table block in "home" mode.
        if mode == "home":
            t_tiles: list[tuple[int, int]] = []
            for yy, row in enumerate(getattr(self, "hr_layout", [])):
                if not isinstance(row, str):
                    continue
                for xx in range(min(len(row), int(self._HR_INT_W))):
                    if row[xx] == "T":
                        t_tiles.append((int(xx), int(yy)))
            if t_tiles:
                min_tx = min(p[0] for p in t_tiles)
                max_tx = max(p[0] for p in t_tiles)
                min_ty = min(p[1] for p in t_tiles)
                max_ty = max(p[1] for p in t_tiles)
                rug = pygame.Rect(
                    map_x + min_tx * tile - 3,
                    map_y + min_ty * tile + 7,
                    (max_tx - min_tx + 1) * tile + 6,
                    (max_ty - min_ty + 1) * tile + 8,
                )
                rug.clamp_ip(map_rect.inflate(-6, -6))
                rug_s = pygame.Surface((rug.w, rug.h), pygame.SRCALPHA)
                rug_s.fill((132, 70, 82, 190))
                for y in range(2, rug.h, 5):
                    pygame.draw.line(rug_s, (210, 150, 150, 55), (2, y), (rug.w - 3, y), 1)
                pygame.draw.rect(rug_s, (40, 18, 20, 220), rug_s.get_rect(), 2, border_radius=4)
                surface.blit(rug_s, rug.topleft)

        bed_done: set[tuple[int, int]] = set()
        elev_done: set[tuple[int, int]] = set()
        s_done: set[tuple[int, int]] = set()
        f_done: set[tuple[int, int]] = set()
        k_done: set[tuple[int, int]] = set()
        t_done: set[tuple[int, int]] = set()
        c_done: set[tuple[int, int]] = set()

        layout = getattr(self, "hr_layout", [])
        for y in range(int(self._HR_INT_H)):
            row = layout[y] if 0 <= y < len(layout) else ""
            for x in range(int(self._HR_INT_W)):
                ch = row[x] if x < len(row) else "W"
                r = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)

                if ch in ("W", "V", "D", "A", "H"):
                    pygame.draw.rect(surface, wall, r)
                    pygame.draw.rect(surface, wall_hi, pygame.Rect(r.x, r.y, r.w, 4))
                    pygame.draw.rect(surface, wall_edge, r, 1)
                    pygame.draw.rect(surface, frame2, pygame.Rect(r.x, r.bottom - 4, r.w, 2))

                    if ch == "V":
                        glass = pygame.Rect(r.x + 3, r.y + 4, r.w - 6, r.h - 8)
                        wv = win_view
                        if wv.get_width() != glass.w or wv.get_height() != glass.h:
                            wv = pygame.transform.scale(wv, (int(glass.w), int(glass.h)))
                        surface.blit(wv, glass.topleft)
                        if wkind in ("rain", "storm", "snow"):
                            self._draw_window_precip(surface, glass, kind=wkind, intensity=inten)
                        pygame.draw.rect(surface, frame, glass.inflate(4, 4), 1, border_radius=2)
                        pygame.draw.rect(surface, outline, glass, 1)
                        pygame.draw.line(surface, (40, 40, 50), (glass.centerx, glass.top + 1), (glass.centerx, glass.bottom - 2), 1)
                    elif ch == "D":
                        dr = pygame.Rect(r.x + 5, r.y + 4, r.w - 10, r.h - 8)
                        pygame.draw.rect(surface, (240, 220, 140), dr, border_radius=2)
                        pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                        pygame.draw.circle(surface, outline, (dr.right - 5, dr.centery), 2)
                    elif ch in ("A", "H"):
                        # Apartment doors: keep a solid wall frame, bottom aligned to the baseboard.
                        # Doors should be taller than one tile (player sprite is 2x inside).
                        door_h = int(clamp(int(tile * 1.7), int(tile + 8), int(tile * 2 - 2)))
                        dr = pygame.Rect(r.x + 2, int(r.bottom - 2 - door_h), r.w - 4, door_h)
                        pygame.draw.rect(surface, wood, dr, border_radius=2)
                        pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                        pygame.draw.rect(surface, wood2, pygame.Rect(dr.x + 2, dr.y + 2, dr.w - 4, 4))
                        knob_y = int(dr.y + dr.h * 0.62)
                        pygame.draw.circle(surface, (240, 220, 140), (dr.right - 4, knob_y), 2)
                        if ch == "H":
                            pygame.draw.rect(surface, (120, 200, 140), pygame.Rect(dr.x + 2, dr.y + 2, dr.w - 4, 3), border_radius=1)
                    continue

                # Objects over the floor texture.
                if ch == "E":
                    if (x, y) in elev_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "E":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "E":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "E":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            elev_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    er = br.inflate(-4, -4)
                    sh = pygame.Surface((er.w, er.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 80), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (er.x + 2, er.y + 3))
                    pygame.draw.rect(surface, steel, er, border_radius=4)
                    pygame.draw.rect(surface, outline, er, 1, border_radius=4)
                    pygame.draw.rect(surface, steel2, pygame.Rect(er.x + 3, er.y + 3, er.w - 6, 3))
                    pygame.draw.line(surface, outline, (er.centerx, er.y + 4), (er.centerx, er.bottom - 5), 1)
                    pygame.draw.rect(surface, (255, 220, 140), pygame.Rect(er.right - 9, er.y + 10, 3, 5), border_radius=1)
                    continue

                if ch == "B":
                    if (int(x), int(y)) in bed_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "B":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "B":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "B":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            bed_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    bed_r = br.inflate(-4, -8)
                    sh = pygame.Surface((bed_r.w, bed_r.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 90), sh.get_rect(), border_radius=6)
                    surface.blit(sh, (bed_r.x + 2, bed_r.y + 3))
                    pygame.draw.rect(surface, bed_matt, bed_r, border_radius=6)
                    pygame.draw.rect(surface, outline, bed_r, 1, border_radius=6)
                    hb = pygame.Rect(bed_r.x, bed_r.y, 8, bed_r.h)
                    pygame.draw.rect(surface, wood2, hb, border_radius=5)
                    pygame.draw.rect(surface, outline, hb, 1, border_radius=5)
                    p1 = pygame.Rect(bed_r.x + 10, bed_r.y + 4, max(10, bed_r.w // 5), 7)
                    p2 = pygame.Rect(p1.x, p1.y + 10, p1.w, 7)
                    pygame.draw.rect(surface, pillow, p1, border_radius=3)
                    pygame.draw.rect(surface, pillow, p2, border_radius=3)
                    pygame.draw.rect(surface, outline, p1, 1, border_radius=3)
                    pygame.draw.rect(surface, outline, p2, 1, border_radius=3)
                    bl = pygame.Rect(bed_r.x + 10, bed_r.y + 20, bed_r.w - 14, max(10, bed_r.h - 24))
                    pygame.draw.rect(surface, bed_blank, bl, border_radius=5)
                    pygame.draw.rect(surface, outline, bl, 1, border_radius=5)
                    for sx in range(bl.x + 6, bl.right - 4, 10):
                        pygame.draw.line(surface, bed_blank2, (sx, bl.y + 2), (sx, bl.bottom - 3), 1)
                    # Bed legs/frame (so it doesn't look like it floats).
                    floor_y = int(br.bottom - 3)
                    leg_y0 = int(bed_r.bottom - 2)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(wood2, add=(-12, -12, -12))
                        for lx in (int(bed_r.x + 6), int(bed_r.right - 8)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                            surface.fill(leg_col, pygame.Rect(int(lx - 1), int(floor_y), 4, 1))
                    continue

                if ch == "K":
                    if (x, y) in k_done:
                        continue

                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "K":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "K":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "K":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            k_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    # Less inset so the countertop reads flush to the block edges.
                    obj = br.inflate(-2, -6)
                    sh = pygame.Surface((obj.w, obj.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 70), sh.get_rect(), border_radius=6)
                    surface.blit(sh, (obj.x + 2, obj.y + 3))

                    # Marble countertop (big visible top, thin front face).
                    top_h = int(clamp(int(obj.h * 0.58), 12, 18))
                    top_r = pygame.Rect(obj.x, obj.y, obj.w, top_h)
                    front_r = pygame.Rect(obj.x, obj.y + top_h - 1, obj.w, obj.h - (top_h - 1))

                    stone_face = (170, 172, 178)
                    stone_face2 = (150, 152, 160)
                    marble_base = (228, 230, 236)

                    pygame.draw.rect(surface, stone_face, front_r, border_radius=6)
                    pygame.draw.rect(surface, outline, front_r, 1, border_radius=6)
                    pygame.draw.rect(surface, stone_face2, pygame.Rect(front_r.x + 1, front_r.y + 2, front_r.w - 2, 3))

                    marble = self._marble_surface(
                        int(top_r.w),
                        int(top_r.h),
                        seed=int(getattr(self, "seed", 0)) ^ (int(x) * 65537) ^ (int(y) * 9719) ^ 0x51F15EED,
                        base=marble_base,
                    )
                    surface.blit(marble, top_r.topleft)
                    pygame.draw.rect(surface, outline, top_r, 1, border_radius=6)
                    pygame.draw.line(surface, outline, (top_r.x + 2, top_r.bottom - 1), (top_r.right - 3, top_r.bottom - 1), 1)

                    # Cabinet seams / handles on the front face (stone drawers).
                    modules = int(max(2, int(bw)))
                    for i in range(1, modules):
                        xline = int(front_r.x + (front_r.w * i) / modules)
                        pygame.draw.line(surface, stone_face2, (xline, front_r.y + 5), (xline, front_r.bottom - 5), 2)
                        pygame.draw.line(surface, outline, (xline, front_r.y + 5), (xline, front_r.bottom - 5), 1)
                    for i in range(modules):
                        cx = int(front_r.x + (front_r.w * (i + 0.5)) / modules)
                        hy = int(front_r.y + front_r.h * 0.55)
                        hrect = pygame.Rect(cx - 2, hy - 1, 4, 2)
                        surface.fill((80, 82, 90), hrect)
                        pygame.draw.rect(surface, outline, hrect, 1)

                    # Sink + stove on the top.
                    sink_w = int(clamp(int(top_r.w * 0.20), 14, 22))
                    sink_h = int(clamp(int(top_r.h - 5), 6, 10))
                    sink = pygame.Rect(top_r.x + int(top_r.w * 0.18), top_r.y + 3, sink_w, sink_h)
                    pygame.draw.rect(surface, steel, sink, border_radius=3)
                    pygame.draw.rect(surface, outline, sink, 1, border_radius=3)
                    pygame.draw.circle(surface, (60, 60, 70), (sink.right - 3, sink.y + 3), 1)
                    pygame.draw.line(surface, steel2, (sink.x + 2, sink.y + 1), (sink.x + 2, sink.y - 3), 1)
                    pygame.draw.line(surface, steel2, (sink.x + 2, sink.y - 3), (sink.x + 7, sink.y - 3), 1)

                    stove_w = int(clamp(int(top_r.w * 0.20), 14, 22))
                    stove_h = sink_h
                    stove = pygame.Rect(top_r.right - int(top_r.w * 0.16) - stove_w, top_r.y + 3, stove_w, stove_h)
                    pygame.draw.rect(surface, (40, 40, 50), stove, border_radius=3)
                    pygame.draw.rect(surface, outline, stove, 1, border_radius=3)
                    pygame.draw.circle(surface, (80, 80, 90), (stove.x + stove_w // 3, stove.centery), 2)
                    pygame.draw.circle(surface, (80, 80, 90), (stove.right - stove_w // 3, stove.centery), 2)

                    # Small items so it reads as a real kitchen.
                    plate = pygame.Rect(top_r.right - 15, top_r.y + 3, 12, 5)
                    pygame.draw.ellipse(surface, (238, 238, 244), plate)
                    pygame.draw.ellipse(surface, outline, plate, 1)
                    bowl = pygame.Rect(top_r.x + 8, top_r.y + 3, 10, 5)
                    pygame.draw.ellipse(surface, (205, 214, 224), bowl)
                    pygame.draw.ellipse(surface, outline, bowl, 1)
                    pot = pygame.Rect(top_r.centerx - 8, top_r.y + 2, 16, 6)
                    pygame.draw.rect(surface, (40, 40, 50), pot, border_radius=3)
                    pygame.draw.rect(surface, outline, pot, 1, border_radius=3)
                    pygame.draw.line(surface, steel2, (pot.x + 2, pot.y + 2), (pot.right - 3, pot.y + 2), 1)

                    # Subtle shelf/rack cue above the counter.
                    rack_y = int(top_r.y - 7)
                    if rack_y >= int(br.y) + 1:
                        pygame.draw.line(surface, outline, (top_r.x + 2, rack_y), (top_r.right - 3, rack_y), 1)
                        for ux in (top_r.x + 10, top_r.centerx, top_r.right - 12):
                            pygame.draw.line(surface, steel2, (int(ux), rack_y + 1), (int(ux), rack_y + 4), 1)
                    continue

                if ch == "T":
                    if (x, y) in t_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "T":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "T":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "T":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            t_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    top = br.inflate(-6, -10)
                    sh = pygame.Surface((top.w, top.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 70), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (top.x + 2, top.y + 3))
                    pygame.draw.rect(surface, wood, top, border_radius=4)
                    pygame.draw.rect(surface, outline, top, 1, border_radius=4)
                    pygame.draw.rect(surface, wood2, pygame.Rect(top.x + 1, top.y + 1, top.w - 2, 4))
                    # Small TV / radio-like decoration.
                    screen = pygame.Rect(top.centerx - 10, top.y + 5, 20, 10)
                    pygame.draw.rect(surface, (36, 38, 44), screen, border_radius=3)
                    pygame.draw.rect(surface, steel2, screen, 1, border_radius=3)
                    pygame.draw.rect(surface, (120, 200, 240), pygame.Rect(screen.x + 3, screen.y + 3, 7, 4))
                    # Table legs (front-view cue).
                    floor_y = int(br.bottom - 3)
                    leg_y0 = int(top.bottom - 1)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0 and top.w >= 12:
                        leg_col = self._tint(wood2, add=(-10, -10, -10))
                        for lx in (int(top.x + 4), int(top.right - 6)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                        # Simple crossbar.
                        if leg_h >= 4:
                            yb = int(floor_y - 2)
                            surface.fill(leg_col, pygame.Rect(int(top.x + 4), yb, int(max(1, top.w - 8)), 1))
                    continue

                if ch == "S":
                    if (x, y) in s_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "S":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "S":  
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "S":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            s_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    # Keep cabinets tall enough even when the block is only 1 tile high (e.g., the bottom-left cabinet).
                    obj = br.inflate(-6, -4)
                    sh = pygame.Surface((obj.w, obj.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 70), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (obj.x + 2, obj.y + 3))
                    # Wardrobe / cabinet: draw top + front face (two visible planes).
                    cab_top = self._tint(wood, add=(14, 12, 10))
                    cab_front = self._tint(wood2, add=(2, 2, 2))
                    cab_front2 = self._tint(cab_front, add=(-16, -14, -12))
                    hi = self._tint(cab_top, add=(22, 22, 22))
                    lo = self._tint(cab_front, add=(-26, -24, -22))

                    top_h = int(clamp(int(obj.h * 0.34), 8, 12))
                    top_r = pygame.Rect(obj.x, obj.y, obj.w, top_h)
                    front_r = pygame.Rect(obj.x, obj.y + top_h - 1, obj.w, obj.h - (top_h - 1))

                    pygame.draw.rect(surface, cab_front, front_r, border_radius=6)
                    pygame.draw.rect(surface, outline, front_r, 1, border_radius=6)
                    pygame.draw.rect(surface, cab_front2, pygame.Rect(front_r.x + 1, front_r.y + 2, front_r.w - 2, 3))

                    pygame.draw.rect(surface, cab_top, top_r, border_radius=6)
                    pygame.draw.rect(surface, outline, top_r, 1, border_radius=6)
                    pygame.draw.line(surface, hi, (top_r.x + 2, top_r.y + 1), (top_r.right - 3, top_r.y + 1), 1)
                    pygame.draw.line(surface, outline, (top_r.x + 2, top_r.bottom - 1), (top_r.right - 3, top_r.bottom - 1), 1)

                    # Inset doors live on the front face (keeps the top readable).
                    door_area = front_r.inflate(-4, -6)
                    if door_area.w > 6 and door_area.h > 8:
                        pygame.draw.rect(surface, self._tint(cab_front, add=(-8, -6, -4)), door_area, border_radius=4)
                        pygame.draw.rect(surface, outline, door_area, 1, border_radius=4)
                    else:
                        door_area = pygame.Rect(front_r)

                    # Tiny feet so it doesn't look like it floats.
                    floor_y = int(br.bottom - 3)
                    leg_y0 = int(front_r.bottom - 2)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(cab_front2, add=(-16, -16, -16))
                        for lx in (int(front_r.x + 6), int(front_r.right - 8)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)

                    open_block = getattr(self, "home_ui_open_block", None)
                    is_open = bool(getattr(self, "home_ui_open", False)) and open_block == (
                        int(x),
                        int(y),
                        int(x) + int(bw) - 1,
                        int(y) + int(bh) - 1,
                    )
                    if is_open:
                        inner = door_area.inflate(-10, -10)
                        if inner.w > 4 and inner.h > 4:
                            pygame.draw.rect(surface, (20, 20, 24), inner, border_radius=3)
                            pygame.draw.rect(surface, outline, inner, 1, border_radius=3)
                            # Inside top surface + shelves.
                            pygame.draw.line(surface, self._tint(cab_front2, add=(18, 18, 18)), (inner.x + 2, inner.y + 2), (inner.right - 3, inner.y + 2), 1)
                            step = max(6, int(inner.h // 3))
                            for sy in range(inner.y + step, inner.bottom - 2, step):
                                pygame.draw.line(surface, cab_front2, (inner.x + 2, sy), (inner.right - 3, sy), 2)
                                pygame.draw.line(surface, outline, (inner.x + 2, sy), (inner.right - 3, sy), 1)
                            # Small visible supplies.
                            can = pygame.Rect(inner.x + 4, inner.y + 5, 5, 7)
                            surface.fill((180, 60, 70), can)
                            pygame.draw.rect(surface, outline, can, 1)
                            jar = pygame.Rect(inner.right - 10, inner.y + 6, 7, 6)
                            surface.fill((90, 140, 190), jar)
                            pygame.draw.rect(surface, outline, jar, 1)

                        # Doors "slid open" to sides.
                        door_w = max(6, int(door_area.w // 5))
                        ld = pygame.Rect(door_area.x + 1, door_area.y + 3, door_w, door_area.h - 6)
                        rd = pygame.Rect(door_area.right - door_w - 1, door_area.y + 3, door_w, door_area.h - 6)
                        for d in (ld, rd):
                            pygame.draw.rect(surface, cab_front, d, border_radius=3)
                            pygame.draw.rect(surface, outline, d, 1, border_radius=3)
                            inset = d.inflate(-2, -4)
                            if inset.w > 2 and inset.h > 2:
                                pygame.draw.rect(surface, self._tint(cab_front, add=(-8, -8, -8)), inset, border_radius=2)
                                pygame.draw.rect(surface, outline, inset, 1, border_radius=2)
                                pygame.draw.line(surface, hi, (inset.x + 1, inset.y + 1), (inset.right - 2, inset.y + 1), 1)
                        for d in (ld, rd):
                            hy = int(d.y + d.h * 0.55)
                            hrect = pygame.Rect(d.centerx - 1, hy - 3, 2, 6)
                            surface.fill((220, 220, 230), hrect)
                            pygame.draw.rect(surface, outline, hrect, 1)
                    else:
                        # Closed doors + handles.
                        doors = max(2, int(door_area.w // 24))
                        doors = int(clamp(doors, 2, 4))
                        y_top = int(door_area.y + 2)
                        y_bot = int(door_area.bottom - 3)
                        for i in range(1, doors):
                            xline = int(door_area.x + (door_area.w * i) / doors)
                            pygame.draw.line(surface, cab_front2, (xline, y_top), (xline, y_bot), 2)
                            pygame.draw.line(surface, outline, (xline, y_top), (xline, y_bot), 1)
                        for i in range(doors):
                            x0 = int(door_area.x + (door_area.w * i) / doors)
                            x1 = int(door_area.x + (door_area.w * (i + 1)) / doors)
                            panel = pygame.Rect(x0 + 2, y_top + 1, (x1 - x0) - 4, max(2, int(y_bot - y_top - 2)))
                            if panel.w > 3 and panel.h > 3:
                                pygame.draw.rect(surface, self._tint(cab_front, add=(-6, -6, -6)), panel, border_radius=2)
                                pygame.draw.rect(surface, outline, panel, 1, border_radius=2)
                                pygame.draw.line(surface, hi, (panel.x + 1, panel.y + 1), (panel.right - 2, panel.y + 1), 1)

                            hx = int(door_area.x + (door_area.w * (i + 0.5)) / doors)
                            hy = int(door_area.y + door_area.h * 0.55)
                            hrect = pygame.Rect(hx - 1, hy - 3, 2, 6)
                            surface.fill((220, 220, 230), hrect)
                            pygame.draw.rect(surface, outline, hrect, 1)
                    continue

                if ch == "F":
                    if (x, y) in f_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "F":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "F":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "F":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            f_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    obj = br.inflate(-4, -4)
                    sh = pygame.Surface((obj.w, obj.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 70), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (obj.x + 2, obj.y + 3))

                    fridge_top = (238, 240, 246)
                    fridge_front = (210, 214, 224)
                    fridge_front2 = (178, 182, 194)
                    hi = (248, 248, 252)

                    top_h = int(clamp(int(obj.h * 0.34), 8, 12))
                    top_r = pygame.Rect(obj.x, obj.y, obj.w, top_h)
                    front_r = pygame.Rect(obj.x, obj.y + top_h - 1, obj.w, obj.h - (top_h - 1))

                    pygame.draw.rect(surface, fridge_front, front_r, border_radius=6)
                    pygame.draw.rect(surface, outline, front_r, 1, border_radius=6)
                    pygame.draw.rect(surface, fridge_front2, pygame.Rect(front_r.x + 1, front_r.y + 2, front_r.w - 2, 3))

                    pygame.draw.rect(surface, fridge_top, top_r, border_radius=6)
                    pygame.draw.rect(surface, outline, top_r, 1, border_radius=6)
                    pygame.draw.line(surface, hi, (top_r.x + 2, top_r.y + 1), (top_r.right - 3, top_r.y + 1), 1)
                    pygame.draw.line(surface, outline, (top_r.x + 2, top_r.bottom - 1), (top_r.right - 3, top_r.bottom - 1), 1)

                    door_area = front_r.inflate(-4, -6)
                    if door_area.w <= 6 or door_area.h <= 8:
                        door_area = pygame.Rect(front_r)

                    open_block = getattr(self, "home_ui_open_block", None)
                    is_open = bool(getattr(self, "home_ui_open", False)) and open_block == (
                        int(x),
                        int(y),
                        int(x) + int(bw) - 1,
                        int(y) + int(bh) - 1,
                    )
                    if is_open:
                        inner = door_area.inflate(-8, -8)
                        if inner.w > 4 and inner.h > 4:
                            pygame.draw.rect(surface, (20, 20, 24), inner, border_radius=3)
                            pygame.draw.rect(surface, outline, inner, 1, border_radius=3)
                            for sy in range(inner.y + 4, inner.bottom - 2, 6):
                                pygame.draw.line(surface, (44, 44, 52), (inner.x + 2, sy), (inner.right - 3, sy), 1)
                            bottle = pygame.Rect(inner.x + 3, inner.bottom - 9, 4, 7)
                            surface.fill((120, 170, 230), bottle)
                            pygame.draw.rect(surface, outline, bottle, 1)
                            can = pygame.Rect(inner.right - 8, inner.bottom - 8, 5, 6)
                            surface.fill((220, 70, 80), can)
                            pygame.draw.rect(surface, outline, can, 1)

                        # Doors slid to sides.
                        door_w = max(6, int(door_area.w // 5))
                        ld = pygame.Rect(door_area.x + 1, door_area.y + 3, door_w, door_area.h - 6)
                        rd = pygame.Rect(door_area.right - door_w - 1, door_area.y + 3, door_w, door_area.h - 6)
                        for d in (ld, rd):
                            pygame.draw.rect(surface, fridge_front, d, border_radius=3)
                            pygame.draw.rect(surface, outline, d, 1, border_radius=3)
                            inset = d.inflate(-2, -4)
                            if inset.w > 2 and inset.h > 2:
                                pygame.draw.rect(surface, self._tint(fridge_front, add=(-8, -8, -8)), inset, border_radius=2)
                                pygame.draw.rect(surface, outline, inset, 1, border_radius=2)
                                pygame.draw.line(surface, hi, (inset.x + 1, inset.y + 1), (inset.right - 2, inset.y + 1), 1)
                    else:
                        # Freezer split line + handle + magnets.
                        split_y = int(door_area.y + door_area.h * 0.38)
                        pygame.draw.line(surface, fridge_front2, (door_area.x + 2, split_y), (door_area.right - 3, split_y), 2)
                        pygame.draw.line(surface, outline, (door_area.x + 2, split_y), (door_area.right - 3, split_y), 1)
                        handle = pygame.Rect(door_area.right - 5, door_area.y + 4, 2, max(6, door_area.h - 8))
                        surface.fill(steel2, handle)
                        pygame.draw.rect(surface, outline, handle, 1)
                        for mx, my, col in (
                            (door_area.x + 3, door_area.y + 4, (240, 120, 120)),
                            (door_area.x + 6, door_area.y + 7, (120, 200, 140)),
                            (door_area.x + 4, door_area.y + 10, (240, 220, 140)),
                        ):
                            mr = pygame.Rect(int(mx), int(my), 3, 3)
                            surface.fill(col, mr)
                            pygame.draw.rect(surface, outline, mr, 1)

                    # Tiny feet so it doesn't look like it floats.
                    floor_y = int(br.bottom - 3)
                    leg_y0 = int(front_r.bottom - 2)
                    leg_h = int(floor_y - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(fridge_front2, add=(-16, -16, -16))
                        for lx in (int(front_r.x + 6), int(front_r.right - 8)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                    continue

                if ch == "C":
                    if (x, y) in c_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "C":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "C":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "C":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            c_done.add((int(x) + int(dx), int(y) + int(dy)))

                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    obj = br.inflate(-6, -10)
                    sh = pygame.Surface((obj.w, obj.h), pygame.SRCALPHA)
                    pygame.draw.rect(sh, (0, 0, 0, 60), sh.get_rect(), border_radius=5)
                    surface.blit(sh, (obj.x + 2, obj.y + 3))

                    is_sofa = bool(mode == "home" and (int(bw) >= 3 or int(bh) >= 2))
                    if is_sofa:
                        back_h = int(clamp(int(obj.h // 3), 7, 12))
                        back = pygame.Rect(obj.x + 1, obj.y + 1, obj.w - 2, back_h)
                        seat_r = pygame.Rect(obj.x, obj.y + back_h - 1, obj.w, obj.h - back_h + 1)
                        pygame.draw.rect(surface, seat, seat_r, border_radius=6)
                        pygame.draw.rect(surface, outline, seat_r, 1, border_radius=6)
                        pygame.draw.rect(surface, seat2, back, border_radius=5)
                        pygame.draw.rect(surface, outline, back, 1, border_radius=5)
                        arm_w = int(clamp(int(obj.w // 10), 4, 8))
                        arm_l = pygame.Rect(seat_r.x, back.bottom - 2, arm_w, seat_r.h - 2)
                        arm_r = pygame.Rect(seat_r.right - arm_w, back.bottom - 2, arm_w, seat_r.h - 2)
                        pygame.draw.rect(surface, seat2, arm_l, border_radius=5)
                        pygame.draw.rect(surface, seat2, arm_r, border_radius=5)
                        pygame.draw.rect(surface, outline, arm_l, 1, border_radius=5)
                        pygame.draw.rect(surface, outline, arm_r, 1, border_radius=5)
                        for i in (1, 2):
                            xline = int(seat_r.x + (seat_r.w * i) / 3)
                            pygame.draw.line(surface, outline, (xline, seat_r.y + 3), (xline, seat_r.bottom - 4), 1)
                        if back.w >= 24:
                            p_w = int(clamp(int(back.w // 3), 10, 18))
                            for px in (back.x + 6, back.right - p_w - 6):
                                p = pygame.Rect(int(px), back.y + 3, int(p_w), max(4, int(back.h - 5)))
                                pygame.draw.rect(surface, self._tint(seat, add=(18, 18, 18)), p, border_radius=3)
                                pygame.draw.rect(surface, outline, p, 1, border_radius=3)
                        floor_y = int(br.bottom - 3)
                        leg_y0 = int(seat_r.bottom - 1)
                        leg_h = int(floor_y - leg_y0)
                        if leg_h > 0:
                            leg_col = self._tint(seat2, add=(-22, -22, -22))
                            for lx in (int(seat_r.x + 6), int(seat_r.right - 8)):
                                leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                                surface.fill(leg_col, leg)
                                pygame.draw.rect(surface, outline, leg, 1)
                    else:
                        pygame.draw.rect(surface, seat, obj, border_radius=5)
                        pygame.draw.rect(surface, outline, obj, 1, border_radius=5)
                        back = pygame.Rect(obj.x + 2, obj.y + 2, obj.w - 4, int(clamp(int(obj.h * 0.45), 5, 10)))
                        pygame.draw.rect(surface, seat2, back, border_radius=4)
                        pygame.draw.rect(surface, outline, back, 1, border_radius=4)
                        seat_r = pygame.Rect(obj.x + 2, back.bottom - 1, obj.w - 4, obj.bottom - (back.bottom - 1))
                        pygame.draw.rect(surface, self._tint(seat, add=(12, 12, 12)), seat_r, border_radius=4)
                        pygame.draw.rect(surface, outline, seat_r, 1, border_radius=4)
                        floor_y = int(br.bottom - 3)
                        leg_y0 = int(seat_r.bottom - 1)
                        leg_h = int(floor_y - leg_y0)
                        if leg_h > 0:
                            leg_col = self._tint(seat2, add=(-18, -18, -18))
                            for lx in (int(seat_r.x + 2), int(seat_r.right - 4)):
                                leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                                surface.fill(leg_col, leg)
                                pygame.draw.rect(surface, outline, leg, 1)
                    continue

                if False and ch == "C":
                    sr = pygame.Rect(r.x + 4, r.y + 4, r.w - 8, r.h - 8)
                    sh = pygame.Surface((sr.w, sr.h), pygame.SRCALPHA)  
                    pygame.draw.rect(sh, (0, 0, 0, 60), sh.get_rect(), border_radius=4)
                    surface.blit(sh, (sr.x + 1, sr.y + 2))
                    pygame.draw.rect(surface, seat, sr, border_radius=4)
                    pygame.draw.rect(surface, outline, sr, 1, border_radius=4)
                    pygame.draw.rect(surface, seat2, pygame.Rect(sr.x + 2, sr.y + 2, sr.w - 4, 4), border_radius=2)
                    # Legs (gives the chair more readable structure).
                    leg_y0 = int(sr.bottom - 1)
                    leg_y1 = int(r.bottom - 3)
                    leg_h = int(leg_y1 - leg_y0)
                    if leg_h > 0:
                        leg_col = self._tint(seat2, add=(-18, -18, -18))        
                        for lx in (int(sr.x + 3), int(sr.right - 5)):
                            leg = pygame.Rect(int(lx), int(leg_y0), 2, int(leg_h))
                            surface.fill(leg_col, leg)
                            pygame.draw.rect(surface, outline, leg, 1)
                    continue

        if str(getattr(self, "hr_mode", "lobby")) == "home" and bool(getattr(self, "hr_edit_mode", False)):
            blocks = getattr(self, "hr_edit_blocks", None)
            if isinstance(blocks, list) and blocks:
                sel_i = int(getattr(self, "hr_edit_index", 0)) % len(blocks)
                _ch, cells = blocks[sel_i]
                min_x = min(int(p[0]) for p in cells)
                max_x = max(int(p[0]) for p in cells)
                min_y = min(int(p[1]) for p in cells)
                max_y = max(int(p[1]) for p in cells)
                sel = pygame.Rect(
                    int(map_x + min_x * tile),
                    int(map_y + min_y * tile),
                    int((max_x - min_x + 1) * tile),
                    int((max_y - min_y + 1) * tile),
                )
                ov = pygame.Surface((sel.w, sel.h), pygame.SRCALPHA)
                ov.fill((255, 220, 140, 50))
                pygame.draw.rect(ov, (255, 220, 140, 120), ov.get_rect(), 2, border_radius=8)
                surface.blit(ov, sel.topleft)

        # Draw player on top of the interior.
        face = pygame.Vector2(self.hr_int_facing)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)
        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"
        pose = str(getattr(self, "player_pose", "")).strip()
        pose_space = str(getattr(self, "player_pose_space", "")).strip()
        pose_anchor = getattr(self, "player_pose_anchor", None)
        use_pose = bool(pose and pose_space == "hr" and pose_anchor is not None)
        if use_pose:
            p = pygame.Vector2(float(pose_anchor[0]), float(pose_anchor[1]))
            if pose == "sleep":
                frame = int(float(getattr(self, "player_pose_phase", 0.0)) * 2.0) % 2
                base = self._get_pose_sprite("sleep", frame=frame)
            else:
                base = self._get_pose_sprite("sit", direction=d, frame=0)
        else:
            p = pygame.Vector2(self.hr_int_pos)
            speed2 = float(self.hr_int_vel.length_squared())
            pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
            pf_run = getattr(self, "player_frames_run", None)
            is_run = bool(getattr(self, "player_sprinting", False))
            pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
            frames = pf.get(d, pf["down"])
            if speed2 <= 0.2 or len(frames) <= 1:
                base = frames[0]
            else:
                walk = frames[1:]
                phase = (float(self.hr_int_walk_phase) % math.tau) / math.tau
                idx = int(phase * len(walk)) % len(walk)
                base = walk[idx]

        scale = int(max(1, int(self._HR_INT_SPRITE_SCALE)))
        spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
        sx = int(round(map_x + p.x))
        sy = int(round(map_y + p.y))
        if use_pose and pose == "sleep":
            shadow = pygame.Rect(0, 0, 20, 8)
            shadow.center = (sx, sy + 4)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 90), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect(center=(sx, sy + 2))
            surface.blit(spr, rect)
        elif use_pose and pose == "sit":
            shadow = pygame.Rect(0, 0, 16, 7)
            shadow.center = (sx, sy + 6)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 110), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect(center=(sx, sy + 4))
            surface.blit(spr, rect)
        else:
            shadow = pygame.Rect(0, 0, 14, 6)
            shadow.center = (sx, sy + 8)
            sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 120), sh.get_rect())
            surface.blit(sh, shadow.topleft)
            rect = spr.get_rect()
            rect.midbottom = (sx, sy + 12)
            surface.blit(spr, rect)

        # Hover: furniture details + outline (home only).
        if str(getattr(self, "hr_mode", "lobby")) == "home":
            mouse = None
            if (
                not self.inv_open
                and not bool(getattr(self, "hr_elevator_ui_open", False))
                and not bool(getattr(self, "home_ui_open", False))
                and not bool(getattr(self, "world_map_open", False))
            ):
                mouse = self.app.screen_to_internal(pygame.mouse.get_pos())
            if mouse is not None:
                mx, my = int(mouse[0]), int(mouse[1])
                if map_rect.collidepoint(mx, my):
                    ix = int((mx - int(map_x)) // int(tile))
                    iy = int((my - int(map_y)) // int(tile))
                    ch = str(self._hr_int_char_at(ix, iy))[:1]
                    movable = {"B", "S", "F", "K", "T", "C"}
                    if ch in movable:
                        w = int(self._HR_INT_W)
                        h = int(self._HR_INT_H)
                        seen: set[tuple[int, int]] = set()
                        stack = [(int(ix), int(iy))]
                        cells: list[tuple[int, int]] = []
                        while stack:
                            cx, cy = stack.pop()
                            cx = int(cx)
                            cy = int(cy)
                            if (cx, cy) in seen:
                                continue
                            if not (0 <= cx < w and 0 <= cy < h):
                                continue
                            if str(self._hr_int_char_at(cx, cy))[:1] != ch:
                                continue
                            seen.add((cx, cy))
                            cells.append((cx, cy))
                            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                        if cells:
                            min_x = min(int(p[0]) for p in cells)
                            max_x = max(int(p[0]) for p in cells)
                            min_y = min(int(p[1]) for p in cells)
                            max_y = max(int(p[1]) for p in cells)
                            sel = pygame.Rect(
                                int(map_x + min_x * tile),
                                int(map_y + min_y * tile),
                                int((max_x - min_x + 1) * tile),
                                int((max_y - min_y + 1) * tile),
                            )
                            ov = pygame.Surface((sel.w, sel.h), pygame.SRCALPHA)
                            ov.fill((255, 220, 140, 20))
                            pygame.draw.rect(ov, (255, 220, 140, 150), ov.get_rect(), 2, border_radius=8)
                            surface.blit(ov, sel.topleft)

                            bw = int(max_x - min_x + 1)
                            bh = int(max_y - min_y + 1)
                            name = "家具"
                            desc = ""
                            if ch == "B":
                                name, desc = ("床", "躺下可恢复体力与心情")
                            elif ch == "S":
                                name, desc = ("储物柜", "打开后可存取物资")
                            elif ch == "F":
                                name, desc = ("冰箱", "存放食物/饮料（可存取）")
                            elif ch == "T":
                                name, desc = ("桌子", "临时放置/整理物品（规划）")
                            elif ch == "K":
                                name, desc = ("厨房", "台面上有锅碗瓢盆（烹饪/净水规划）")
                            elif ch == "C":
                                if bw >= 3 or bh >= 2:
                                    name, desc = ("沙发", "坐下休息（规划）")
                                else:
                                    name, desc = ("椅子", "坐下休息（规划）")
                            if bool(getattr(self, "hr_edit_mode", False)):
                                hint = "左键拖拽 | 方向键微调 | R 退出摆放"
                            else:
                                hint = "R 进入摆放模式 | 左键拖拽"
                            self._hover_tooltip = ([str(name), str(desc), str(hint)], (mx, my))
        return

        floor_a = (64, 62, 58)
        floor_b = (58, 56, 52)
        wall = (22, 22, 26)
        wall_hi = (38, 38, 46)
        door_col = (240, 220, 140)
        apt_door = (120, 92, 64)
        apt_door2 = (90, 68, 46)
        elev = (110, 112, 120)
        elev2 = (74, 76, 86)
        glass = (90, 140, 190)
        seat = (56, 72, 140)
        seat2 = (40, 52, 112)
        shelf = (92, 78, 62)
        shelf2 = (66, 56, 46)
        table = (88, 72, 54)
        table2 = (60, 50, 38)
        kitchen = (86, 78, 68)
        kitchen2 = (60, 54, 46)
        steel = (220, 220, 230)
        bed = (170, 150, 140)
        bed2 = (128, 108, 100)

        title = "高层住宅"
        mode = str(getattr(self, "hr_mode", "lobby"))
        floor = int(getattr(self, "hr_floor", 0))
        if mode == "home":
            subtitle = "你的家"
        elif floor <= 0:
            subtitle = "大堂"
        else:
            subtitle = f"{floor} 层"
            if int(getattr(self, "hr_home_floor", -1)) == floor:
                subtitle += "（家）"
        draw_text(surface, self.app.font_s, f"{title} - {subtitle}", (INTERNAL_W // 2, panel.top - 14), pygame.Color(240, 240, 240), anchor="center")

        bed_done: set[tuple[int, int]] = set()
        elev_done: set[tuple[int, int]] = set()
        for y in range(int(self._HR_INT_H)):
            row = self.hr_layout[y] if 0 <= y < len(self.hr_layout) else ""
            for x in range(int(self._HR_INT_W)):
                ch = row[x] if x < len(row) else "W"
                r = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)

                if ch in ("W", "V", "D", "A", "H"):
                    pygame.draw.rect(surface, wall, r)
                    pygame.draw.rect(surface, wall_hi, pygame.Rect(r.x, r.y, r.w, 3))
                    pygame.draw.rect(surface, (0, 0, 0), r, 1)
                    if ch == "V":
                        g = pygame.Rect(r.x + 3, r.y + 4, r.w - 6, r.h - 8)
                        pygame.draw.rect(surface, glass, g, border_radius=2)
                        pygame.draw.rect(surface, (10, 10, 12), g, 1, border_radius=2)
                        pygame.draw.line(surface, (180, 210, 240), (g.left + 1, g.top + 1), (g.right - 2, g.top + 1), 1)
                        pygame.draw.line(surface, (40, 40, 50), (g.centerx, g.top + 1), (g.centerx, g.bottom - 2), 1)
                    elif ch == "D":
                        dr = pygame.Rect(r.x + 5, r.y + 4, r.w - 10, r.h - 8)
                        pygame.draw.rect(surface, door_col, dr, border_radius=2)
                        pygame.draw.rect(surface, (10, 10, 12), dr, 1, border_radius=2)
                        pygame.draw.circle(surface, (10, 10, 12), (dr.right - 5, dr.centery), 2)
                    elif ch in ("A", "H"):
                        dr = pygame.Rect(r.x + 4, r.y + 5, r.w - 8, r.h - 10)
                        pygame.draw.rect(surface, apt_door, dr, border_radius=2)
                        pygame.draw.rect(surface, (10, 10, 12), dr, 1, border_radius=2)
                        pygame.draw.rect(surface, apt_door2, pygame.Rect(dr.x + 2, dr.y + 2, dr.w - 4, 3))
                        pygame.draw.circle(surface, (240, 220, 140), (dr.right - 4, dr.centery), 2)
                        if ch == "H":
                            pygame.draw.rect(surface, (120, 200, 140), pygame.Rect(dr.x + 2, dr.y - 2, dr.w - 4, 3), border_radius=1)
                    continue

                col = floor_a if ((x + y) % 2 == 0) else floor_b
                pygame.draw.rect(surface, col, r)
                pygame.draw.rect(surface, (0, 0, 0), r, 1)

                if ch == "E":
                    if (x, y) in elev_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "E":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "E":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "E":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            elev_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    er = pygame.Rect(br.x + 2, br.y + 2, br.w - 4, br.h - 4)
                    pygame.draw.rect(surface, elev, er, border_radius=3)
                    pygame.draw.rect(surface, (10, 10, 12), er, 1, border_radius=3)
                    pygame.draw.rect(surface, elev2, pygame.Rect(er.x + 3, er.y + 3, er.w - 6, 3))
                    pygame.draw.line(surface, (10, 10, 12), (er.centerx, er.y + 4), (er.centerx, er.bottom - 5), 1)
                    continue
                if ch == "C":
                    pygame.draw.rect(surface, seat, pygame.Rect(r.x + 4, r.y + 4, r.w - 8, r.h - 8), border_radius=3)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 4, r.y + 4, r.w - 8, r.h - 8), 1, border_radius=3)
                    pygame.draw.rect(surface, seat2, pygame.Rect(r.x + 6, r.y + 6, r.w - 12, 4))
                elif ch == "T":
                    pygame.draw.rect(surface, table, pygame.Rect(r.x + 3, r.y + 6, r.w - 6, r.h - 10), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 3, r.y + 6, r.w - 6, r.h - 10), 1, border_radius=2)
                    pygame.draw.rect(surface, table2, pygame.Rect(r.x + 4, r.y + 8, r.w - 8, 3))
                elif ch == "K":
                    pygame.draw.rect(surface, kitchen, pygame.Rect(r.x + 2, r.y + 5, r.w - 4, r.h - 8), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 2, r.y + 5, r.w - 4, r.h - 8), 1, border_radius=2)
                    pygame.draw.rect(surface, kitchen2, pygame.Rect(r.x + 3, r.y + 7, r.w - 6, 3))
                    pygame.draw.rect(surface, steel, pygame.Rect(r.x + 6, r.y + 6, 6, 4), border_radius=1)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 6, r.y + 6, 6, 4), 1, border_radius=1)
                elif ch == "B":
                    if (int(x), int(y)) in bed_done:
                        continue
                    bw = 1
                    while self._hr_int_char_at(int(x) + int(bw), int(y)) == "B":
                        bw += 1
                    bh = 1
                    while True:
                        ny = int(y) + int(bh)
                        if self._hr_int_char_at(int(x), int(ny)) != "B":
                            break
                        ok = True
                        for dx in range(int(bw)):
                            if self._hr_int_char_at(int(x) + int(dx), int(ny)) != "B":
                                ok = False
                                break
                        if not ok:
                            break
                        bh += 1
                    for dy in range(int(bh)):
                        for dx in range(int(bw)):
                            bed_done.add((int(x) + int(dx), int(y) + int(dy)))
                    br = pygame.Rect(map_x + int(x) * tile, map_y + int(y) * tile, int(bw) * tile, int(bh) * tile)
                    bed_r = pygame.Rect(br.x + 2, br.y + 3, br.w - 4, br.h - 6)
                    pygame.draw.rect(surface, bed, bed_r, border_radius=4)
                    pygame.draw.rect(surface, (10, 10, 12), bed_r, 1, border_radius=4)
                    pillow_w = int(clamp(int(bed_r.w // 3), 8, max(8, bed_r.w - 4)))
                    pillow = pygame.Rect(bed_r.x + 2, bed_r.y + 2, pillow_w, 6)
                    pygame.draw.rect(surface, (230, 230, 235), pillow, border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pillow, 1, border_radius=2)
                    pygame.draw.rect(surface, bed2, pygame.Rect(bed_r.x + 2, bed_r.y + 10, bed_r.w - 4, 6), border_radius=2)
                    continue
                elif ch == "S":
                    pygame.draw.rect(surface, shelf, pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6), border_radius=2)
                    pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6), 1, border_radius=2)
                    pygame.draw.line(surface, shelf2, (r.x + 4, r.y + r.h // 2), (r.right - 5, r.y + r.h // 2), 2)

        p = pygame.Vector2(self.hr_int_pos)
        speed2 = float(self.hr_int_vel.length_squared())
        face = pygame.Vector2(self.hr_int_facing)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)
        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"
        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = pf.get(d, pf["down"])
        if speed2 <= 0.2 or len(frames) <= 1:
            base = frames[0]
        else:
            walk = frames[1:]
            phase = (float(self.hr_int_walk_phase) % math.tau) / math.tau
            idx = int(phase * len(walk)) % len(walk)
            base = walk[idx]

        scale = int(max(1, int(self._HR_INT_SPRITE_SCALE)))
        spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
        sx = int(round(map_x + p.x))
        sy = int(round(map_y + p.y))
        shadow = pygame.Rect(0, 0, 14, 6)
        shadow.center = (sx, sy + 8)
        pygame.draw.ellipse(surface, (0, 0, 0), shadow)
        rect = spr.get_rect()
        rect.midbottom = (sx, sy + 12)
        surface.blit(spr, rect)

    def _draw_sch_elevator_ui(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        surface.blit(overlay, (0, 0))

        options: list[int] = list(getattr(self, "sch_elevator_options", []))
        if not options:
            options = [1]
            self.sch_elevator_options = options
        cols = max(1, int(getattr(self, "sch_elevator_cols", 4)))
        rows_per_page = 4
        n = len(options)
        page_size = max(1, cols * rows_per_page)
        sel_i = int(getattr(self, "sch_elevator_sel", 0)) % max(1, n)
        page = int(sel_i // page_size)
        page_count = int(max(1, int(math.ceil(float(n) / float(page_size)))))
        start = int(page * page_size)
        end = int(min(int(n), int(start + page_size)))
        page_opts = list(options[start:end])
        rows = int(max(1, int(math.ceil(float(len(page_opts)) / float(cols)))))

        btn_w = 56
        btn_h = 28
        gap = 8
        grid_w = cols * btn_w + (cols - 1) * gap
        grid_h = rows * btn_h + (rows - 1) * gap
        panel = pygame.Rect(0, 0, grid_w + 40, grid_h + 98)
        panel.center = (INTERNAL_W // 2, INTERNAL_H // 2)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        draw_text(surface, self.app.font_m, "电梯", (panel.centerx, panel.top + 16), pygame.Color(240, 240, 240), anchor="center")
        draw_text(
            surface,
            self.app.font_s,
            f"选择楼层（{int(page) + 1}/{int(page_count)}页）",
            (panel.centerx, panel.top + 36),
            pygame.Color(180, 180, 195),
            anchor="center",
        )
        typed = str(getattr(self, "sch_elevator_input", "")).strip()
        draw_text(
            surface,
            self.app.font_s,
            f"输入: {typed if typed else '—'}",
            (panel.centerx, panel.top + 52),
            pygame.Color(200, 200, 215),
            anchor="center",
        )

        x0 = panel.centerx - grid_w // 2
        y0 = panel.top + 66
        cur_floor = int(getattr(self, "sch_floor", 1))

        for local_i, f in enumerate(page_opts):
            i = int(start) + int(local_i)
            r = int(local_i) // cols
            c = int(local_i) % cols
            x = x0 + c * (btn_w + gap)
            y = y0 + r * (btn_h + gap)
            br = pygame.Rect(x, y, btn_w, btn_h)
            selected = int(i) == int(sel_i)
            is_cur = int(f) == cur_floor
            bg = (44, 44, 52) if selected else (26, 26, 32)
            border = (240, 240, 255) if selected else (90, 90, 110)
            if is_cur and not selected:
                border = (255, 220, 140)
            pygame.draw.rect(surface, bg, br, border_radius=8)
            pygame.draw.rect(surface, border, br, 2, border_radius=8)
            label = f"{int(f)}F"
            draw_text(surface, self.app.font_s, label, (br.centerx, br.centery), pygame.Color(230, 230, 240), anchor="center")

        draw_text(
            surface,
            self.app.font_s,
            "方向键移动 | PgUp/PgDn翻页 | 数字直达 | Enter确认 | Esc取消",
            (panel.centerx, panel.bottom - 14),
            pygame.Color(160, 160, 175),
            anchor="center",
        )

    def _draw_hr_elevator_ui(self, surface: pygame.Surface) -> None:     
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        surface.blit(overlay, (0, 0))

        options: list[int] = list(getattr(self, "hr_elevator_options", []))
        if not options:
            options = [0]
        cols = max(1, int(getattr(self, "hr_elevator_cols", 4)))
        rows_per_page = 4
        n = len(options)
        page_size = max(1, cols * rows_per_page)
        sel_i = int(getattr(self, "hr_elevator_sel", 0)) % max(1, n)
        page = int(sel_i // page_size)
        page_count = int(max(1, int(math.ceil(float(n) / float(page_size)))))
        start = int(page * page_size)
        end = int(min(int(n), int(start + page_size)))
        page_opts = list(options[start:end])
        rows = int(max(1, int(math.ceil(float(len(page_opts)) / float(cols)))))

        btn_w = 56
        btn_h = 28
        gap = 8
        grid_w = cols * btn_w + (cols - 1) * gap
        grid_h = rows * btn_h + (rows - 1) * gap
        panel = pygame.Rect(0, 0, grid_w + 40, grid_h + 98)
        panel.center = (INTERNAL_W // 2, INTERNAL_H // 2)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        draw_text(surface, self.app.font_m, "电梯", (panel.centerx, panel.top + 16), pygame.Color(240, 240, 240), anchor="center")
        draw_text(
            surface,
            self.app.font_s,
            f"选择楼层（{int(page) + 1}/{int(page_count)}页）",
            (panel.centerx, panel.top + 36),
            pygame.Color(180, 180, 195),
            anchor="center",
        )
        typed = str(getattr(self, "hr_elevator_input", "")).strip()
        draw_text(
            surface,
            self.app.font_s,
            f"输入: {typed if typed else '—'}",
            (panel.centerx, panel.top + 52),
            pygame.Color(200, 200, 215),
            anchor="center",
        )

        x0 = panel.centerx - grid_w // 2
        y0 = panel.top + 66
        cur_floor = int(getattr(self, "hr_floor", 0))
        home_floor = int(getattr(self, "hr_home_floor", -1))

        for local_i, f in enumerate(page_opts):
            i = int(start) + int(local_i)
            r = int(local_i) // cols
            c = int(local_i) % cols
            x = x0 + c * (btn_w + gap)
            y = y0 + r * (btn_h + gap)
            br = pygame.Rect(x, y, btn_w, btn_h)
            selected = int(i) == int(sel_i)
            is_cur = int(f) == cur_floor
            bg = (44, 44, 52) if selected else (26, 26, 32)
            border = (240, 240, 255) if selected else (90, 90, 110)     
            if is_cur and not selected:
                border = (255, 220, 140)
            if int(f) == int(home_floor) and int(home_floor) > 0 and not selected and not is_cur:
                border = (120, 200, 140)
            pygame.draw.rect(surface, bg, br, border_radius=8)
            pygame.draw.rect(surface, border, br, 2, border_radius=8)   
            label = "大堂" if int(f) == 0 else str(int(f))
            draw_text(surface, self.app.font_s, label, (br.centerx, br.centery), pygame.Color(230, 230, 240), anchor="center")
            if int(f) == int(home_floor) and int(home_floor) > 0:
                pygame.draw.circle(surface, (120, 200, 140), (br.right - 10, br.y + 8), 3)

        draw_text(
            surface,
            self.app.font_s,
            "方向键移动 | PgUp/PgDn翻页 | 数字直达 | Enter确认 | Esc取消",
            (panel.centerx, panel.bottom - 14),
            pygame.Color(160, 160, 175),
            anchor="center",
        )

    def _rv_ui_open(self) -> None:
        if not self._can_access_rv():
            self._set_hint("靠近房车按 H 打开房车内部", seconds=1.2)
            return
        self.inv_open = False
        self.rv_ui_open = True
        self.rv_ui_focus = "map"
        door = self._rv_int_find("D")
        if door is None:
            self.rv_ui_map = (1, 1)
        else:
            self.rv_ui_map = door
        self._set_rv_ui_status("房车内部：方向键移动 | Enter互动 | Tab切换区域 | Esc关闭", seconds=2.0)

    def _rv_ui_close(self) -> None:
        self.rv_ui_open = False
        self.rv_ui_focus = "map"

    def _rv_ui_toggle(self) -> None:
        if getattr(self, "rv_ui_open", False):
            self._rv_ui_close()
        else:
            self._rv_ui_open()

    def _rv_ui_char_at(self, x: int, y: int) -> str:
        x = int(x)
        y = int(y)
        if not (0 <= x < int(self._RV_INT_W) and 0 <= y < int(self._RV_INT_H)):
            return "W"
        layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)
        row = layout[y]
        if 0 <= x < len(row):
            return row[x]
        return "W"

    def _rv_ui_tile_at(self, x: int, y: int) -> int:
        ch = self._rv_ui_char_at(x, y)
        return int(self._RV_INT_LEGEND.get(ch, int(self.T_FLOOR)))

    def _rv_ui_cycle_focus(self) -> None:
        cur = str(getattr(self, "rv_ui_focus", "map"))
        if cur == "map":
            self.rv_ui_focus = "player"
        elif cur == "player":
            self.rv_ui_focus = "storage"
        else:
            self.rv_ui_focus = "map"

    def _rv_ui_move_map(self, dx: int, dy: int) -> None:
        x, y = self.rv_ui_map
        nx = int(x) + int(dx)
        ny = int(y) + int(dy)
        if not (0 <= nx < int(self._RV_INT_W) and 0 <= ny < int(self._RV_INT_H)):
            return
        if self._rv_ui_char_at(nx, ny) in ("W", "V"):
            return
        self.rv_ui_map = (nx, ny)

    def _rv_ui_transfer(self, *, from_player: bool) -> None:
        if from_player:
            src = self.inventory
            dst = self.rv_storage
            idx = int(self.rv_ui_player_index)
        else:
            src = self.rv_storage
            dst = self.inventory
            idx = int(self.rv_ui_storage_index)

        if not (0 <= idx < len(src.slots)):
            self._set_rv_ui_status("槽位无效", seconds=1.2)
            return
        st = src.slots[idx]
        if st is None:
            self._set_rv_ui_status("空", seconds=1.0)
            return
        before = int(st.qty)
        left = int(dst.add(st.item_id, int(st.qty), self._ITEMS))
        moved = int(before - left)
        if moved <= 0:
            self._set_rv_ui_status("空间不足", seconds=1.2)
            return
        if left <= 0:
            src.slots[idx] = None
        else:
            st.qty = int(left)
        idef = self._ITEMS.get(st.item_id)
        name = idef.name if idef is not None else st.item_id
        self._set_rv_ui_status(f"转移：{name} x{moved}", seconds=1.6)

    def _rv_ui_interact(self) -> None:
        x, y = self.rv_ui_map
        ch = self._rv_ui_char_at(int(x), int(y))
        if ch == "D":
            self._rv_ui_close()
            return
        if ch == "B":
            self.player.stamina = 100.0
            self.player.morale = float(clamp(self.player.morale + 18.0, 0.0, 100.0))
            self.player.condition = float(clamp(self.player.condition + 4.0, 0.0, 100.0))
            self._set_rv_ui_status("休息：体力恢复 / 心态提升", seconds=1.6)
            return
        if ch == "S":
            self.rv_ui_focus = "storage"
            self._set_rv_ui_status("储物柜：在背包/储物间按 Enter 转移", seconds=1.8)
            return
        if ch in ("T", "C"):
            self._set_rv_ui_status("工作台/驾驶舱（占位，后续加制作/维修）", seconds=1.8)
            return
        self._set_rv_ui_status("这里没有可用交互", seconds=1.2)

    def _handle_rv_ui_key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE,):
            self._rv_ui_close()
            return
        if key in (pygame.K_TAB,):
            self._rv_ui_cycle_focus()
            return
        if key in (pygame.K_RETURN, pygame.K_SPACE):
            focus = str(getattr(self, "rv_ui_focus", "map"))
            if focus == "map":
                self._rv_ui_interact()
            elif focus == "player":
                self._rv_ui_transfer(from_player=True)
            else:
                self._rv_ui_transfer(from_player=False)
            return

        move: tuple[int, int] | None = None
        if key in (pygame.K_LEFT, pygame.K_a):
            move = (-1, 0)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            move = (1, 0)
        elif key in (pygame.K_UP, pygame.K_w):
            move = (0, -1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            move = (0, 1)
        if move is None:
            return

        focus = str(getattr(self, "rv_ui_focus", "map"))
        dx, dy = int(move[0]), int(move[1])
        if focus == "map":
            self._rv_ui_move_map(dx, dy)
            return

        if focus == "player":
            cols = max(1, int(self.inventory.cols))
            if dx != 0:
                self.rv_ui_player_index = (int(self.rv_ui_player_index) + dx) % len(self.inventory.slots)
            else:
                self.rv_ui_player_index = (int(self.rv_ui_player_index) + dy * cols) % len(self.inventory.slots)
            return

        cols = max(1, int(self.rv_storage.cols))
        if dx != 0:
            self.rv_ui_storage_index = (int(self.rv_ui_storage_index) + dx) % len(self.rv_storage.slots)
        else:
            self.rv_ui_storage_index = (int(self.rv_ui_storage_index) + dy * cols) % len(self.rv_storage.slots)

    def _home_ui_storage_kind_norm(self) -> str:
        kind = str(getattr(self, "home_ui_storage_kind", "cabinet")).strip().lower()
        return "fridge" if kind == "fridge" else "cabinet"

    def _home_ui_storage_inv(self) -> HardcoreSurvivalState._Inventory:
        if self._home_ui_storage_kind_norm() == "fridge":
            inv = getattr(self, "fridge_storage", None)
            if inv is not None:
                return inv
        return self.home_storage

    def _home_ui_storage_label(self) -> str:
        return "冰箱" if self._home_ui_storage_kind_norm() == "fridge" else "柜子"

    def _home_ui_storage_title(self) -> str:
        return "冰箱" if self._home_ui_storage_kind_norm() == "fridge" else "家中储物柜"

    def _home_ui_open(
        self,
        *,
        open_block: tuple[int, int, int, int] | None = None,
        storage_kind: str = "cabinet",
    ) -> None:
        if not getattr(self, "hr_interior", False):
            return
        if str(getattr(self, "hr_mode", "lobby")) != "home":
            return
        self.inv_open = False
        self.rv_ui_open = False
        self._gallery_open = False
        self.home_ui_open = True
        self.home_ui_focus = "storage"
        self.home_ui_storage_kind = "fridge" if str(storage_kind).strip().lower() == "fridge" else "cabinet"
        self.home_ui_open_block = open_block
        self.home_ui_player_index = int(
            clamp(int(getattr(self, "home_ui_player_index", 0)), 0, len(self.inventory.slots) - 1)
        )
        storage = self._home_ui_storage_inv()
        self.home_ui_storage_index = int(
            clamp(int(getattr(self, "home_ui_storage_index", 0)), 0, len(storage.slots) - 1)
        )
        title = "冰箱" if self._home_ui_storage_kind_norm() == "fridge" else "储物柜"
        target = self._home_ui_storage_label()
        self._set_home_ui_status(
            f"{title}：Tab 切换 背包/{target} | Enter 转移 | Esc 关闭",
            seconds=2.0,
        )

    def _home_ui_close(self) -> None:
        self.home_ui_open = False
        self.home_ui_focus = "storage"
        self.home_ui_open_block = None

    def _home_ui_cycle_focus(self) -> None:
        cur = str(getattr(self, "home_ui_focus", "storage"))
        self.home_ui_focus = "player" if cur != "player" else "storage"

    def _home_ui_transfer(self, *, from_player: bool) -> None:
        storage = self._home_ui_storage_inv()
        if from_player:
            src = self.inventory
            dst = storage
            idx = int(getattr(self, "home_ui_player_index", 0))
        else:
            src = storage
            dst = self.inventory
            idx = int(getattr(self, "home_ui_storage_index", 0))

        if not (0 <= idx < len(src.slots)):
            self._set_home_ui_status("槽位无效", seconds=1.2)
            return
        st = src.slots[idx]
        if st is None:
            self._set_home_ui_status("空", seconds=1.0)
            return
        before = int(st.qty)
        left = int(dst.add(st.item_id, int(st.qty), self._ITEMS))
        moved = int(before - left)
        if moved <= 0:
            self._set_home_ui_status("空间不足", seconds=1.2)
            return
        if left <= 0:
            src.slots[idx] = None
        else:
            st.qty = int(left)
        idef = self._ITEMS.get(st.item_id)
        name = idef.name if idef is not None else st.item_id
        self._set_home_ui_status(f"转移: {name} x{moved}", seconds=1.6)

    def _handle_home_ui_key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE,):
            self._home_ui_close()
            return
        if key in (pygame.K_TAB,):
            self._home_ui_cycle_focus()
            return
        if key in (pygame.K_RETURN, pygame.K_SPACE):
            focus = str(getattr(self, "home_ui_focus", "storage"))
            if focus == "player":
                self._home_ui_transfer(from_player=True)
            else:
                self._home_ui_transfer(from_player=False)
            return

        move: tuple[int, int] | None = None
        if key in (pygame.K_LEFT, pygame.K_a):
            move = (-1, 0)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            move = (1, 0)
        elif key in (pygame.K_UP, pygame.K_w):
            move = (0, -1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            move = (0, 1)
        if move is None:
            return

        focus = str(getattr(self, "home_ui_focus", "storage"))
        dx, dy = int(move[0]), int(move[1])
        if focus == "player":
            cols = max(1, int(self.inventory.cols))
            if dx != 0:
                self.home_ui_player_index = (int(self.home_ui_player_index) + dx) % len(self.inventory.slots)
            else:
                self.home_ui_player_index = (int(self.home_ui_player_index) + dy * cols) % len(self.inventory.slots)
            return

        storage = self._home_ui_storage_inv()
        cols = max(1, int(storage.cols))
        if dx != 0:
            self.home_ui_storage_index = (int(self.home_ui_storage_index) + dx) % len(storage.slots)
        else:
            self.home_ui_storage_index = (int(self.home_ui_storage_index) + dy * cols) % len(storage.slots)

    def _handle_home_ui_mouse(self, event: pygame.event.Event) -> None:
        if not getattr(self, "home_ui_open", False):
            return
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return
        if not hasattr(event, "pos"):
            return
        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return
        mx, my = int(internal[0]), int(internal[1])

        # Mirror the layout math in _draw_home_storage_ui() so hit-testing matches.
        panel = pygame.Rect(16, 14, INTERNAL_W - 32, INTERNAL_H - 28)
        slot = 22
        gap = 4

        storage = self._home_ui_storage_inv()
        storage_label = self._home_ui_storage_label()

        pcols = max(1, int(self.inventory.cols))
        prows = int(math.ceil(len(self.inventory.slots) / pcols))
        pgrid_w = pcols * slot + (pcols - 1) * gap
        pgrid_h = prows * slot + (prows - 1) * gap

        scols = max(1, int(storage.cols))
        srows = int(math.ceil(len(storage.slots) / scols))
        sgrid_w = scols * slot + (scols - 1) * gap
        sgrid_h = srows * slot + (srows - 1) * gap

        inv_y = panel.top + 46
        left_x = panel.left + 14
        right_x = panel.right - 14 - sgrid_w

        content_h = max(pgrid_h, sgrid_h) + 20
        inv_y = int(clamp(inv_y, panel.top + 42, panel.bottom - 52 - content_h))

        pgrid = pygame.Rect(int(left_x), int(inv_y), int(pgrid_w), int(pgrid_h))
        sgrid = pygame.Rect(int(right_x), int(inv_y), int(sgrid_w), int(sgrid_h))

        # Clickable "tabs" (the labels above each grid).
        lab_y = int(inv_y - 14)
        bw, bh = self.app.font_s.size("背包")
        sw, sh = self.app.font_s.size(storage_label)
        tab_player = pygame.Rect(int(left_x), int(lab_y), int(bw), int(bh))
        tab_storage = pygame.Rect(int(right_x), int(lab_y), int(sw), int(sh))

        def hit_grid(
            *,
            grid: pygame.Rect,
            cols: int,
            count: int,
            x0: int,
            y0: int,
        ) -> int | None:
            if not grid.collidepoint(mx, my):
                return None
            lx = int(mx - int(x0))
            ly = int(my - int(y0))
            cell = int(slot + gap)
            if cell <= 0:
                return None
            cx = int(lx // cell)
            cy = int(ly // cell)
            if cx < 0 or cy < 0:
                return None
            if cx >= int(cols):
                return None
            # Ignore clicks on the gap area.
            if int(lx % cell) >= int(slot) or int(ly % cell) >= int(slot):
                return None
            idx = int(cy * int(cols) + cx)
            if not (0 <= idx < int(count)):
                return None
            return int(idx)

        if event.type == pygame.MOUSEBUTTONDOWN:
            btn = int(getattr(event, "button", 0))
            if btn == 1:
                if tab_player.collidepoint(mx, my):
                    self.home_ui_focus = "player"
                    return
                if tab_storage.collidepoint(mx, my):
                    self.home_ui_focus = "storage"
                    return
                idx = hit_grid(grid=pgrid, cols=pcols, count=len(self.inventory.slots), x0=left_x, y0=inv_y)
                if idx is not None:
                    self.home_ui_focus = "player"
                    self.home_ui_player_index = int(idx)
                    return
                idx = hit_grid(grid=sgrid, cols=scols, count=len(storage.slots), x0=right_x, y0=inv_y)
                if idx is not None:
                    self.home_ui_focus = "storage"
                    self.home_ui_storage_index = int(idx)
                    return
            # Right-click transfers for mouse-only usability.
            if btn == 3:
                idx = hit_grid(grid=pgrid, cols=pcols, count=len(self.inventory.slots), x0=left_x, y0=inv_y)
                if idx is not None:
                    self.home_ui_focus = "player"
                    self.home_ui_player_index = int(idx)
                    self._home_ui_transfer(from_player=True)
                    return
                idx = hit_grid(grid=sgrid, cols=scols, count=len(storage.slots), x0=right_x, y0=inv_y)
                if idx is not None:
                    self.home_ui_focus = "storage"
                    self.home_ui_storage_index = int(idx)
                    self._home_ui_transfer(from_player=False)
                    return

    def _tile_solid(self, tile_id: int) -> bool:
        t = self._TILES.get(int(tile_id))
        return bool(t.solid) if t is not None else False

    def _tile_slow(self, tile_id: int) -> float:
        t = self._TILES.get(int(tile_id))
        return float(t.slow) if t is not None else 1.0

    def _collide_rect_world(self, rect: pygame.Rect) -> list[pygame.Rect]:
        left = int(math.floor(rect.left / self.TILE_SIZE))
        right = int(math.floor((rect.right - 1) / self.TILE_SIZE))       
        top = int(math.floor(rect.top / self.TILE_SIZE))
        bottom = int(math.floor((rect.bottom - 1) / self.TILE_SIZE))

        hits: list[pygame.Rect] = []
        for ty in range(top, bottom + 1):
            for tx in range(left, right + 1):
                tile = int(self.world.get_tile(tx, ty))
                solid = bool(self._tile_solid(tile))
                if tile == int(self.T_DOOR):
                    # Prevent stepping "onto" portal/sealed doors (e.g. high-rises),
                    # which makes it look like the player can walk on the facade.
                    leads_to_floor = False
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if int(self.world.get_tile(int(tx + dx), int(ty + dy))) == int(self.T_FLOOR):
                            leads_to_floor = True
                            break
                    if not leads_to_floor:
                        solid = True
                if not solid:
                    continue
                tile_rect = pygame.Rect(tx * self.TILE_SIZE, ty * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if rect.colliderect(tile_rect):
                    hits.append(tile_rect)

        # Solid world props (signs/toys/etc).
        start_cx = left // self.CHUNK_SIZE
        end_cx = right // self.CHUNK_SIZE
        start_cy = top // self.CHUNK_SIZE
        end_cy = bottom // self.CHUNK_SIZE
        for cy in range(int(start_cy), int(end_cy) + 1):
            for cx in range(int(start_cx), int(end_cx) + 1):
                chunk = self.world.get_chunk(int(cx), int(cy))
                for pr in getattr(chunk, "props", []):
                    pdef = self._PROP_DEFS.get(str(getattr(pr, "prop_id", "")))
                    if pdef is None or not bool(getattr(pdef, "solid", False)):
                        continue
                    w, h = getattr(pdef, "collider", (10, 10))
                    w = max(2, int(w))
                    h = max(2, int(h))
                    prect = pygame.Rect(
                        int(round(float(pr.pos.x) - w / 2)),
                        int(round(float(pr.pos.y) - h / 2)),
                        int(w),
                        int(h),
                    )
                    if rect.colliderect(prect):
                        hits.append(prect)
        return hits

    def _collide_rect_world_vehicle(self, rect: pygame.Rect) -> list[pygame.Rect]:
        left = int(math.floor(rect.left / self.TILE_SIZE))
        right = int(math.floor((rect.right - 1) / self.TILE_SIZE))
        top = int(math.floor(rect.top / self.TILE_SIZE))
        bottom = int(math.floor((rect.bottom - 1) / self.TILE_SIZE))

        hits: list[pygame.Rect] = []
        for ty in range(top, bottom + 1):
            for tx in range(left, right + 1):
                tile = int(self.world.get_tile(tx, ty))
                # Vehicles must not enter buildings: treat interior floor/doors as solid.
                if tile in (int(self.T_FLOOR), int(self.T_DOOR)):
                    solid = True
                else:
                    solid = bool(self._tile_solid(tile))
                if not solid:
                    continue
                tile_rect = pygame.Rect(
                    int(tx * self.TILE_SIZE),
                    int(ty * self.TILE_SIZE),
                    int(self.TILE_SIZE),
                    int(self.TILE_SIZE),
                )
                if rect.colliderect(tile_rect):
                    hits.append(tile_rect)

        # Solid world props (signs/toys/etc) still block vehicles.
        start_cx = left // self.CHUNK_SIZE
        end_cx = right // self.CHUNK_SIZE
        start_cy = top // self.CHUNK_SIZE
        end_cy = bottom // self.CHUNK_SIZE
        for cy in range(int(start_cy), int(end_cy) + 1):
            for cx in range(int(start_cx), int(end_cx) + 1):
                chunk = self.world.get_chunk(int(cx), int(cy))
                for pr in getattr(chunk, "props", []):
                    pdef = self._PROP_DEFS.get(str(getattr(pr, "prop_id", "")))
                    if pdef is None or not bool(getattr(pdef, "solid", False)):
                        continue
                    w, h = getattr(pdef, "collider", (10, 10))
                    w = max(2, int(w))
                    h = max(2, int(h))
                    prect = pygame.Rect(
                        int(round(float(pr.pos.x) - w / 2)),
                        int(round(float(pr.pos.y) - h / 2)),
                        int(w),
                        int(h),
                    )
                    if rect.colliderect(prect):
                        hits.append(prect)
        return hits

    def _move_with_collisions(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float) -> pygame.Vector2:
        rect = self.player.rect_at(pos)
        dx = float(vel.x * dt)
        dy = float(vel.y * dt)

        if dx != 0.0:
            rect.x += int(round(dx))
            for hit in self._collide_rect_world(rect):
                if dx > 0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right

        if dy != 0.0:
            rect.y += int(round(dy))
            for hit in self._collide_rect_world(rect):
                if dy > 0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom

        return pygame.Vector2(rect.centerx, rect.centery)

    def _move_box(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float, *, w: int, h: int) -> pygame.Vector2:
        # Use float movement (sub-pixel accumulation) then resolve collisions
        # with an int rect. This keeps pixel-art rendering crisp while allowing
        # slow entities (e.g., zombies) to actually move.
        w = int(w)
        h = int(h)
        p = pygame.Vector2(pos)
        dx = float(vel.x) * float(dt)
        dy = float(vel.y) * float(dt)

        if dx != 0.0:
            p.x += float(dx)
            rect = pygame.Rect(
                iround(float(p.x) - float(w) / 2.0),
                iround(float(p.y) - float(h) / 2.0),
                int(w),
                int(h),
            )
            for hit in self._collide_rect_world(rect):
                if dx > 0.0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right
                p.x = float(rect.centerx)

        if dy != 0.0:
            p.y += float(dy)
            rect = pygame.Rect(
                iround(float(p.x) - float(w) / 2.0),
                iround(float(p.y) - float(h) / 2.0),
                int(w),
                int(h),
            )
            for hit in self._collide_rect_world(rect):
                if dy > 0.0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom
                p.y = float(rect.centery)

        return p

    def _move_box_vehicle(self, pos: pygame.Vector2, vel: pygame.Vector2, dt: float, *, w: int, h: int) -> pygame.Vector2:
        rect = pygame.Rect(
            iround(float(pos.x) - float(w) / 2.0),
            iround(float(pos.y) - float(h) / 2.0),
            int(w),
            int(h),
        )
        dx = float(vel.x * dt)
        dy = float(vel.y * dt)

        if dx != 0.0:
            rect.x += int(round(dx))
            for hit in self._collide_rect_world_vehicle(rect):
                if dx > 0:
                    rect.right = hit.left
                else:
                    rect.left = hit.right

        if dy != 0.0:
            rect.y += int(round(dy))
            for hit in self._collide_rect_world_vehicle(rect):
                if dy > 0:
                    rect.bottom = hit.top
                else:
                    rect.top = hit.bottom

        return pygame.Vector2(rect.centerx, rect.centery)

    def update(self, dt: float) -> None:
        if self.hint_left > 0.0:
            self.hint_left = max(0.0, float(self.hint_left) - dt)
            if self.hint_left <= 0.0:
                self.hint_text = ""

        if getattr(self, "rv_ui_status_left", 0.0) > 0.0:
            self.rv_ui_status_left = max(0.0, float(self.rv_ui_status_left) - dt)
            if self.rv_ui_status_left <= 0.0:
                self.rv_ui_status = ""

        if getattr(self, "home_ui_status_left", 0.0) > 0.0:
            self.home_ui_status_left = max(0.0, float(self.home_ui_status_left) - dt)
            if self.home_ui_status_left <= 0.0:
                self.home_ui_status = ""

        if getattr(self, "muzzle_flash_left", 0.0) > 0.0:
            self.muzzle_flash_left = max(0.0, float(self.muzzle_flash_left) - dt)

        if self.player.hp <= 0:
            if self.dead_left <= 0.0:
                self.dead_left = 2.0
                self._set_hint("你死了… 2秒后回主菜单", seconds=2.0)
            else:
                self.dead_left = max(0.0, float(self.dead_left) - dt)
                if self.dead_left <= 0.0:
                    self.app.set_state(MainMenuState(self.app))
            return

        if self.inv_open or getattr(self, "rv_ui_open", False) or getattr(self, "home_ui_open", False) or getattr(self, "hr_elevator_ui_open", False) or getattr(self, "sch_elevator_ui_open", False) or getattr(self, "_gallery_open", False) or getattr(self, "world_map_open", False):
            self.player.vel.update(0, 0)
            self.player.walk_phase *= 0.85
            self.rv.vel.update(0, 0)
            self.bike.vel.update(0, 0)
            return

        self.world_time_s += dt
        self._update_weather(dt)

        if getattr(self, "noise_left", 0.0) > 0.0:
            self.noise_left = max(0.0, float(self.noise_left) - dt)
            if self.noise_left <= 0.0:
                self.noise_radius = 0.0

        # Survival stats (prototype tuning).
        self.player.hunger = float(clamp(self.player.hunger - dt * 0.10, 0.0, 100.0))
        self.player.thirst = float(clamp(self.player.thirst - dt * 0.16, 0.0, 100.0))

        focus = pygame.Vector2(self.player.pos)
        nearest = 99999.0
        for z in self.zombies:
            nearest = min(nearest, float((z.pos - focus).length()))

        stress = 0.0
        if self.player.hunger < 25.0:
            stress += (25.0 - self.player.hunger) / 25.0
        if self.player.thirst < 25.0:
            stress += 1.2 * (25.0 - self.player.thirst) / 25.0
        if self.player.hp < 40:
            stress += (40.0 - float(self.player.hp)) / 40.0
        if nearest < 90.0:
            stress += (90.0 - nearest) / 90.0 * 1.4

        morale_decay = 0.010 + 0.035 * stress
        self.player.morale = float(clamp(self.player.morale - dt * morale_decay * 20.0, 0.0, 100.0))

        cond_down = 0.0
        if self.player.hunger <= 0.0:
            cond_down += 0.10
        if self.player.thirst <= 0.0:
            cond_down += 0.14
        if self.player.hp < 60:
            cond_down += 0.06
        cond_up = 0.0
        if self.mount == "rv" and self.rv.vel.length_squared() <= 0.1 and nearest > 140.0:
            cond_up += 0.08
            if self.player.hunger > 50.0 and self.player.thirst > 50.0:
                cond_up += 0.05
        self.player.condition = float(clamp(self.player.condition + dt * (cond_up - cond_down) * 10.0, 0.0, 100.0))

        # Starvation / dehydration damage.
        dmg_rate = 0.0
        if self.player.hunger <= 0.0:
            dmg_rate += 0.55
        if self.player.thirst <= 0.0:
            dmg_rate += 0.85
        if dmg_rate > 0.0:
            self.starve_accum += dmg_rate * dt
            while self.starve_accum >= 1.0:
                self.starve_accum -= 1.0
                self.player.hp = max(0, int(self.player.hp) - 1)

        keys = pygame.key.get_pressed()
        move = pygame.Vector2(0, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move.y += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move.x += 1

        # Sit/Sleep pose locks movement inside interiors.
        pose = getattr(self, "player_pose", None)
        if pose:
            cur_space = "world"
            if getattr(self, "sch_interior", False):
                cur_space = "sch"
            elif getattr(self, "hr_interior", False):
                cur_space = "hr"
            elif getattr(self, "rv_interior", False):
                cur_space = "rv"
            if str(getattr(self, "player_pose_space", "")) != cur_space:
                self._clear_player_pose()
            else:
                self.player_pose_phase = float(getattr(self, "player_pose_phase", 0.0)) + dt
                left = float(getattr(self, "player_pose_left", 0.0))
                if left > 0.0:
                    self.player_pose_left = max(0.0, left - dt)
                    if self.player_pose_left <= 0.0:
                        self._clear_player_pose()
                if getattr(self, "player_pose", None) is not None:
                    if move.length_squared() > 0.001:
                        self._clear_player_pose()
                    else:
                        move = pygame.Vector2(0, 0)

        move_raw = pygame.Vector2(move)
        if move.length_squared() > 0.001:
            move = move.normalize()
            if self.mount != "rv":
                self.player.facing = pygame.Vector2(move.x, move.y)

        if getattr(self, "sch_interior", False):
            base_speed = 60.0
            self.sch_int_vel = move * base_speed
            self.player_sprinting = False
            if self.sch_int_vel.length_squared() > 0.1:
                self.sch_int_facing = pygame.Vector2(self.sch_int_vel).normalize()
                self.sch_int_walk_phase += dt * 10.0 * (self.sch_int_vel.length() / base_speed)
            else:
                self.sch_int_walk_phase *= 0.90
            self.sch_int_pos = self._sch_int_move_box(
                self.sch_int_pos,
                self.sch_int_vel,
                dt,
                w=int(getattr(self, "rv_int_player_w", self._RV_INT_PLAYER_W)),
                h=int(getattr(self, "rv_int_player_h", self._RV_INT_PLAYER_H)),
            )
            self.player.vel.update(0, 0)
            self.rv.vel.update(0, 0)
            self.bike.vel.update(0, 0)
            self._update_gun_timers(dt, allow_fire=False)
            return

        if getattr(self, "hr_interior", False):
            base_speed = 60.0

            # "Walk into the lobby then to the elevator" feel on entry.
            if bool(getattr(self, "hr_auto_walk_to_elevator", False)):
                # Allow player input to cancel the auto-walk.
                if move.length_squared() > 0.001:
                    self.hr_auto_walk_to_elevator = False
                else:
                    delay = float(getattr(self, "hr_auto_walk_delay", 0.0))
                    if delay > 0.0:
                        self.hr_auto_walk_delay = max(0.0, delay - dt)
                        move = pygame.Vector2(0, 0)
                    elif str(getattr(self, "hr_mode", "lobby")) == "lobby":
                        elev = self._hr_int_find("E")
                        if elev is None:
                            self.hr_auto_walk_to_elevator = False
                        else:
                            ex, ey = elev
                            tile = int(self._HR_INT_TILE_SIZE)
                            target = pygame.Vector2((ex + 0.5) * tile, (ey + 0.5) * tile)
                            diff = target - pygame.Vector2(self.hr_int_pos)
                            d2 = float(diff.length_squared())
                            if d2 <= 3.0 * 3.0:
                                self.hr_auto_walk_to_elevator = False
                                move = pygame.Vector2(0, 0)
                                if not bool(getattr(self, "hr_elevator_ui_open", False)):
                                    self._hr_elevator_open()
                            else:
                                if diff.length_squared() > 0.001:
                                    move = diff.normalize()
                                    base_speed = 56.0

            self.hr_int_vel = move * base_speed
            self.player_sprinting = False
            if self.hr_int_vel.length_squared() > 0.1:
                self.hr_int_facing = pygame.Vector2(self.hr_int_vel).normalize()
                self.hr_int_walk_phase += dt * 10.0 * (self.hr_int_vel.length() / base_speed)
            else:
                self.hr_int_walk_phase *= 0.90
            self.hr_int_pos = self._hr_int_move_box(
                self.hr_int_pos,
                self.hr_int_vel,
                dt,
                w=int(getattr(self, "rv_int_player_w", self._RV_INT_PLAYER_W)),
                h=int(getattr(self, "rv_int_player_h", self._RV_INT_PLAYER_H)),
            )
            self.player.vel.update(0, 0)
            self.rv.vel.update(0, 0)
            self.bike.vel.update(0, 0)
            self._update_gun_timers(dt, allow_fire=False)
            return

        if getattr(self, "rv_interior", False):
            base_speed = 60.0
            self.rv_int_vel = move * base_speed
            self.player_sprinting = False
            if self.rv_int_vel.length_squared() > 0.1:
                self.rv_int_facing = pygame.Vector2(self.rv_int_vel).normalize()
                self.rv_int_walk_phase += dt * 10.0 * (self.rv_int_vel.length() / base_speed)
            else:
                self.rv_int_walk_phase *= 0.90
            self.rv_int_pos = self._rv_int_move_box(
                self.rv_int_pos,
                self.rv_int_vel,
                dt,
                w=int(getattr(self, "rv_int_player_w", self._RV_INT_PLAYER_W)),
                h=int(getattr(self, "rv_int_player_h", self._RV_INT_PLAYER_H)),
            )
            self.player.pos.update(self.rv.pos)
            self.player.vel.update(0, 0)
            self._update_gun_timers(dt, allow_fire=False)
            return

        if self.mount == "rv":
            mid = getattr(self.rv, "model_id", "rv")
            model = self._CAR_MODELS.get(str(mid))
            if model is None:
                model = self._CAR_MODELS.get("rv") or next(iter(self._CAR_MODELS.values()))

            throttle = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle -= 1.0
            steer_in = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steer_in -= 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steer_in += 1.0

            tx = int(math.floor(self.rv.pos.x / self.TILE_SIZE))
            ty = int(math.floor(self.rv.pos.y / self.TILE_SIZE))
            tile_under = self.world.get_tile(tx, ty)
            traction = float(self._tile_slow(tile_under))

            boost = 1.15 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
            fuel_mod = 0.55 if float(self.rv.fuel) <= 0.0 else 1.0
            max_fwd = float(model.max_fwd) * traction * fuel_mod * boost
            max_rev = float(model.max_rev) * traction * fuel_mod
            accel = float(model.accel) * traction * fuel_mod * boost
            brake = float(model.brake) * traction
            drag = 1.8 + (1.0 - traction) * 2.2

            max_steer = float(getattr(model, "steer_max", 0.62))  # radians
            steer_resp = 10.0
            target_steer = float(clamp(steer_in, -1.0, 1.0)) * max_steer        
            self.rv.steer = float(self.rv.steer + (target_steer - float(self.rv.steer)) * clamp(dt * steer_resp, 0.0, 1.0))

            speed = float(self.rv.speed)
            if throttle > 0.0:
                speed += accel * dt
            elif throttle < 0.0:
                if speed > 8.0:
                    speed -= brake * dt
                else:
                    speed -= (accel * 0.75) * dt
            else:
                speed *= max(0.0, 1.0 - drag * dt)

            speed = float(clamp(speed, -max_rev, max_fwd))
            self.rv.speed = speed

            wheelbase = float(max(8.0, getattr(model, "wheelbase", 20.0)))
            turn = 0.0
            if abs(speed) > 0.5:
                turn = (speed / wheelbase) * math.tan(float(self.rv.steer))
            self.rv.heading = float((float(self.rv.heading) + turn * dt) % math.tau)

            forward = pygame.Vector2(math.cos(float(self.rv.heading)), math.sin(float(self.rv.heading)))
            self.rv.vel = forward * speed

            if abs(speed) > 4.0:
                self.rv_anim += dt * (2.0 + abs(speed) * 0.06)
            else:
                self.rv_anim *= 0.85

            before = pygame.Vector2(self.rv.pos)
            self.rv.pos = self._move_box_vehicle(self.rv.pos, self.rv.vel, dt, w=self.rv.w, h=self.rv.h)
            dist = float((self.rv.pos - before).length())
            if dist > 0.001:
                self.rv.fuel = max(0.0, float(self.rv.fuel) - dist * float(getattr(model, "fuel_per_px", 0.0125)))
            self.player.pos.update(self.rv.pos)
            self.player.vel.update(self.rv.vel)
            self.player.walk_phase *= 0.85
            self.player_sprinting = False
        elif self.mount == "bike":
            base_speed = float(self._bike_base_speed())
            tx = int(math.floor(self.bike.pos.x / self.TILE_SIZE))       
            ty = int(math.floor(self.bike.pos.y / self.TILE_SIZE))        
            tile_under = self.world.get_tile(tx, ty)
            speed = base_speed * self._tile_slow(tile_under)
            speed *= self._weather_move_mult()

            mid = str(getattr(self.bike, "model_id", "bike"))
            uses_fuel = bool(mid.startswith("moto"))
            fuel = float(getattr(self.bike, "fuel", 0.0))
            if uses_fuel and fuel > 0.05:
                self._bike_no_fuel_warned = False

            want_sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            moving_now = move.length_squared() > 0.001
            if uses_fuel and fuel <= 0.01:
                if moving_now and want_sprint and not bool(getattr(self, "_bike_no_fuel_warned", False)):
                    self._set_hint("摩托车没油了", seconds=1.0)
                    self._bike_no_fuel_warned = True
                # Dead engine: push slowly (also uses stamina a bit).
                speed *= 0.55
                stamina = float(self.player.stamina)
                if moving_now:
                    self.player.stamina = float(clamp(stamina - dt * 10.0, 0.0, 100.0))
                else:
                    self.player.stamina = float(clamp(stamina + dt * 12.0, 0.0, 100.0))
                want_sprint = False
            elif self._bike_uses_stamina():
                stamina = float(self.player.stamina)
                can_sprint = bool(want_sprint and moving_now and stamina > 10.0)
                if can_sprint:
                    speed *= 1.25
                    self.player.stamina = float(clamp(stamina - dt * 18.0, 0.0, 100.0))
                else:
                    self.player.stamina = float(clamp(stamina + dt * 10.0, 0.0, 100.0))
            else:
                # Motorized two-wheelers: shift is throttle, stamina should recover.
                if want_sprint and moving_now:
                    speed *= 1.35
                self.player.stamina = float(clamp(float(self.player.stamina) + dt * 14.0, 0.0, 100.0))

            self.bike.vel = move * speed
            if self.bike.vel.length_squared() > 0.1:
                if abs(self.bike.vel.y) >= abs(self.bike.vel.x):
                    self.bike_dir = "down" if self.bike.vel.y >= 0 else "up"
                else:
                    self.bike_dir = "right" if self.bike.vel.x >= 0 else "left"
                self.bike_anim += dt * 10.0
            else:
                self.bike_anim *= 0.85

            before = pygame.Vector2(self.bike.pos)
            self.bike.pos = self._move_box_vehicle(self.bike.pos, self.bike.vel, dt, w=self.bike.w, h=self.bike.h)
            dist = float((self.bike.pos - before).length())
            if uses_fuel and dist > 0.001:
                fuel_per_px = {
                    "moto": 0.0024,
                    "moto_lux": 0.0022,
                    "moto_long": 0.0026,
                }.get(mid, 0.0024)
                before_fuel = float(getattr(self.bike, "fuel", 0.0))
                self.bike.fuel = float(clamp(before_fuel - dist * float(fuel_per_px), 0.0, 100.0))
                if before_fuel > 0.01 and float(self.bike.fuel) <= 0.01:
                    self._set_hint("摩托车没油了", seconds=1.0)
                    self._bike_no_fuel_warned = True
            self.player.pos.update(self.bike.pos)
            self.player.vel.update(self.bike.vel)
            self.player.walk_phase *= 0.85
            self.player_sprinting = False
        else:
            base_speed = 60.0
            tx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
            ty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
            tile_under = self.world.get_tile(tx, ty)
            speed = base_speed * self._tile_slow(tile_under)
            speed *= 0.80 + 0.20 * (float(self.player.condition) / 100.0)
            speed *= self._weather_move_mult()

            want_sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            moving_now = move_raw.length_squared() > 0.001
            sprint_intent = bool(want_sprint and moving_now)
            stamina = float(self.player.stamina)

            # Sprint gating with hysteresis: once stamina is depleted, the
            # player must recover some stamina before sprinting again. This
            # prevents tiny "bursts" of sprint when stamina hovers near zero.
            sprint_start_need = 25.0
            sprint_stop_need = 8.0
            was_sprinting = bool(getattr(self, "player_sprinting", False))
            if sprint_intent:
                can_sprint = stamina > (sprint_stop_need if was_sprinting else sprint_start_need)
            else:
                can_sprint = False

            if can_sprint:
                speed *= 1.35
                self.player.stamina = float(clamp(stamina - dt * 28.0, 0.0, 100.0))
            else:
                regen_mod = 0.50 + 0.50 * (min(self.player.hunger, self.player.thirst) / 100.0)
                regen_mod *= 0.55 + 0.45 * (self.player.morale / 100.0)
                self.player.stamina = float(clamp(stamina + dt * 16.0 * regen_mod, 0.0, 100.0))
            self.player_sprinting = bool(can_sprint)
            # Keep raw 8-way input (no normalization) so diagonal walking
            # doesn't "stutter" in pixel space.
            self.player.vel = move_raw * speed

            if self.player.vel.length_squared() > 0.1:
                self.player.walk_phase += dt * 10.0 * (self.player.vel.length() / base_speed)
            else:
                self.player.walk_phase *= 0.92

            # Walk-into-door: special buildings (highrise/school) are portals.
            if self._try_walk_into_special_door(move, dt):
                return

            self.player.pos = self._move_box(self.player.pos, self.player.vel, dt, w=self.player.w, h=self.player.h)

        focus = pygame.Vector2(self.player.pos)
        target_cam_x = float(focus.x - INTERNAL_W / 2)
        target_cam_y = float(focus.y - INTERNAL_H / 2)

        # Camera: direct follow (no smoothing). Keep float + int versions.
        self.cam_fx = float(target_cam_x)
        self.cam_fy = float(target_cam_y)
        # Pixel-perfect camera: snap to int pixels. Use iround() (instead of
        # floor) to avoid diagonal "flicker" from asymmetric rounding.
        self.cam_x = int(iround(float(self.cam_fx)))
        self.cam_y = int(iround(float(self.cam_fy)))

        self._stream_world_chunks()

        if self.gun is not None and float(self.gun.reload_left) > 0.0 and getattr(self, "_reload_lock_dir", None) is not None:
            lock = pygame.Vector2(self._reload_lock_dir)
            if lock.length_squared() > 0.001:
                self.aim_dir = lock.normalize()
            else:
                self.aim_dir = pygame.Vector2(1, 0)
        else:
            self.aim_dir = self._compute_aim_dir()

        self._update_gun_timers(dt, allow_fire=True)

        self._update_zombies(dt)
        self._update_bullets(dt)

    def _set_hint(self, text: str, *, seconds: float = 1.2) -> None:
        self.hint_text = str(text)
        self.hint_left = float(seconds)

    def _player_tile(self) -> tuple[int, int]:
        tx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
        ty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
        return tx, ty

    def _player_chunk(self) -> tuple[int, int]:
        tx, ty = self._player_tile()
        return tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE

    def _stream_world_chunks(self) -> None:
        # Stream/generate chunks ahead of the camera to avoid hitches when
        # walking diagonally across chunk corners. Render code only "peeks"
        # tiles/chunks and never generates them.
        if bool(getattr(self, "rv_interior", False)) or bool(getattr(self, "hr_interior", False)) or bool(
            getattr(self, "sch_interior", False)
        ):
            return

        tx, ty = self._player_tile()
        pcx = int(tx) // int(self.CHUNK_SIZE)
        pcy = int(ty) // int(self.CHUNK_SIZE)

        # Always ensure the current chunk exists (physics/collisions rely on it).
        self.world.get_chunk(int(pcx), int(pcy))

        # Collision safety near chunk edges: ensure neighbor chunks that the
        # player's collider might touch soon.
        lx = int(tx - pcx * int(self.CHUNK_SIZE))
        ly = int(ty - pcy * int(self.CHUNK_SIZE))
        margin = 3  # tiles
        need_x = [0]
        if lx <= int(margin):
            need_x.append(-1)
        elif lx >= int(self.CHUNK_SIZE) - 1 - int(margin):
            need_x.append(1)
        need_y = [0]
        if ly <= int(margin):
            need_y.append(-1)
        elif ly >= int(self.CHUNK_SIZE) - 1 - int(margin):
            need_y.append(1)
        for oy in need_y:
            for ox in need_x:
                if int(ox) == 0 and int(oy) == 0:
                    continue
                self.world.get_chunk(int(pcx + ox), int(pcy + oy))

        # Request chunks in/around the camera view for smooth scrolling.
        cam_x = int(getattr(self, "cam_x", 0))
        cam_y = int(getattr(self, "cam_y", 0))
        start_tx = int(math.floor(float(cam_x) / float(self.TILE_SIZE))) - 2
        start_ty = int(math.floor(float(cam_y) / float(self.TILE_SIZE))) - 2
        end_tx = int(math.floor(float(cam_x + INTERNAL_W) / float(self.TILE_SIZE))) + 2
        end_ty = int(math.floor(float(cam_y + INTERNAL_H) / float(self.TILE_SIZE))) + 2

        chunk_pad = 1
        start_cx = start_tx // int(self.CHUNK_SIZE) - int(chunk_pad)
        end_cx = end_tx // int(self.CHUNK_SIZE) + int(chunk_pad)
        start_cy = start_ty // int(self.CHUNK_SIZE) - int(chunk_pad)
        end_cy = end_ty // int(self.CHUNK_SIZE) + int(chunk_pad)
        for cy in range(int(start_cy), int(end_cy) + 1):
            for cx in range(int(start_cx), int(end_cx) + 1):
                self.world.request_chunk(int(cx), int(cy))

        tick = int(getattr(self, "_stream_tick", 0)) + 1
        self._stream_tick = int(tick)

        mount = getattr(self, "mount", None)
        moving = False
        if mount == "rv" and getattr(self, "rv", None) is not None:
            moving = self.rv.vel.length_squared() > 0.1
        elif mount == "bike" and getattr(self, "bike", None) is not None:
            moving = self.bike.vel.length_squared() > 0.1
        else:
            moving = self.player.vel.length_squared() > 0.1

        # Budgeted generation: generate more when standing still; while moving,
        # generate occasionally to keep ahead without constant stalls.
        budget = 2 if not moving else 0
        if moving:
            interval = 6 if mount is not None else 12
            if (tick % int(interval)) == 0:
                budget = 1
        if budget > 0:
            self.world.pump_generation(max_chunks=int(budget))

    def _find_nearest_item(self, *, radius_px: float) -> tuple["_Chunk | None", "_WorldItem | None"]:
        cx, cy = self._player_chunk()
        best_d2 = float(radius_px * radius_px)
        best_item: HardcoreSurvivalState._WorldItem | None = None
        best_chunk: HardcoreSurvivalState._Chunk | None = None

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                chunk = self.world.get_chunk(cx + ox, cy + oy)
                for it in chunk.items:
                    d2 = float((it.pos - self.player.pos).length_squared())
                    if d2 < best_d2:
                        best_d2 = d2
                        best_item = it
                        best_chunk = chunk

        return best_chunk, best_item

    def _find_nearest_parked_bike(
        self, *, radius_px: float
    ) -> tuple["_Chunk | None", "_ParkedBike | None", float]:
        cx, cy = self._player_chunk()
        best_d2 = float(radius_px * radius_px)
        best_bike: HardcoreSurvivalState._ParkedBike | None = None
        best_chunk: HardcoreSurvivalState._Chunk | None = None

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                chunk = self.world.get_chunk(cx + ox, cy + oy)
                for b in getattr(chunk, "bikes", []):
                    pos = getattr(b, "pos", None)
                    if pos is None:
                        continue
                    d2 = float((pygame.Vector2(pos) - self.player.pos).length_squared())
                    if d2 < best_d2:
                        best_d2 = d2
                        best_bike = b
                        best_chunk = chunk

        return best_chunk, best_bike, best_d2

    def _stash_personal_bike_as_parked(self) -> None:
        if not hasattr(self, "bike") or self.bike is None:
            return
        mid = str(getattr(self.bike, "model_id", "bike"))
        d = str(getattr(self, "bike_dir", "right"))
        pos = pygame.Vector2(self.bike.pos)
        w, h = self._two_wheel_collider_px(mid)

        def rect_at(p: pygame.Vector2) -> pygame.Rect:
            return pygame.Rect(
                int(iround(float(p.x) - float(w) / 2.0)),
                int(iround(float(p.y) - float(h) / 2.0)),
                int(w),
                int(h),
            )

        def inside_any_building(tx: int, ty: int) -> bool:
            ch = self.world.get_chunk(int(tx) // self.CHUNK_SIZE, int(ty) // self.CHUNK_SIZE)
            for bx0, by0, bw, bh, _roof_kind, _floors in getattr(ch, "buildings", []):
                if int(bx0) <= int(tx) < int(bx0) + int(bw) and int(by0) <= int(ty) < int(by0) + int(bh):
                    return True
            return False

        # Don't leave a parked bike "on top of" building roofs/facades.
        tx0 = int(math.floor(pos.x / self.TILE_SIZE))
        ty0 = int(math.floor(pos.y / self.TILE_SIZE))
        if inside_any_building(tx0, ty0) or self._collide_rect_world_vehicle(rect_at(pos)):
            found: pygame.Vector2 | None = None
            # Spiral search around the current tile.
            search_r = 6
            for r in range(int(search_r) + 1):
                for dy in range(-int(r), int(r) + 1):
                    for dx in range(-int(r), int(r) + 1):
                        if max(abs(int(dx)), abs(int(dy))) != int(r):
                            continue
                        tx = int(tx0 + int(dx))
                        ty = int(ty0 + int(dy))
                        if inside_any_building(tx, ty):
                            continue
                        p = pygame.Vector2(
                            (float(tx) + 0.5) * float(self.TILE_SIZE),
                            (float(ty) + 0.5) * float(self.TILE_SIZE),
                        )
                        if self._collide_rect_world_vehicle(rect_at(p)):
                            continue
                        found = p
                        break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                return
            pos = found

        tx = int(math.floor(pos.x / self.TILE_SIZE))
        ty = int(math.floor(pos.y / self.TILE_SIZE))
        chunk = self.world.get_chunk(tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE)
        for pb in getattr(chunk, "bikes", []):
            if (pygame.Vector2(getattr(pb, "pos", pos)) - pos).length_squared() <= (6.0 * 6.0):
                return
        chunk.bikes.append(
            HardcoreSurvivalState._ParkedBike(
                pos=pos,
                model_id=mid,
                dir=d,
                frame=0,
                fuel=float(getattr(self.bike, "fuel", 0.0)),
            )
        )

    def _special_building_for_door_tile(
        self, tx: int, ty: int
    ) -> HardcoreSurvivalState._SpecialBuilding | None:
        door = (int(tx), int(ty))
        cx = int(door[0]) // self.CHUNK_SIZE
        cy = int(door[1]) // self.CHUNK_SIZE
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                chunk = self.world.get_chunk(int(cx + ox), int(cy + oy))
                for sb in getattr(chunk, "special_buildings", []):
                    if door in getattr(sb, "door_tiles", ()):
                        return sb
        return None

    def _try_walk_into_special_door(self, move: pygame.Vector2, dt: float) -> bool:
        if self.mount is not None:
            return False
        if move.length_squared() <= 0.001:
            return False

        rect = self.player.rect_at(self.player.pos)
        dx_f = float(self.player.vel.x) * float(dt)
        dy_f = float(self.player.vel.y) * float(dt)
        dx = int(round(dx_f))
        dy = int(round(dy_f))
        if dx == 0 and abs(dx_f) > 0.001:
            dx = 1 if dx_f > 0 else -1
        if dy == 0 and abs(dy_f) > 0.001:
            dy = 1 if dy_f > 0 else -1
        if dx == 0 and dy == 0:
            return False

        next_rect = rect.move(int(dx), int(dy))
        tx, ty = self._player_tile()
        candidates = [
            (tx, ty),
            (tx + 1, ty),
            (tx - 1, ty),
            (tx, ty + 1),
            (tx, ty - 1),
            (tx + 1, ty + 1),
            (tx + 1, ty - 1),
            (tx - 1, ty + 1),
            (tx - 1, ty - 1),
        ]
        for cx, cy in candidates:
            cx = int(cx)
            cy = int(cy)
            if int(self.world.get_tile(cx, cy)) != int(self.T_DOOR):
                continue
            door_rect = pygame.Rect(
                int(cx * self.TILE_SIZE),
                int(cy * self.TILE_SIZE),
                int(self.TILE_SIZE),
                int(self.TILE_SIZE),
            )
            if not next_rect.colliderect(door_rect):
                continue

            # Only enter if you're actually walking *towards* the door.
            door_center = pygame.Vector2(
                (float(cx) + 0.5) * float(self.TILE_SIZE),
                (float(cy) + 0.5) * float(self.TILE_SIZE),
            )
            to_door = door_center - pygame.Vector2(self.player.pos)
            if to_door.length_squared() > 0.001:
                to_door = to_door.normalize()
                if float(move.dot(to_door)) < 0.25:
                    continue

            sb = self._special_building_for_door_tile(cx, cy)
            if sb is None:
                continue
            kind = str(getattr(sb, "kind", ""))
            if kind == "highrise":
                self._hr_interior_enter(sb)
                return True
            if kind == "school":
                self._sch_interior_enter(sb)
                return True
        return False

    def _interact_primary(self) -> None:
        # Primary interact key on the world map: pickup only (doors are walk-into).
        self._try_pickup()

    def _try_enter_highrise(self) -> bool:
        if getattr(self, "hr_interior", False):
            return False
        tx, ty = self._player_tile()
        candidates = [
            (tx, ty),
            (tx + 1, ty),
            (tx - 1, ty),
            (tx, ty + 1),
            (tx, ty - 1),
            (tx + 1, ty + 1),
            (tx + 1, ty - 1),
            (tx - 1, ty + 1),
            (tx - 1, ty - 1),
        ]
        for cx, cy in candidates:
            if self.world.get_tile(int(cx), int(cy)) != int(self.T_DOOR):
                continue
            chunk = self.world.get_chunk(int(cx) // self.CHUNK_SIZE, int(cy) // self.CHUNK_SIZE)
            for sb in getattr(chunk, "special_buildings", []):
                if getattr(sb, "kind", "") != "highrise":
                    continue
                if (int(cx), int(cy)) not in set(getattr(sb, "door_tiles", ())):
                    continue
                if self.mount is not None:
                    self._set_hint("下车后再进入高层住宅", seconds=1.2)
                    return True
                self._hr_interior_enter(sb)
                return True
        return False

    def _try_enter_school(self) -> bool:
        if getattr(self, "sch_interior", False):
            return False
        tx, ty = self._player_tile()
        candidates = [
            (tx, ty),
            (tx + 1, ty),
            (tx - 1, ty),
            (tx, ty + 1),
            (tx, ty - 1),
            (tx + 1, ty + 1),
            (tx + 1, ty - 1),
            (tx - 1, ty + 1),
            (tx - 1, ty - 1),
        ]
        for cx, cy in candidates:
            if self.world.get_tile(int(cx), int(cy)) != int(self.T_DOOR):
                continue
            chunk = self.world.get_chunk(int(cx) // self.CHUNK_SIZE, int(cy) // self.CHUNK_SIZE)
            for sb in getattr(chunk, "special_buildings", []):
                if getattr(sb, "kind", "") != "school":
                    continue
                if (int(cx), int(cy)) not in set(getattr(sb, "door_tiles", ())):
                    continue
                if self.mount is not None:
                    self._set_hint("下车后再进入学校", seconds=1.2)
                    return True
                self._sch_interior_enter(sb)
                return True
        return False

    def _try_pickup(self) -> None:
        chunk, it = self._find_nearest_item(radius_px=18.0)
        if it is None or chunk is None:
            self._set_hint("附近没有可拾取物")
            return
        idef = self._ITEMS.get(it.item_id)
        if idef is None:
            self._set_hint("未知物品")
            return
        left = self.inventory.add(it.item_id, int(it.qty), self._ITEMS)
        taken = int(it.qty) - int(left)
        if taken <= 0:
            self._set_hint("背包满了")
            return
        if left <= 0:
            try:
                chunk.items.remove(it)
            except ValueError:
                pass
        else:
            it.qty = int(left)
        self._set_hint(f"拾取 {idef.name} x{taken}")

    def _drop_selected(self) -> None:
        stack = self.inventory.drop_slot(int(self.inv_index))
        if stack is None:
            self._set_hint("空槽位")
            return
        idef = self._ITEMS.get(stack.item_id)
        name = idef.name if idef is not None else stack.item_id
        tx, ty = self._player_tile()
        chunk = self.world.get_chunk(tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE)
        offset = pygame.Vector2(random.randint(-6, 6), random.randint(-6, 6))
        chunk.items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(self.player.pos) + offset, item_id=stack.item_id, qty=int(stack.qty)))
        self._set_hint(f"丢弃 {name} x{stack.qty}")

    def _use_consumable(self, item_id: str) -> bool:
        item_id = str(item_id)
        if item_id == "cola":
            self.player.thirst = clamp(self.player.thirst + 42.0, 0.0, 100.0)
            self.player.morale = clamp(self.player.morale + 5.0, 0.0, 100.0)
            self.player.stamina = clamp(self.player.stamina + 6.0, 0.0, 100.0)
            self._set_hint("喝了可乐：口渴++ 心态+")
            return True
        if item_id == "food_can":
            self.player.hunger = clamp(self.player.hunger + 38.0, 0.0, 100.0)
            self.player.morale = clamp(self.player.morale + 4.0, 0.0, 100.0)
            self.player.condition = clamp(self.player.condition + 2.0, 0.0, 100.0)
            self._set_hint("吃了罐头：饥饿+ 心态+")
            return True
        if item_id == "water":
            self.player.thirst = clamp(self.player.thirst + 50.0, 0.0, 100.0)
            self.player.morale = clamp(self.player.morale + 2.0, 0.0, 100.0)
            self._set_hint("喝了水：口渴+")
            return True
        if item_id == "bandage":
            self.player.hp = int(clamp(self.player.hp + 18, 0, 100))
            self.player.condition = clamp(self.player.condition + 3.0, 0.0, 100.0)
            self._set_hint("使用绷带：生命+")
            return True
        if item_id == "medkit":
            self.player.hp = int(clamp(self.player.hp + 55, 0, 100))
            self.player.condition = clamp(self.player.condition + 10.0, 0.0, 100.0)
            self.player.morale = clamp(self.player.morale + 3.0, 0.0, 100.0)
            self._set_hint("使用急救包：生命++")
            return True
        return False

    def _equip_selected(self) -> None:
        if not (0 <= int(self.inv_index) < len(self.inventory.slots)):
            return
        stack = self.inventory.slots[int(self.inv_index)]
        if stack is None:
            self._set_hint("空槽位")
            return
        idef = self._ITEMS.get(stack.item_id)
        if idef is None:
            self._set_hint("未知物品")
            return

        if idef.kind != "gun":
            if idef.kind in ("food", "drink", "med"):
                if not self._use_consumable(stack.item_id):
                    self._set_hint("暂时不能使用")
                    return
                stack.qty -= 1
                if stack.qty <= 0:
                    self.inventory.slots[int(self.inv_index)] = None
                return
            self._set_hint("不能使用/装备")
            return

        gun_def = self._GUNS.get(stack.item_id)
        if gun_def is None:
            self._set_hint("未知枪械")
            return

        if stack.qty > 1:
            stack.qty -= 1
        else:
            self.inventory.slots[int(self.inv_index)] = None

        if self.gun is not None:
            left = self.inventory.add(self.gun.gun_id, 1, self._ITEMS)
            if left > 0:
                self.inventory.add(stack.item_id, 1, self._ITEMS)
                self._set_hint("背包满了，无法换枪")
                return

        self.gun = HardcoreSurvivalState._Gun(gun_id=gun_def.id, mag=int(gun_def.mag_size))
        self._reload_lock_dir = None
        self._set_hint(f"装备 {gun_def.name}")

    def _start_reload(self) -> None:
        if self.gun is None:
            self._set_hint("没有装备枪")
            return
        gun_def = self._GUNS.get(self.gun.gun_id)
        if gun_def is None:
            self._set_hint("未知枪械")
            return
        if self.gun.reload_left > 0.0:
            return
        need = int(gun_def.mag_size) - int(self.gun.mag)
        if need <= 0:
            self._set_hint("弹匣已满")
            return
        if self.inventory.count(gun_def.ammo_item) <= 0:
            self._set_hint("没有弹药")
            return
        lock = pygame.Vector2(self.aim_dir)
        if lock.length_squared() <= 0.001:
            lock = pygame.Vector2(self.player.facing)
        self._reload_lock_dir = lock.normalize() if lock.length_squared() > 0.001 else pygame.Vector2(1, 0)
        self.gun.reload_total = float(gun_def.reload_s)
        self.gun.reload_left = float(gun_def.reload_s)
        self._set_hint("换弹中…", seconds=0.8)

    def _toggle_vehicle(self) -> None:
        if self.mount is not None:
            if self.mount == "rv":
                vehicle_pos = pygame.Vector2(self.rv.pos)
                vw, vh = float(self.rv.w), float(self.rv.h)
                self.rv.vel.update(0, 0)
                self.rv.speed = 0.0
                self.rv.steer = 0.0
            else:
                vehicle_pos = pygame.Vector2(self.bike.pos)
                vw, vh = float(self.bike.w), float(self.bike.h)
                self.bike.vel.update(0, 0)

            exits = [
                pygame.Vector2(vw / 2 + self.player.w / 2 + 4, 0),
                pygame.Vector2(-(vw / 2 + self.player.w / 2 + 4), 0),
                pygame.Vector2(0, vh / 2 + self.player.h / 2 + 4),
                pygame.Vector2(0, -(vh / 2 + self.player.h / 2 + 4)),
            ]
            placed = False
            for off in exits:
                p = vehicle_pos + off
                if not self._collide_rect_world(self.player.rect_at(p)):
                    self.player.pos.update(p)
                    placed = True
                    break
            if not placed:
                self.player.pos.update(vehicle_pos + pygame.Vector2(vw / 2 + 12, 0))

            self.mount = None
            self._set_hint("下车")
            return

        rv_d2 = float((self.player.pos - self.rv.pos).length_squared())
        bike_d2 = float((self.player.pos - self.bike.pos).length_squared())
        parked_chunk, parked_bike, parked_d2 = self._find_nearest_parked_bike(radius_px=18.0)
        can_rv = rv_d2 <= (22.0 * 22.0)
        can_bike = bike_d2 <= (18.0 * 18.0)
        can_parked = parked_bike is not None and parked_chunk is not None
        if not can_rv and not can_bike and not can_parked:
            self._set_hint("离载具太远")
            return

        if can_parked and (not can_rv or parked_d2 <= rv_d2) and (not can_bike or parked_d2 < bike_d2):
            # Take a parked two-wheeler (including motorcycles) and mount it.
            try:
                self._stash_personal_bike_as_parked()
            except Exception:
                pass
            pb = parked_bike
            self.bike.pos.update(pygame.Vector2(getattr(pb, "pos", self.player.pos)))
            self.bike.vel.update(0, 0)
            self.bike.model_id = str(getattr(pb, "model_id", "bike"))
            self.bike.fuel = float(getattr(pb, "fuel", 0.0))
            self.bike_dir = str(getattr(pb, "dir", "right"))
            self.bike_anim = 0.0
            self._apply_bike_model()
            try:
                parked_chunk.bikes.remove(pb)
            except Exception:
                pass
            self.mount = "bike"
            self.player.pos.update(self.bike.pos)
            self._set_hint(f"骑上 {self._two_wheel_name(self.bike.model_id)}")
            return

        if can_bike and (not can_rv or bike_d2 <= rv_d2):
            self.mount = "bike"
            self.player.pos.update(self.bike.pos)
            self._set_hint(f"骑上 {self._two_wheel_name(getattr(self.bike, 'model_id', 'bike'))}")
            return

        self.mount = "rv"
        self.player.pos.update(self.rv.pos)
        self._set_hint("上房车")

    def _compute_aim_dir(self) -> pygame.Vector2:
        m = self.app.screen_to_internal(pygame.mouse.get_pos())
        if m is None:
            d = pygame.Vector2(self.player.facing)
            return d.normalize() if d.length_squared() > 0.001 else pygame.Vector2(1, 0)
        mw = pygame.Vector2(m) + pygame.Vector2(float(self.cam_x), float(self.cam_y))
        v = mw - self.player.pos
        if v.length_squared() <= 0.001:
            d = pygame.Vector2(self.player.facing)
            return d.normalize() if d.length_squared() > 0.001 else pygame.Vector2(1, 0)
        return v.normalize()

    def _fire(self) -> None:
        if self.gun is None:
            return
        gun_def = self._GUNS.get(self.gun.gun_id)
        if gun_def is None:
            return
        if self.gun.reload_left > 0.0 or self.gun.cooldown_left > 0.0:
            return
        if int(self.gun.mag) <= 0:
            self._set_hint("没子弹：按R换弹", seconds=0.8)
            return

        aim = pygame.Vector2(self.aim_dir)
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(self.player.facing)
        if aim.length_squared() <= 0.001:
            aim = pygame.Vector2(1, 0)
        aim = aim.normalize()
        base_ang = math.atan2(aim.y, aim.x)
        spread = math.radians(float(gun_def.spread_deg))
        ang = base_ang + random.uniform(-spread, spread)
        d = pygame.Vector2(math.cos(ang), math.sin(ang))

        # Spawn from muzzle (hand-bone) so visuals match bullets.
        height_delta = 0
        av = getattr(self, "avatar", None)
        if av is not None:
            hidx = int(getattr(av, "height", 1))
            height_delta = 1 if hidx == 0 else -1 if hidx == 2 else 0

        face = pygame.Vector2(aim)
        if abs(face.y) >= abs(face.x):
            spr_dir = "down" if face.y >= 0 else "up"
        else:
            spr_dir = "right" if face.x >= 0 else "left"

        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = pf.get(spr_dir, pf["down"])
        moving = self.player.vel.length_squared() > 1.0
        bob = 0
        step_idx = 0
        idle_anim = True
        if moving and len(frames) > 1:
            idle_anim = False
            walk = frames[1:]
            phase = (float(self.player.walk_phase) % math.tau) / math.tau
            step_idx = int(phase * len(walk)) % len(walk)
            if is_run and step_idx in (1, 4):
                bob = 1
            spr = walk[step_idx]
        else:
            spr = frames[0]

        rect = spr.get_rect()
        rect.midbottom = (
            iround(float(self.player.pos.x)),
            iround(float(self.player.pos.y) + float(self.player.h) / 2.0),
        )
        rect.y += int(bob)
        sk = self._survivor_skeleton_nodes(
            spr_dir,
            int(step_idx),
            idle=bool(idle_anim),
            height_delta=int(height_delta),
            run=bool(is_run),
        )
        hand_node = sk.get("r_hand")
        if hand_node is not None:
            hx, hy = hand_node
            base_hand = pygame.Vector2(int(rect.left + hx), int(rect.top + hy))
        else:
            base_hand = pygame.Vector2(rect.centerx, rect.centery + 3)
        muzzle = base_hand + aim * 17.0
        spawn = pygame.Vector2(muzzle)
        self.bullets.append(
            HardcoreSurvivalState._Bullet(
                pos=spawn,
                vel=d * float(gun_def.bullet_speed),
                ttl=1.35,
                dmg=int(gun_def.damage),
            )
        )
        self.gun.mag = int(self.gun.mag) - 1
        self.gun.cooldown_left = 1.0 / max(0.1, float(gun_def.fire_rate))
        self.app.play_sfx("shot")
        self.muzzle_flash_left = 0.06
        self.noise_left = max(float(getattr(self, "noise_left", 0.0)), 0.55)
        self.noise_radius = max(float(getattr(self, "noise_radius", 0.0)), 220.0)

    def _update_bullets(self, dt: float) -> None:
        if not self.bullets:
            return
        alive: list[HardcoreSurvivalState._Bullet] = []
        for b in self.bullets:
            b.pos += b.vel * dt
            b.ttl -= dt
            if b.ttl <= 0.0:
                continue
            tx = int(math.floor(b.pos.x / self.TILE_SIZE))
            ty = int(math.floor(b.pos.y / self.TILE_SIZE))
            tile = self.world.get_tile(tx, ty)
            if self._tile_solid(tile):
                continue

            hit = False
            for z in self.zombies:
                if z.rect().collidepoint(int(round(b.pos.x)), int(round(b.pos.y))):
                    z.hp = int(z.hp) - int(b.dmg)
                    if int(z.hp) > 0:
                        z.stagger_left = max(float(getattr(z, "stagger_left", 0.0)), 0.12)
                        if b.vel.length_squared() > 0.1:
                            z.stagger_vel = pygame.Vector2(b.vel).normalize() * 85.0
                    else:
                        self._on_zombie_killed(z)
                    hit = True
                    break
            if hit:
                continue
            alive.append(b)
        self.bullets = alive
        self.zombies = [z for z in self.zombies if int(z.hp) > 0]

    def _drop_world_item(self, pos: pygame.Vector2, item_id: str, qty: int) -> None:
        item_id = str(item_id)
        qty = int(qty)
        if qty <= 0:
            return
        tx = int(math.floor(float(pos.x) / self.TILE_SIZE))
        ty = int(math.floor(float(pos.y) / self.TILE_SIZE))
        chunk = self.world.get_chunk(tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE)
        chunk.items.append(HardcoreSurvivalState._WorldItem(pos=pygame.Vector2(pos), item_id=item_id, qty=qty))

    def _on_zombie_killed(self, z: "HardcoreSurvivalState._Zombie") -> None:
        kind = str(getattr(z, "kind", "walker"))
        pos = pygame.Vector2(z.pos)

        # Lightweight loot loop: enough to reinforce survival, not enough to break it.
        if random.random() < 0.22:
            self._drop_world_item(pos, "scrap", random.randint(1, 2))
        if random.random() < (0.12 if kind == "runner" else 0.06):
            self._drop_world_item(pos, "ammo_9mm", random.randint(3, 7))
        if random.random() < 0.07:
            self._drop_world_item(pos, "bandage", 1)
        if random.random() < 0.05:
            self._drop_world_item(pos, random.choice(["cola", "food_can", "water"]), 1)

    def _pick_zombie_kind(self, *, in_town: bool) -> str:
        season = int(self._season_index())
        tday = (float(self.world_time_s) % float(self.DAY_LENGTH_S)) / max(1e-6, float(self.DAY_LENGTH_S))
        night = bool(tday < 0.25 or tday > 0.75)

        weights: dict[str, float] = {"walker": 0.72, "runner": 0.22, "screamer": 0.06}
        if in_town:
            weights = {"walker": 0.55, "runner": 0.25, "screamer": 0.20}

        if night:
            weights["runner"] += 0.10
            weights["screamer"] += 0.06
            weights["walker"] -= 0.16

        # Season flavor (lightweight).
        if season == 1:  # summer
            weights["screamer"] += 0.05
            weights["walker"] -= 0.05
        elif season == 2:  # autumn
            weights["walker"] += 0.08
            weights["runner"] -= 0.04
        elif season == 3:  # winter
            weights["walker"] += 0.10
            weights["runner"] -= 0.06
            weights["screamer"] -= 0.04

        total = 0.0
        for v in weights.values():
            total += max(0.0, float(v))
        if total <= 0.0001:
            return "walker"
        r = random.random() * total
        acc = 0.0
        for k in ("walker", "runner", "screamer"):
            acc += max(0.0, float(weights.get(k, 0.0)))
            if r <= acc:
                return k
        return "walker"

    def _spawn_one_zombie(self, target_pos: pygame.Vector2, *, in_town: bool) -> None:
        kind = self._pick_zombie_kind(in_town=in_town)
        mdef = self._MONSTER_DEFS.get(kind, self._MONSTER_DEFS["walker"])
        for _ in range(14):
            ang = random.random() * math.tau
            dist = random.randint(110, 180) if in_town else random.randint(140, 240)
            pos = pygame.Vector2(target_pos) + pygame.Vector2(math.cos(ang), math.sin(ang)) * float(dist)
            tx = int(math.floor(pos.x / self.TILE_SIZE))
            ty = int(math.floor(pos.y / self.TILE_SIZE))
            if self._tile_solid(self.world.get_tile(tx, ty)):
                continue
            hp = int(mdef.hp) + (6 if in_town else 0)
            speed = float(mdef.speed)
            z = HardcoreSurvivalState._Zombie(
                pos=pos,
                vel=pygame.Vector2(0, 0),
                hp=hp,
                speed=speed,
                kind=str(kind),
                dir="down",
                anim=random.random() * 2.0,
                attack_left=random.random() * float(mdef.attack_cd),
                scream_left=(random.random() * float(mdef.scream_cd) if float(mdef.scream_cd) > 0.0 else 0.0),
                wander_dir=pygame.Vector2(0, 0),
                wander_left=0.0,
                stagger_left=0.0,
                stagger_vel=pygame.Vector2(0, 0),
            )
            self.zombies.append(z)
            return

    def _update_zombies(self, dt: float) -> None:
        target = pygame.Vector2(self.player.pos)

        cap = max(0, int(getattr(self, "zombie_cap", 8)))
        if cap <= 0:
            return

        self.spawn_left -= dt
        if self.spawn_left <= 0.0 and len(self.zombies) < cap:
            tx = int(math.floor(target.x / self.TILE_SIZE))
            ty = int(math.floor(target.y / self.TILE_SIZE))
            chunk = self.world.get_chunk(tx // self.CHUNK_SIZE, ty // self.CHUNK_SIZE)
            in_town = chunk.town_kind is not None
            self.spawn_left = 6.5 if in_town else 10.0
            spawn_n = 1
            for _ in range(spawn_n):
                self._spawn_one_zombie(target, in_town=in_town)

        if getattr(self, "zombie_frozen", False):
            for z in self.zombies:
                z.vel.update(0, 0)
            return

        for z in self.zombies:
            kind = str(getattr(z, "kind", "walker"))
            mdef = self._MONSTER_DEFS.get(kind, self._MONSTER_DEFS["walker"])

            z.attack_left = max(0.0, float(getattr(z, "attack_left", 0.0)) - dt)
            z.scream_left = max(0.0, float(getattr(z, "scream_left", 0.0)) - dt)
            z.wander_left = max(0.0, float(getattr(z, "wander_left", 0.0)) - dt)
            z.stagger_left = max(0.0, float(getattr(z, "stagger_left", 0.0)) - dt)

            if float(z.stagger_left) > 0.0:
                z.pos = self._move_box(z.pos, z.stagger_vel, dt, w=z.w, h=z.h)
                z.stagger_vel *= 0.86
                z.anim = float(z.anim) + dt * 10.0
                continue

            to_p = target - z.pos
            d2 = float(to_p.length_squared())

            sense = float(mdef.sense)
            # Vehicles are loud: pull monsters from farther away (even if RV is "safe").
            if self.mount == "rv":
                sense += 80.0 + min(120.0, abs(float(getattr(self.rv, "speed", 0.0))) * 1.1)
            if getattr(self, "noise_left", 0.0) > 0.0:
                sense = max(sense, float(getattr(self, "noise_radius", 0.0)))

            alerted = d2 <= (sense * sense)
            if alerted:
                if kind == "screamer" and float(z.scream_left) <= 0.0 and d2 <= float(mdef.scream_radius) ** 2:
                    z.scream_left = float(mdef.scream_cd)
                    if len(self.zombies) < cap:
                        for _ in range(int(mdef.scream_spawn)):
                            if len(self.zombies) >= cap:
                                break
                            self._spawn_one_zombie(target, in_town=True)
                    self._set_hint("尖叫声……更多怪物来了！", seconds=1.3)

                if d2 > 1.0:
                    dir_vec = to_p.normalize()
                    want = dir_vec * float(z.speed)
                    accel = 5.5 if kind == "runner" else 3.6 if kind == "screamer" else 2.8
                    a = float(clamp(1.0 - math.exp(-dt * float(accel)), 0.0, 1.0))
                    z.vel = z.vel.lerp(want, a)
                else:
                    z.vel.update(0, 0)
            else:
                if float(z.wander_left) <= 0.0 or z.wander_dir.length_squared() <= 0.001:
                    ang = random.random() * math.tau
                    z.wander_dir = pygame.Vector2(math.cos(ang), math.sin(ang))
                    z.wander_left = random.uniform(0.6, 1.8)
                want = pygame.Vector2(z.wander_dir) * float(mdef.roam_speed)
                accel = 3.0 if kind == "runner" else 2.2 if kind == "screamer" else 1.8
                a = float(clamp(1.0 - math.exp(-dt * float(accel)), 0.0, 1.0))
                z.vel = z.vel.lerp(want, a)

            if z.vel.length_squared() > 0.1:
                ztx = int(math.floor(z.pos.x / self.TILE_SIZE))
                zty = int(math.floor(z.pos.y / self.TILE_SIZE))
                z_under = self.world.get_tile(ztx, zty)
                z.vel *= self._tile_slow(z_under) * self._weather_move_mult()
                if abs(float(z.vel.y)) >= abs(float(z.vel.x)):
                    z.dir = "down" if float(z.vel.y) >= 0 else "up"
                else:
                    z.dir = "right" if float(z.vel.x) >= 0 else "left"
                anim_base = 1.8 if kind == "walker" else 2.4 if kind == "screamer" else 3.4
                z.anim = float(z.anim) + dt * float(anim_base) * (z.vel.length() / max(1.0, float(z.speed)))
            else:
                z.anim = float(z.anim) * 0.88

            z.pos = self._move_box(z.pos, z.vel, dt, w=z.w, h=z.h)

            if self.mount != "rv" and float(z.attack_left) <= 0.0 and (z.pos - self.player.pos).length_squared() <= float(mdef.attack_range) ** 2:
                z.attack_left = float(mdef.attack_cd)
                self.player.hp = max(0, int(self.player.hp) - int(mdef.dmg))
                self.player.morale = float(clamp(self.player.morale - (2.0 + float(mdef.dmg) * 0.35), 0.0, 100.0))

    def _season_index(self) -> int:
        day = int(self.world_time_s / max(1e-6, self.DAY_LENGTH_S)) + 1
        return ((day - 1) // max(1, int(self.SEASON_LENGTH_DAYS))) % len(self.SEASONS)

    def _tint(self, c: tuple[int, int, int], *, add: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
        r = int(clamp(c[0] + add[0], 0, 255))
        g = int(clamp(c[1] + add[1], 0, 255))
        b = int(clamp(c[2] + add[2], 0, 255))
        return r, g, b

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        if edge0 == edge1:
            return 0.0
        t = float(clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _time_of_day(self) -> float:
        return (float(self.world_time_s) % float(self.DAY_LENGTH_S)) / max(
            1e-6,
            float(self.DAY_LENGTH_S),
        )

    def _daylight_amount(self) -> tuple[float, float]:
        tday = float(self._time_of_day())
        dawn_start, dawn_end = 0.22, 0.28
        dusk_start, dusk_end = 0.72, 0.78

        if tday < dawn_start or tday >= dusk_end:
            return 0.0, tday
        if tday < dawn_end:
            return float(self._smoothstep(dawn_start, dawn_end, tday)), tday
        if tday < dusk_start:
            return 1.0, tday
        return float(1.0 - self._smoothstep(dusk_start, dusk_end, tday)), tday

    def _draw_day_night_overlay(self, surface: pygame.Surface, *, in_rv: bool) -> None:
        daylight, tday = self._daylight_amount()
        max_alpha = 160 if not in_rv else 120
        a = int(round(float(max_alpha) * (1.0 - float(daylight))))
        if a <= 0:
            return

        if 0.0 < daylight < 1.0:
            col = (36, 26, 64) if tday < 0.5 else (64, 34, 26)  # dawn / dusk
        else:
            col = (12, 14, 30)  # night tint

        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((int(col[0]), int(col[1]), int(col[2]), int(a)))
        surface.blit(overlay, (0, 0))

    def _season_index_for_day(self, day: int) -> int:
        return ((int(day) - 1) // max(1, int(self.SEASON_LENGTH_DAYS))) % len(self.SEASONS)

    @staticmethod
    def _pick_weighted_key(rng: random.Random, weights: dict[str, float], order: Sequence[str], default: str) -> str:
        total = 0.0
        for v in weights.values():
            total += max(0.0, float(v))
        if total <= 0.0001:
            return default
        r = rng.random() * total
        acc = 0.0
        for k in order:
            acc += max(0.0, float(weights.get(k, 0.0)))
            if r <= acc:
                return str(k)
        return default

    def _weather_for_day(self, day: int) -> tuple[str, float, float]:
        day = max(1, int(day))
        season_idx = int(self._season_index_for_day(day))
        seed = int(self._hash2_u32(day, season_idx, int(self.seed) ^ 0x2D5A61B9))
        rng = random.Random(seed)

        if season_idx == 0:  # spring
            weights = {"clear": 0.46, "cloudy": 0.30, "rain": 0.20, "storm": 0.04, "snow": 0.0}
        elif season_idx == 1:  # summer
            weights = {"clear": 0.55, "cloudy": 0.20, "rain": 0.16, "storm": 0.09, "snow": 0.0}
        elif season_idx == 2:  # autumn
            weights = {"clear": 0.38, "cloudy": 0.28, "rain": 0.22, "storm": 0.12, "snow": 0.0}
        else:  # winter
            weights = {"clear": 0.36, "cloudy": 0.30, "rain": 0.06, "storm": 0.08, "snow": 0.20}

        kind = self._pick_weighted_key(rng, weights, self.WEATHER_KINDS, "clear")
        if kind == "clear":
            intensity = 0.0
        elif kind == "cloudy":
            intensity = float(rng.uniform(0.30, 0.70))
        elif kind == "rain":
            intensity = float(rng.uniform(0.35, 0.80))
        elif kind == "storm":
            intensity = float(rng.uniform(0.75, 1.0))
        else:  # snow
            intensity = float(rng.uniform(0.35, 0.95))

        wind_scale = 14.0
        if kind in ("rain", "snow"):
            wind_scale = 28.0
        if kind == "storm":
            wind_scale = 44.0
        wind = float(rng.uniform(-1.0, 1.0) * wind_scale)
        return str(kind), float(intensity), float(wind)

    def _set_weather_targets_for_day(self, day: int) -> None:
        kind, intensity, wind = self._weather_for_day(day)
        self.weather_target_kind = str(kind)
        self.weather_target_intensity = float(clamp(float(intensity), 0.0, 1.0))
        self.weather_wind_target = float(clamp(float(wind), -60.0, 60.0))

    def _sync_weather_particles(self) -> None:
        kind = str(getattr(self, "weather_kind", "clear"))
        intensity = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        rng = getattr(self, "weather_rng", None) or random.Random(int(self.seed) ^ 0xA53C9F17)
        self.weather_rng = rng

        if kind in ("rain", "storm"):
            base = 70 if kind == "rain" else 110
            target = int(base + 120 * intensity)
            while len(self._rain_drops) < target:
                x = float(rng.uniform(-20.0, float(INTERNAL_W) + 20.0))
                y = float(rng.uniform(-float(INTERNAL_H), float(INTERNAL_H)))
                spd = float(rng.uniform(320.0, 620.0) * (0.65 + 0.45 * intensity))
                self._rain_drops.append([x, y, spd])
            if len(self._rain_drops) > target:
                self._rain_drops = self._rain_drops[:target]
        else:
            self._rain_drops.clear()

        if kind == "snow":
            target = int(40 + 90 * intensity)
            while len(self._snow_flakes) < target:
                x = float(rng.uniform(-10.0, float(INTERNAL_W) + 10.0))
                y = float(rng.uniform(-float(INTERNAL_H), float(INTERNAL_H)))
                spd = float(rng.uniform(18.0, 46.0) * (0.60 + 0.55 * intensity))
                drift = float(rng.uniform(-16.0, 16.0))
                self._snow_flakes.append([x, y, spd, drift])
            if len(self._snow_flakes) > target:
                self._snow_flakes = self._snow_flakes[:target]
        else:
            self._snow_flakes.clear()

    def _update_weather(self, dt: float) -> None:
        dt = float(max(0.0, dt))
        day = int(self.world_time_s / max(1e-6, float(self.DAY_LENGTH_S))) + 1
        if int(day) != int(getattr(self, "weather_day", 0)):
            self.weather_day = int(day)
            self._set_weather_targets_for_day(int(day))

        # Smooth wind.
        wind = float(getattr(self, "weather_wind", 0.0))
        wind_t = float(getattr(self, "weather_wind_target", 0.0))
        wind_step = 18.0 * dt
        if wind < wind_t:
            wind = min(wind_t, wind + wind_step)
        else:
            wind = max(wind_t, wind - wind_step)
        self.weather_wind = float(wind)

        kind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        target_kind = str(getattr(self, "weather_target_kind", kind))
        target_inten = float(clamp(float(getattr(self, "weather_target_intensity", inten)), 0.0, 1.0))

        step = dt / max(1e-6, float(self.WEATHER_TRANSITION_S))
        step = float(clamp(step, 0.0, 1.0))

        if kind != target_kind:
            inten = max(0.0, inten - step)
            if inten <= 0.02:
                kind = target_kind
        else:
            if inten < target_inten:
                inten = min(target_inten, inten + step)
            else:
                inten = max(target_inten, inten - step)

        self.weather_kind = str(kind)
        self.weather_intensity = float(inten)

        # Lightning (storm only).
        flash_left = float(getattr(self, "weather_flash_left", 0.0))
        if flash_left > 0.0:
            flash_left = max(0.0, flash_left - dt)
        elif kind == "storm" and inten > 0.55:
            rng = getattr(self, "weather_rng", None) or random.Random(int(self.seed) ^ 0xA53C9F17)
            if rng.random() < dt * (0.06 + 0.18 * inten):
                flash_left = 0.12
        self.weather_flash_left = float(flash_left)

        # Update particles positions.
        self._sync_weather_particles()
        if kind in ("rain", "storm") and self._rain_drops:
            dx = float(self.weather_wind) * 0.25
            for drop in self._rain_drops:
                drop[0] += dx * dt
                drop[1] += float(drop[2]) * dt
                if drop[1] > float(INTERNAL_H) + 16.0 or drop[0] < -40.0 or drop[0] > float(INTERNAL_W) + 40.0:
                    drop[0] = float(self.weather_rng.uniform(-20.0, float(INTERNAL_W) + 20.0))
                    drop[1] = float(self.weather_rng.uniform(-18.0, -2.0))
                    drop[2] = float(self.weather_rng.uniform(320.0, 620.0) * (0.65 + 0.45 * inten))

        if kind == "snow" and self._snow_flakes:
            dx = float(self.weather_wind) * 0.20
            for flake in self._snow_flakes:
                flake[0] += (dx + float(flake[3])) * dt
                flake[1] += float(flake[2]) * dt
                if flake[1] > float(INTERNAL_H) + 12.0:
                    flake[0] = float(self.weather_rng.uniform(-10.0, float(INTERNAL_W) + 10.0))
                    flake[1] = float(self.weather_rng.uniform(-16.0, -2.0))
                if flake[0] < -16.0:
                    flake[0] = float(INTERNAL_W) + 12.0
                elif flake[0] > float(INTERNAL_W) + 16.0:
                    flake[0] = -12.0

    def _weather_move_mult(self) -> float:
        kind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        if kind == "rain":
            return float(1.0 - 0.10 * inten)
        if kind == "storm":
            return float(1.0 - 0.16 * inten)
        if kind == "snow":
            return float(1.0 - 0.12 * inten)
        return 1.0

    def _draw_weather_effects(self, surface: pygame.Surface, *, in_rv: bool) -> None:
        kind = str(getattr(self, "weather_kind", "clear"))
        inten = float(clamp(float(getattr(self, "weather_intensity", 0.0)), 0.0, 1.0))
        if inten <= 0.01:
            return
        layer: pygame.Surface = getattr(self, "_weather_layer", None) or pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        self._weather_layer = layer
        layer.fill((0, 0, 0, 0))

        daylight, _tday = self._daylight_amount()
        drop_col = (200, 200, 216) if daylight >= 0.55 else (140, 140, 155)
        tint_mult = 0.55 if in_rv else 1.0

        rv_clips: list[pygame.Rect] | None = None
        if in_rv:
            tile = int(getattr(self, "_RV_INT_TILE_SIZE", 20))
            map_w = int(getattr(self, "_RV_INT_W", 21)) * tile
            map_h = int(getattr(self, "_RV_INT_H", 7)) * tile
            map_x = (INTERNAL_W - map_w) // 2
            map_y = 92
            panel = pygame.Rect(map_x - 10, map_y - 10, map_w + 20, map_h + 20) 
            rv_clips = []
            if panel.top > 0:
                rv_clips.append(pygame.Rect(0, 0, INTERNAL_W, panel.top))       
            if panel.bottom < INTERNAL_H:
                rv_clips.append(pygame.Rect(0, panel.bottom, INTERNAL_W, INTERNAL_H - panel.bottom))
            if panel.left > 0:
                rv_clips.append(pygame.Rect(0, panel.top, panel.left, panel.h)) 
            if panel.right < INTERNAL_W:
                rv_clips.append(pygame.Rect(panel.right, panel.top, INTERNAL_W - panel.right, panel.h))

            # Let precipitation be visible through RV windows (clipped to glass).
            layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)
            if isinstance(layout, list):
                for y, row in enumerate(layout):
                    for x in range(min(len(row), int(getattr(self, "_RV_INT_W", 21)))):
                        if row[x] != "V":
                            continue
                        tr = pygame.Rect(map_x + x * tile, map_y + y * tile, tile, tile)
                        glass = pygame.Rect(tr.x + 3, tr.y + 4, tr.w - 6, tr.h - 8)
                        if glass.w > 0 and glass.h > 0:
                            rv_clips.append(glass)

        # Base tint per weather.
        if kind == "cloudy":
            a = int(round(40.0 * inten * tint_mult))
            pygame.draw.rect(layer, (40, 44, 52, a), layer.get_rect())
        elif kind == "rain":
            a = int(round(55.0 * inten * tint_mult))
            pygame.draw.rect(layer, (32, 40, 58, a), layer.get_rect())   
        elif kind == "storm":
            a = int(round(85.0 * inten * tint_mult))
            pygame.draw.rect(layer, (22, 26, 44, a), layer.get_rect())   
        elif kind == "snow":
            a = int(round(32.0 * inten * tint_mult))
            pygame.draw.rect(layer, (180, 190, 210, a), layer.get_rect())

        # Precipitation.
        if kind in ("rain", "storm") and self._rain_drops:
            alpha = int(round((120 if kind == "rain" else 160) * inten))
            alpha = int(clamp(alpha, 20, 210))
            dx = int(round(clamp(float(self.weather_wind) * 0.05, -4.0, 4.0)))
            ln = 6 + int(round(4.0 * inten))
            if rv_clips:
                for cr in rv_clips:
                    layer.set_clip(cr)
                    for x, y, _spd in self._rain_drops:
                        xi = int(x)
                        yi = int(y)
                        pygame.draw.line(layer, (drop_col[0], drop_col[1], drop_col[2], alpha), (xi, yi), (xi + dx, yi + ln), 1)
                layer.set_clip(None)
            else:
                for x, y, _spd in self._rain_drops:
                    xi = int(x)
                    yi = int(y)
                    pygame.draw.line(layer, (drop_col[0], drop_col[1], drop_col[2], alpha), (xi, yi), (xi + dx, yi + ln), 1)
        elif kind == "snow" and self._snow_flakes:
            alpha = int(round(170.0 * inten))
            alpha = int(clamp(alpha, 30, 230))
            if rv_clips:
                for cr in rv_clips:
                    layer.set_clip(cr)
                    for x, y, _spd, _drift in self._snow_flakes:
                        xi = int(x)
                        yi = int(y)
                        if 0 <= xi < INTERNAL_W and 0 <= yi < INTERNAL_H:
                            layer.fill((235, 235, 245, alpha), pygame.Rect(xi, yi, 1, 1))
                            if (xi + yi) % 7 == 0:
                                layer.fill((235, 235, 245, max(20, alpha - 60)), pygame.Rect(xi, yi + 1, 1, 1))
                layer.set_clip(None)
            else:
                for x, y, _spd, _drift in self._snow_flakes:
                    xi = int(x)
                    yi = int(y)
                    if 0 <= xi < INTERNAL_W and 0 <= yi < INTERNAL_H:
                        layer.fill((235, 235, 245, alpha), pygame.Rect(xi, yi, 1, 1))
                        if (xi + yi) % 7 == 0:
                            layer.fill((235, 235, 245, max(20, alpha - 60)), pygame.Rect(xi, yi + 1, 1, 1))

        # Lightning flash overlay.
        flash_left = float(getattr(self, "weather_flash_left", 0.0))
        if kind == "storm" and flash_left > 0.0:
            p = float(clamp(flash_left / 0.12, 0.0, 1.0))
            a = int(round(210.0 * p))
            pygame.draw.rect(layer, (255, 255, 255, a), layer.get_rect())

        surface.blit(layer, (0, 0))

    def _tile_color(self, tile_id: int) -> tuple[int, int, int]:
        t = self._TILES.get(int(tile_id))
        base = t.color if t is not None else (255, 0, 255)
        season = self._season_index()
        if tile_id in (self.T_GRASS, self.T_FOREST):
            if season == 0:  # spring
                return self._tint(base, add=(4, 10, 4))
            if season == 1:  # summer
                return self._tint(base, add=(0, 0, 0))
            if season == 2:  # autumn
                return self._tint(base, add=(18, 8, -10))
            return self._tint(base, add=(28, 28, 28))  # winter
        if tile_id == self.T_WATER and season == 3:
            return self._tint(base, add=(10, 10, 22))
        return base

    def _minimap_color(self, tile_id: int) -> tuple[int, int, int]:
        # Minimap / world map palette.
        # Do NOT reveal interior layouts: all building-interior tiles (walls/floors/doors/furniture)
        # are rendered as a solid "roof" blob.
        tile_id = int(tile_id)
        if tile_id in (
            int(self.T_WALL),
            int(self.T_FLOOR),
            int(self.T_DOOR),
            int(self.T_TABLE),
            int(self.T_SHELF),
            int(self.T_BED),
        ):
            return (72, 72, 92)
        if tile_id == int(self.T_ROAD):
            return (92, 92, 96)
        if tile_id == int(self.T_HIGHWAY):
            return (82, 82, 88)
        if tile_id == int(self.T_PAVEMENT):
            return (72, 72, 76)
        if tile_id == int(self.T_PARKING):
            return (62, 62, 70)
        if tile_id == int(self.T_COURT):
            return (126, 92, 70)
        if tile_id == int(self.T_SIDEWALK):
            return (174, 154, 104)
        if tile_id == int(self.T_BRICK):
            return (166, 112, 96)
        if tile_id == int(self.T_CONCRETE):
            return (118, 118, 128)
        if tile_id == int(self.T_SAND):
            return (194, 176, 112)
        if tile_id == int(self.T_BOARDWALK):
            return (156, 116, 68)
        if tile_id == int(self.T_MARSH):
            return (62, 98, 66)
        return self._tile_color(tile_id)

    def _draw_world_tile(self, surface: pygame.Surface, rect: pygame.Rect, *, tx: int, ty: int, tile_id: int) -> None:
        tx = int(tx)
        ty = int(ty)
        tile_id = int(tile_id)
        col = self._tile_color(tile_id)
        surface.fill(col, rect)

        seed = int(self.seed) ^ (tile_id * 0x9E3779B9) ^ 0xB5297A4D
        h = int(self._hash2_u32(tx, ty, seed))

        # Simple gritty texture pass (cheap pixel-noise), inspired by the reference tone.
        hard = tile_id in (
            self.T_ROAD,
            self.T_HIGHWAY,
            self.T_PAVEMENT,
            self.T_PARKING,
            self.T_COURT,
            self.T_FLOOR,
            self.T_SIDEWALK,
            self.T_BRICK,
            self.T_CONCRETE,
            self.T_BOARDWALK,
        )
        soft = tile_id in (self.T_GRASS, self.T_FOREST, self.T_WATER, self.T_SAND, self.T_MARSH)

        if hard or soft:
            for j in range(2):
                if ((h >> (j * 3)) & 7) != 0:
                    continue
                ox = int((h >> (8 + j * 6)) % max(1, int(self.TILE_SIZE)))
                oy = int((h >> (12 + j * 6)) % max(1, int(self.TILE_SIZE)))
                dv = -14 if ((h >> (20 + j)) & 1) else 12
                if tile_id in (self.T_GRASS, self.T_FOREST):
                    add = (dv // 4, dv, dv // 4)
                elif tile_id == self.T_WATER:
                    add = (dv // 5, dv // 5, dv)
                else:
                    add = (dv, dv, dv)
                surface.fill(self._tint(col, add=add), pygame.Rect(rect.x + ox, rect.y + oy, 1, 1))

        if tile_id == self.T_SIDEWALK:
            grout = self._tint(col, add=(-22, -18, -10))
            if (tx % 2) == 0:
                surface.fill(grout, pygame.Rect(rect.x, rect.y + 1, 1, rect.h - 2))
            if (ty % 2) == 0:
                surface.fill(grout, pygame.Rect(rect.x + 1, rect.y, rect.w - 2, 1))

        if tile_id == self.T_BRICK:
            mortar = self._tint(col, add=(-30, -22, -20))
            surface.fill(mortar, pygame.Rect(rect.x + 1, rect.y, rect.w - 2, 1))
            surface.fill(mortar, pygame.Rect(rect.x + 1, rect.y + rect.h // 2, rect.w - 2, 1))
            for row in range(2):
                y0 = rect.y + row * (rect.h // 2) + 1
                hgt = (rect.h // 2) - 1
                shift = 0 if ((tx + ty + row) % 2) == 0 else 2
                for x_off in (shift, (shift + 5) % self.TILE_SIZE):
                    surface.fill(mortar, pygame.Rect(rect.x + x_off, y0, 1, hgt))

        if tile_id == self.T_CONCRETE:
            seam = self._tint(col, add=(-18, -18, -22))
            if (tx % 4) == 0:
                surface.fill(seam, pygame.Rect(rect.x, rect.y + 1, 1, rect.h - 2))
            if (ty % 4) == 0:
                surface.fill(seam, pygame.Rect(rect.x + 1, rect.y, rect.w - 2, 1))

        if tile_id == self.T_SAND:
            # Beach speckles + dune hints.
            grain = self._tint(col, add=(18, 14, -8))
            if ((h >> 3) & 3) == 0:
                surface.fill(grain, pygame.Rect(rect.x + 2, rect.y + 2, 1, 1))
            if ((h >> 6) & 3) == 0:
                surface.fill(grain, pygame.Rect(rect.right - 3, rect.y + 3, 1, 1))
            if (ty % 5) == 0:
                dune = self._tint(col, add=(-14, -10, -6))
                surface.fill(dune, pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, 1))

        if tile_id == self.T_MARSH:
            # Mud + shallow puddles.
            mud = self._tint(col, add=(-16, -10, -16))
            if ((h >> 4) & 7) == 0:
                surface.fill(mud, pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, 1))
            if ((h >> 7) & 7) == 0:
                water = (40, 70, 96)
                surface.fill(water, pygame.Rect(rect.x + 2, rect.y + 3, 2, 1))

        if tile_id == self.T_BOARDWALK:
            # Wood planks.
            plank = self._tint(col, add=(-24, -18, -10))
            if (ty % 2) == 0:
                surface.fill(plank, pygame.Rect(rect.x + 1, rect.y + 2, rect.w - 2, 1))
            if (tx % 4) == 0:
                seam = self._tint(col, add=(-34, -26, -18))
                surface.fill(seam, pygame.Rect(rect.x + 1, rect.y + 1, 1, rect.h - 2))
            if ((h >> 5) & 31) == 0:
                nail = self._tint(col, add=(26, 18, 10))
                surface.fill(nail, pygame.Rect(rect.centerx, rect.centery, 1, 1))

        if tile_id == self.T_PARKING:
            # Painted stall lines + curb edge so parking lots read clearly.
            line = self._tint(col, add=(86, 86, 92))
            line2 = self._tint(col, add=(58, 58, 64))
            if (tx % 3) == 0:
                surface.fill(line, pygame.Rect(rect.x + 1, rect.y + 1, 1, rect.h - 2))
            if (ty % 6) == 0:
                surface.fill(line, pygame.Rect(rect.x + 1, rect.y + 2, rect.w - 2, 1))
            # Occasional wheel-stop / marking block.
            if ((tx + ty + (h & 7)) % 17) == 0:
                surface.fill(line2, pygame.Rect(rect.centerx - 2, rect.centery + 2, 4, 1))

            curb = self._tint(col, add=(-34, -34, -34))
            if int(self.world.peek_tile(tx, ty - 1)) != int(self.T_PARKING):
                surface.fill(curb, pygame.Rect(rect.x, rect.y, rect.w, 1))
            if int(self.world.peek_tile(tx, ty + 1)) != int(self.T_PARKING):
                surface.fill(curb, pygame.Rect(rect.x, rect.bottom - 1, rect.w, 1))
            if int(self.world.peek_tile(tx - 1, ty)) != int(self.T_PARKING):
                surface.fill(curb, pygame.Rect(rect.x, rect.y, 1, rect.h))
            if int(self.world.peek_tile(tx + 1, ty)) != int(self.T_PARKING):
                surface.fill(curb, pygame.Rect(rect.right - 1, rect.y, 1, rect.h))

        if tile_id == self.T_HIGHWAY:
            # Abandoned highway: lane marks + cracks.
            mark = self._tint(col, add=(44, 44, 50))
            left = int(self.world.peek_tile(tx - 1, ty)) == int(self.T_HIGHWAY)
            right = int(self.world.peek_tile(tx + 1, ty)) == int(self.T_HIGHWAY)
            up = int(self.world.peek_tile(tx, ty - 1)) == int(self.T_HIGHWAY)
            down = int(self.world.peek_tile(tx, ty + 1)) == int(self.T_HIGHWAY)
            horiz = (left or right) and not (up or down)
            vert = (up or down) and not (left or right)
            if horiz and (tx % 6) == 0:
                surface.fill((210, 190, 110), pygame.Rect(rect.centerx - 2, rect.centery, 4, 1))
            elif vert and (ty % 6) == 0:
                surface.fill((210, 190, 110), pygame.Rect(rect.centerx, rect.centery - 2, 1, 4))
            # Shoulder edges.
            edge = self._tint(col, add=(32, 32, 36))
            if horiz:
                surface.fill(edge, pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, 1))
                surface.fill(edge, pygame.Rect(rect.x + 1, rect.bottom - 2, rect.w - 2, 1))
            elif vert:
                surface.fill(edge, pygame.Rect(rect.x + 1, rect.y + 1, 1, rect.h - 2))
                surface.fill(edge, pygame.Rect(rect.right - 2, rect.y + 1, 1, rect.h - 2))
            # Cracks.
            if (h & 31) == 0:
                crack = self._tint(col, add=(-30, -30, -30))
                pygame.draw.line(surface, crack, (rect.x + 2, rect.y + 7), (rect.right - 3, rect.y + 3), 1)

        if tile_id == self.T_ROAD:
            mark = self._tint(col, add=(32, 32, 36))
            left = int(self.world.peek_tile(tx - 1, ty)) == int(self.T_ROAD)
            right = int(self.world.peek_tile(tx + 1, ty)) == int(self.T_ROAD)
            up = int(self.world.peek_tile(tx, ty - 1)) == int(self.T_ROAD)
            down = int(self.world.peek_tile(tx, ty + 1)) == int(self.T_ROAD)
            horiz = (left or right) and not (up or down)
            vert = (up or down) and not (left or right)
            if horiz and (tx % 4) == 0:
                surface.fill(mark, pygame.Rect(rect.centerx - 2, rect.centery, 4, 1))
            elif vert and (ty % 4) == 0:
                surface.fill(mark, pygame.Rect(rect.centerx, rect.centery - 2, 1, 4))

        if tile_id in (self.T_PAVEMENT, self.T_SIDEWALK, self.T_BRICK, self.T_CONCRETE, self.T_BOARDWALK):
            curb = self._tint(col, add=(-26, -26, -26))
            if int(self.world.peek_tile(tx, ty - 1)) in (int(self.T_ROAD), int(self.T_HIGHWAY)):
                surface.fill(curb, pygame.Rect(rect.x, rect.y, rect.w, 1))
            if int(self.world.peek_tile(tx - 1, ty)) in (int(self.T_ROAD), int(self.T_HIGHWAY)):
                surface.fill(curb, pygame.Rect(rect.x, rect.y, 1, rect.h))
            curb_hi = self._tint(col, add=(16, 16, 16))
            if int(self.world.peek_tile(tx, ty + 1)) in (int(self.T_ROAD), int(self.T_HIGHWAY)):
                surface.fill(curb_hi, pygame.Rect(rect.x, rect.bottom - 1, rect.w, 1))
            if int(self.world.peek_tile(tx + 1, ty)) in (int(self.T_ROAD), int(self.T_HIGHWAY)):
                surface.fill(curb_hi, pygame.Rect(rect.right - 1, rect.y, 1, rect.h))

        if tile_id == self.T_FLOOR:
            shadow = self._tint(col, add=(-18, -18, -18))
            if int(self.world.peek_tile(tx, ty - 1)) == int(self.T_WALL):
                surface.fill(shadow, pygame.Rect(rect.x, rect.y, rect.w, 1))
            if int(self.world.peek_tile(tx - 1, ty)) == int(self.T_WALL):
                surface.fill(shadow, pygame.Rect(rect.x, rect.y, 1, rect.h))

    def _blit_sprite_outline(self, surface: pygame.Surface, spr: pygame.Surface, rect: pygame.Rect, *, color: tuple[int, int, int]) -> None:
        r, g, b = (int(color[0]), int(color[1]), int(color[2]))
        key = (id(spr), int(r), int(g), int(b))
        cache: dict[tuple[int, int, int, int], pygame.Surface] = getattr(self, "_sprite_outline_cache", {})
        out = cache.get(key)
        if out is None:
            mask = pygame.mask.from_surface(spr)
            shape = mask.to_surface(setcolor=(r, g, b, 255), unsetcolor=(0, 0, 0, 0))
            out = pygame.Surface((spr.get_width() + 2, spr.get_height() + 2), pygame.SRCALPHA)
            # 1px outline by stamping the silhouette around the sprite.
            out.blit(shape, (0, 1))
            out.blit(shape, (2, 1))
            out.blit(shape, (1, 0))
            out.blit(shape, (1, 2))
            cache[key] = out
            self._sprite_outline_cache = cache
        surface.blit(out, (int(rect.left) - 1, int(rect.top) - 1))
        surface.blit(spr, rect)

    def _draw_hover_tooltip(self, surface: pygame.Surface) -> None:
        tip = getattr(self, "_hover_tooltip", None)
        if tip is None:
            return
        lines, pos = tip
        if not lines:
            return
        mx, my = int(pos[0]), int(pos[1])

        font = self.app.font_s
        pad = 6
        line_h = max(10, int(font.get_height()))
        w = 0
        for ln in lines:
            w = max(w, int(font.size(str(ln))[0]))
        w = int(w + pad * 2)
        h = int(len(lines) * line_h + pad * 2)

        x = mx + 12
        y = my + 12
        if x + w > INTERNAL_W - 4:
            x = max(4, INTERNAL_W - 4 - w)
        if y + h > INTERNAL_H - 4:
            y = max(4, INTERNAL_H - 4 - h)

        panel = pygame.Rect(int(x), int(y), int(w), int(h))
        ui = pygame.Surface((panel.w, panel.h), pygame.SRCALPHA)
        ui.fill((10, 10, 14, 220))
        pygame.draw.rect(ui, (90, 90, 110, 235), ui.get_rect(), 1, border_radius=8)
        surface.blit(ui, panel.topleft)

        for i, ln in enumerate(lines):
            col = pygame.Color(240, 240, 244) if i == 0 else pygame.Color(180, 180, 190)
            draw_text(surface, font, str(ln), (panel.left + pad, panel.top + pad + i * line_h), col, anchor="topleft")

    def _draw_vehicle_props(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
    ) -> None:
        start_cx = start_tx // self.CHUNK_SIZE
        end_cx = end_tx // self.CHUNK_SIZE
        start_cy = start_ty // self.CHUNK_SIZE
        end_cy = end_ty // self.CHUNK_SIZE

        hovered = None  # (d2, spr, rect, lines, (mx,my))
        mouse = None
        if (
            not self.inv_open
            and not bool(getattr(self, "rv_ui_open", False))
            and not bool(getattr(self, "world_map_open", False))
        ):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue

                for car in getattr(chunk, "cars", []):
                    p = pygame.Vector2(getattr(car, "pos", pygame.Vector2(0, 0))) - pygame.Vector2(cam_x, cam_y)
                    sx = iround(float(p.x))
                    sy = iround(float(p.y))
                    if sx < -80 or sx > INTERNAL_W + 80 or sy < -80 or sy > INTERNAL_H + 80:
                        continue
                    mid = str(getattr(car, "model_id", "schoolbus"))
                    steer_state = int(getattr(car, "steer_state", 0))
                    frame = int(getattr(car, "frame", 0)) % 2
                    base = self._CAR_BASE.get((mid, int(steer_state), int(frame)))
                    shadow_base = self._CAR_SHADOW.get((mid, int(steer_state), int(frame)))
                    if base is None:
                        base = self._CAR_BASE.get(("schoolbus", 0, 0))
                        shadow_base = self._CAR_SHADOW.get(("schoolbus", 0, 0))
                        if base is None:
                            continue
                    deg = -math.degrees(float(getattr(car, "heading", 0.0)))
                    spr = rotate_pixel_sprite(base, deg, step_deg=5.0)
                    rect = spr.get_rect(center=(sx, sy))
                    if shadow_base is not None:
                        sh = rotate_pixel_sprite(shadow_base, deg, step_deg=5.0)
                        srect = sh.get_rect(center=(rect.centerx + 2, rect.centery + 6))
                        surface.blit(sh, srect)
                    surface.blit(spr, rect)
                    if mouse is not None:
                        mx, my = int(mouse[0]), int(mouse[1])
                        if rect.collidepoint(mx, my):
                            model = self._CAR_MODELS.get(str(mid))
                            name = model.name if model is not None else str(mid)
                            lines = [f"{name}", "类型: 载具(停放)"]
                            d2 = float((mx - sx) * (mx - sx) + (my - sy) * (my - sy))
                            if hovered is None or d2 < float(hovered[0]):
                                hovered = (d2, spr, rect.copy(), lines, (mx, my))

                for b in getattr(chunk, "bikes", []):
                    p = pygame.Vector2(getattr(b, "pos", pygame.Vector2(0, 0))) - pygame.Vector2(cam_x, cam_y)
                    sx = iround(float(p.x))
                    sy = iround(float(p.y))
                    if sx < -40 or sx > INTERNAL_W + 40 or sy < -40 or sy > INTERNAL_H + 40:
                        continue
                    mid = str(getattr(b, "model_id", "bike"))
                    d = str(getattr(b, "dir", "right"))
                    frm = int(getattr(b, "frame", 0)) % 2
                    model = self._TWO_WHEEL_FRAMES.get(mid) or self._TWO_WHEEL_FRAMES.get("bike") or {}
                    frames = model.get(d) or model.get("right") or self._BIKE_FRAMES.get("right", [])
                    if not frames:
                        continue
                    spr = frames[min(int(frm), len(frames) - 1)]
                    _cw, ch = self._two_wheel_collider_px(mid)
                    ground_y = iround(float(sy) + float(ch) / 2.0)
                    rect = spr.get_rect()
                    baseline_y = int(self._sprite_baseline_y(spr))
                    rect.centerx = int(sx)
                    rect.bottom = iround(float(ground_y) + float(rect.h - baseline_y))
                    shadow = self._two_wheel_shadow_rect(rect, d, ground_y=int(ground_y))
                    sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
                    pygame.draw.ellipse(sh, (0, 0, 0, 120), sh.get_rect())
                    surface.blit(sh, shadow.topleft)
                    surface.blit(spr, rect)
                    if mouse is not None:
                        mx, my = int(mouse[0]), int(mouse[1])
                        if rect.collidepoint(mx, my):
                            name = self._two_wheel_name(str(mid))
                            wpos = pygame.Vector2(getattr(b, "pos", pygame.Vector2(0, 0)))
                            near = (self.player.pos - wpos).length_squared() <= (18.0 * 18.0)
                            if self.mount is None:
                                prompt = "F 骑车" if near else "靠近后: F 骑车"
                            else:
                                prompt = "先下车再换车"
                            lines = [f"{name}", "类型: 载具(停放)", prompt]
                            if str(mid).startswith("moto"):
                                lines.insert(1, f"燃油: {int(getattr(b, 'fuel', 0.0))}")
                            d2 = float((mx - sx) * (mx - sx) + (my - sy) * (my - sy))
                            if hovered is None or d2 < float(hovered[0]):
                                hovered = (d2, spr, rect.copy(), lines, (mx, my))

        if hovered is None or mouse is None:
            return
        _d2, spr, rect, lines, mpos = hovered
        hi = (255, 245, 140)
        self._blit_sprite_outline(surface, spr, rect, color=hi)
        self._hover_tooltip = (list(lines), (int(mpos[0]), int(mpos[1])))

    def _draw_world_props(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
    ) -> None:
        # Mouse hover: outline + tooltip.
        hovered = None  # (d2, prop, sprite|None, rect, sx, sy)
        mouse = None
        if not self.inv_open and not bool(getattr(self, "rv_ui_open", False)) and not bool(
            getattr(self, "world_map_open", False)
        ):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())

        start_cx = start_tx // self.CHUNK_SIZE
        end_cx = end_tx // self.CHUNK_SIZE
        start_cy = start_ty // self.CHUNK_SIZE
        end_cy = end_ty // self.CHUNK_SIZE

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue
                for pr in getattr(chunk, "props", []):
                    sx = int(round(pr.pos.x - cam_x))
                    sy = int(round(pr.pos.y - cam_y))
                    if sx < -64 or sx > INTERNAL_W + 64 or sy < -64 or sy > INTERNAL_H + 64:
                        continue

                    pdef = self._PROP_DEFS.get(str(getattr(pr, "prop_id", "")))
                    spr = None
                    if pdef is not None:
                        sprites = getattr(pdef, "sprites", ())
                        if sprites:
                            idx = int(getattr(pr, "variant", 0)) % len(sprites)
                            spr = sprites[int(idx)]

                    if spr is not None:
                        rect = spr.get_rect(center=(sx, sy))
                        sw = max(4, int(rect.w) - 6)
                        shadow = pygame.Rect(0, 0, sw, 4)
                        shadow.center = (rect.centerx, rect.bottom - 1)
                        sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
                        pygame.draw.ellipse(sh, (0, 0, 0, 100), sh.get_rect())
                        surface.blit(sh, shadow.topleft)
                        surface.blit(spr, rect)
                    else:
                        rect = pygame.Rect(sx - 4, sy - 4, 8, 8)
                        pygame.draw.rect(surface, (0, 0, 0), rect)
                        pygame.draw.rect(surface, (210, 80, 220), rect.inflate(-2, -2))

                    if mouse is not None:
                        mx, my = int(mouse[0]), int(mouse[1])
                        if rect.collidepoint(mx, my):
                            d2 = float((mx - sx) * (mx - sx) + (my - sy) * (my - sy))
                            if hovered is None or d2 < float(hovered[0]):
                                hovered = (d2, pr, spr if spr is not None else None, rect.copy(), int(sx), int(sy))

        if hovered is None or mouse is None:
            return

        _d2, pr, spr, rect, _sx, _sy = hovered
        hi = (255, 245, 140)
        if spr is not None:
            self._blit_sprite_outline(surface, spr, rect, color=hi)
        else:
            pygame.draw.rect(surface, hi, rect, 1)

        pdef = self._PROP_DEFS.get(str(getattr(pr, "prop_id", "")))
        name = pdef.name if pdef is not None else str(getattr(pr, "prop_id", "prop"))
        cat = pdef.category if pdef is not None else "场景道具"
        self._hover_tooltip = ([name, f"类型: {cat}"], (int(mouse[0]), int(mouse[1])))

    def _draw_world_items(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
    ) -> None:
        # Mouse hover: outline + tooltip.
        hovered = None  # (d2, item, sprite|None, rect, sx, sy)
        mouse = None
        if not self.inv_open and not bool(getattr(self, "rv_ui_open", False)) and not bool(getattr(self, "world_map_open", False)):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())

        start_cx = start_tx // self.CHUNK_SIZE
        end_cx = end_tx // self.CHUNK_SIZE
        start_cy = start_ty // self.CHUNK_SIZE
        end_cy = end_ty // self.CHUNK_SIZE

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue
                for it in chunk.items:
                    sx = iround(float(it.pos.x) - float(cam_x))
                    sy = iround(float(it.pos.y) - float(cam_y))
                    if sx < -8 or sx > INTERNAL_W + 8 or sy < -8 or sy > INTERNAL_H + 8:
                        continue
                    spr = self._ITEM_SPRITES_WORLD.get(it.item_id) or self._ITEM_SPRITES.get(it.item_id)
                    if spr is not None:
                        rect = spr.get_rect()
                        rect.midbottom = (sx, sy + 6)
                        sw = max(4, rect.w - 8)
                        shadow = pygame.Rect(0, 0, sw, 5)
                        shadow.center = (sx, sy + 4)
                        pygame.draw.ellipse(surface, (0, 0, 0), shadow)  
                        surface.blit(spr, rect)
                    else:
                        icon = self._ITEM_ICONS.get(it.item_id)
                        if icon is None:
                            idef = self._ITEMS.get(it.item_id)
                            col = idef.color if idef is not None else (255, 0, 255)
                            rect = pygame.Rect(sx - 3, sy - 3, 6, 6)
                            pygame.draw.rect(surface, (0, 0, 0), rect)
                            pygame.draw.rect(surface, col, rect.inflate(-2, -2))
                        else:
                            bg = pygame.Rect(sx - 5, sy - 5, 10, 10)
                            pygame.draw.rect(surface, (0, 0, 0), bg, border_radius=3)
                            rect = icon.get_rect(center=(sx, sy))
                            surface.blit(icon, rect)

                    if mouse is not None and rect is not None:
                        mx, my = int(mouse[0]), int(mouse[1])
                        if rect.collidepoint(mx, my):
                            d2 = float((mx - sx) * (mx - sx) + (my - sy) * (my - sy))
                            if hovered is None or d2 < float(hovered[0]):
                                hovered = (d2, it, spr if spr is not None else None, rect.copy(), int(sx), int(sy))

        if hovered is None or mouse is None:
            return

        _d2, it, spr, rect, _sx, _sy = hovered
        hi = (255, 245, 140)
        if spr is not None:
            self._blit_sprite_outline(surface, spr, rect, color=hi)
        else:
            pygame.draw.rect(surface, (0, 0, 0), rect.inflate(6, 6), 1)
            pygame.draw.rect(surface, hi, rect.inflate(4, 4), 1)

        idef = self._ITEMS.get(it.item_id)
        name = idef.name if idef is not None else str(it.item_id)
        qty = int(getattr(it, "qty", 1))
        kind = str(getattr(idef, "kind", "")) if idef is not None else ""
        kind_map = {
            "food": "食物",
            "drink": "饮料",
            "med": "医疗",
            "ammo": "弹药",
            "mat": "材料",
            "gun": "武器",
        }
        klabel = kind_map.get(kind, "物品") if kind else "物品"
        first = f"{name}  x{qty}" if qty > 1 else name
        self._hover_tooltip = ([first, f"类型: {klabel}", "E 拾取"], (int(mouse[0]), int(mouse[1])))

    def _draw_rv(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        rvp = pygame.Vector2(self.rv.pos) - pygame.Vector2(cam_x, cam_y)        
        moving = self.mount == "rv" and abs(float(self.rv.speed)) > 6.0
        frame = int(self.rv_anim) % 2 if moving else 0
        steer_state = 0
        if abs(float(self.rv.steer)) > 0.08:
            steer_state = 1 if float(self.rv.steer) > 0.0 else -1
        mid = getattr(self.rv, "model_id", "schoolbus")
        base = self._CAR_BASE.get((str(mid), int(steer_state), int(frame)))
        shadow_base = self._CAR_SHADOW.get((str(mid), int(steer_state), int(frame)))
        if base is None:
            base = self._CAR_BASE.get(("schoolbus", int(steer_state), int(frame)))
            shadow_base = self._CAR_SHADOW.get(("schoolbus", int(steer_state), int(frame)))
            if base is None:
                return
        deg = -math.degrees(float(self.rv.heading))
        spr = rotate_pixel_sprite(base, deg, step_deg=5.0)

        rect = spr.get_rect(center=(int(round(rvp.x)), int(round(rvp.y))))
        if shadow_base is not None:
            shadow = rotate_pixel_sprite(shadow_base, deg, step_deg=5.0)        
            srect = shadow.get_rect(center=(rect.centerx + 2, rect.centery + 6))
            surface.blit(shadow, srect)
        surface.blit(spr, rect)

        mouse = None
        if (
            not self.inv_open
            and not bool(getattr(self, "rv_ui_open", False))
            and not bool(getattr(self, "world_map_open", False))
        ):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())
        if mouse is not None:
            mx, my = int(mouse[0]), int(mouse[1])
            if rect.collidepoint(mx, my):
                model = self._CAR_MODELS.get(str(mid))
                name = model.name if model is not None else str(mid)
                near = (self.player.pos - self.rv.pos).length_squared() <= (22.0 * 22.0)
                if self.mount == "rv":
                    prompt = "F 下车  |  H 内部"
                else:
                    prompt = "F 上车  |  H 内部" if near else "靠近后: F 上车 / H 内部"
                lines = [f"房车：{name}", f"燃油: {int(self.rv.fuel)}", prompt]
                hi = (255, 245, 140)
                self._blit_sprite_outline(surface, spr, rect, color=hi)
                self._hover_tooltip = (lines, (mx, my))

    def _draw_bike(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        bp = pygame.Vector2(self.bike.pos) - pygame.Vector2(cam_x, cam_y)       
        moving = self.mount == "bike" and self.bike.vel.length_squared() > 0.1
        frame = int(self.bike_anim) % 2 if moving else 0
        d = getattr(self, "bike_dir", "right")
        mid = str(getattr(self.bike, "model_id", "bike"))
        model = self._TWO_WHEEL_FRAMES.get(mid) or self._TWO_WHEEL_FRAMES.get("bike") or {}
        frames = model.get(str(d)) or model.get("right") or self._BIKE_FRAMES.get("right", [])
        if not frames:
            return
        spr = frames[min(int(frame), len(frames) - 1)]
        ground_y = int(round(float(bp.y) + float(getattr(self.bike, "h", 10)) / 2.0))
        rect = spr.get_rect()
        baseline_y = int(self._sprite_baseline_y(spr))
        rect.centerx = int(round(bp.x))
        rect.bottom = int(round(float(ground_y) + float(rect.h - baseline_y)))

        # Soft ground shadow so the two-wheeler doesn't look like it's floating.
        shadow = self._two_wheel_shadow_rect(rect, str(d), ground_y=int(ground_y))
        sh = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)       
        pygame.draw.ellipse(sh, (0, 0, 0, 130), sh.get_rect())
        surface.blit(sh, shadow.topleft)
        surface.blit(spr, rect)

        mouse = None
        if (
            not self.inv_open
            and not bool(getattr(self, "rv_ui_open", False))
            and not bool(getattr(self, "world_map_open", False))
        ):
            mouse = self.app.screen_to_internal(pygame.mouse.get_pos())
        if mouse is not None:
            mx, my = int(mouse[0]), int(mouse[1])
            if rect.collidepoint(mx, my):
                near = (self.player.pos - self.bike.pos).length_squared() <= (18.0 * 18.0)
                if self.mount == "bike":
                    prompt = "F 下车"
                else:
                    prompt = "F 骑车" if near else "靠近后: F 骑车"
                lines = [self._two_wheel_name(mid), "类型: 载具", prompt]
                if str(mid).startswith("moto"):
                    lines.insert(1, f"燃油: {int(getattr(self.bike, 'fuel', 0.0))}")
                hi = (255, 245, 140)
                self._blit_sprite_outline(surface, spr, rect, color=hi)
                self._hover_tooltip = (lines, (mx, my))

        if self.mount == "bike":
            rider: pygame.Surface | None = None
            rf = getattr(self, "cyclist_frames", getattr(self, "_CYCLIST_FRAMES", {})).get(d)
            if rf:
                rider = rf[int(frame) % len(rf)]
            else:
                pf = getattr(self, "player_frames", self._PLAYER_FRAMES)
                frames = pf.get(d, pf["down"])
                if frames:
                    if moving and len(frames) > 1:
                        walk = frames[1:]
                        rider = walk[int(self.bike_anim) % len(walk)]     
                    else:
                        rider = frames[0]

            if rider is not None:
                rrect = rider.get_rect()
                seat_off = 5 if d in ("up", "down") else 4
                if str(mid).startswith("moto"):
                    seat_off = max(2, int(seat_off) - 1)
                rrect.midbottom = (rect.centerx, rect.centery + int(seat_off))
                surface.blit(rider, rrect)
            else:
                hair = self._PLAYER_PAL["H"]
                skin = self._PLAYER_PAL["S"]
                coat = self._PLAYER_PAL["C"]
                hx = rect.centerx
                hy = rect.centery - 6
                pygame.draw.rect(surface, hair, pygame.Rect(hx - 2, hy - 3, 4, 3))
                pygame.draw.rect(surface, skin, pygame.Rect(hx - 2, hy, 4, 3))
                pygame.draw.rect(surface, coat, pygame.Rect(hx - 3, hy + 3, 6, 4))

    def _roof_surface(self, w_tiles: int, h_tiles: int, alpha: int, *, roof_kind: int = 0) -> pygame.Surface:
        w_tiles = max(1, int(w_tiles))
        h_tiles = max(1, int(h_tiles))
        alpha = int(clamp(int(alpha), 0, 255))
        w_px = w_tiles * self.TILE_SIZE
        h_px = h_tiles * self.TILE_SIZE
        roof_kind = int(roof_kind) & 0xFFFFFFFF
        key = (w_px, h_px, alpha, roof_kind)
        cached = self._roof_cache.get(key)
        if cached is not None:
            return cached

        style = int((roof_kind >> 8) & 0xFF)
        var = int(roof_kind & 0xFF)

        def tint(c: tuple[int, int, int], dv: int) -> tuple[int, int, int]:
            return (
                int(clamp(int(c[0]) + dv, 0, 255)),
                int(clamp(int(c[1]) + dv, 0, 255)),
                int(clamp(int(c[2]) + dv, 0, 255)),
            )

        # Stronger per-building roof palette so city blocks read distinctly.
        dv = (var % 9) - 4
        base_rgb = (28, 28, 34)
        stripe_rgb = (44, 44, 54)
        accent_rgb = (70, 70, 86)
        outline_rgb = (10, 10, 12)
        if style == 1:  # 住宅
            base_rgb = (62, 48, 36)
            stripe_rgb = (82, 60, 44)
            accent_rgb = (110, 80, 58)
        elif style == 2:  # 超市
            base_rgb = (34, 36, 40)
            stripe_rgb = (52, 54, 60)
            accent_rgb = (90, 92, 98)
        elif style == 3:  # 医院
            base_rgb = (74, 78, 88)
            stripe_rgb = (106, 110, 122)
            accent_rgb = (170, 174, 186)
        elif style == 4:  # 监狱
            base_rgb = (18, 18, 22)
            stripe_rgb = (34, 34, 40)
            accent_rgb = (66, 66, 78)
        elif style == 5:  # 学校
            base_rgb = (72, 54, 34)
            stripe_rgb = (98, 72, 44)
            accent_rgb = (134, 100, 62)
        elif style == 6:  # 高层住宅
            base_rgb = (42, 46, 62)
            stripe_rgb = (62, 66, 86)
            accent_rgb = (104, 110, 142)
        elif style == 7:  # 新华书店
            base_rgb = (36, 44, 66)
            stripe_rgb = (56, 66, 98)
            accent_rgb = (232, 200, 120)
        elif style == 8:  # 中式建筑
            base_rgb = (34, 64, 44)
            stripe_rgb = (52, 92, 62)
            accent_rgb = (200, 170, 110)

        base_rgb = tint(base_rgb, dv)
        stripe_rgb = tint(stripe_rgb, dv)
        accent_rgb = tint(accent_rgb, dv)

        roof = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
        roof.fill((*base_rgb, alpha))

        if style == 2:
            # Flat roof + HVAC units.
            for y in range(0, h_px, 5):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            unit = (*accent_rgb, alpha)
            for uy in range(6 + (var % 3), h_px - 6, 14):
                for ux in range(6 + ((var >> 2) % 5), w_px - 8, 18):
                    if ((ux + uy + var) % 3) != 0:
                        continue
                    pygame.draw.rect(roof, unit, pygame.Rect(ux, uy, 7, 4), border_radius=1)
            # Long skylight strips (makes supermarkets read big/flat).
            if w_px >= 26 and h_px >= 16:
                glass = (110, 140, 168, alpha)
                frame = (*outline_rgb, alpha)
                for sy in range(8 + (var % 7), h_px - 10, 18):
                    if ((sy + var) % 2) != 0:
                        continue
                    sky = pygame.Rect(6, sy, w_px - 12, 5)
                    roof.fill(glass, sky)
                    pygame.draw.rect(roof, frame, sky, 1, border_radius=1)
        elif style == 3:
            # Medical roof: light + cross marks.
            for y in range(0, h_px, 4):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            cross = (*accent_rgb, alpha)
            for cy in range(8, h_px - 8, 18):
                for cx in range(8, w_px - 8, 18):
                    if ((cx + cy + var) % 2) != 0:
                        continue
                    roof.fill(cross, pygame.Rect(cx - 1, cy - 3, 2, 7))
                    roof.fill(cross, pygame.Rect(cx - 3, cy - 1, 7, 2))
        elif style == 4:
            # Bars / grid.
            for x in range(0, w_px, 4):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (x, 0), (x, h_px - 1), 1)
            for y in range(0, h_px, 8):
                pygame.draw.line(roof, (*accent_rgb, alpha), (0, y), (w_px - 1, y), 1)
        elif style == 5:
            # Skylights.
            for y in range(0, h_px, 4):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            sky = (*accent_rgb, alpha)
            for uy in range(6, h_px - 8, 12):
                for ux in range(6, w_px - 10, 20):
                    if ((ux * 3 + uy * 5 + var) % 4) != 0:
                        continue
                    roof.fill(sky, pygame.Rect(ux, uy, 10, 4))
            # Small solar-panel array for schools.
            if w_px >= 26 and h_px >= 18:
                panel = (46, 70, 88, alpha)
                frame = (*outline_rgb, alpha)
                px0 = 6
                py0 = int(h_px - 12)
                for i in range(0, min(4, int((w_px - 12) // 10))):
                    r = pygame.Rect(px0 + i * 10, py0, 8, 4)
                    roof.fill(panel, r)
                    pygame.draw.rect(roof, frame, r, 1, border_radius=1)
        elif style == 1:
            # Two-tone gable shading + shingles.
            ridge_y = int(h_px // 2)
            roof.fill((*tint(base_rgb, -10), alpha), pygame.Rect(0, 0, w_px, ridge_y))
            roof.fill((*tint(base_rgb, 8), alpha), pygame.Rect(0, ridge_y, w_px, h_px - ridge_y))
            # Shingles diagonal.
            for y in range(-w_px, h_px, 4):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y + w_px - 1), 1)
            # A subtle ridge line to sell "house roof" at small sizes.
            if w_px >= 16 and h_px >= 12:
                ridge = (*tint(accent_rgb, 16), alpha)
                ry = int(h_px // 2)
                pygame.draw.line(roof, ridge, (1, ry), (w_px - 2, ry), 1)
            # Chimney.
            if w_px >= 26 and h_px >= 18:
                cx = 6 if (var & 1) == 0 else (w_px - 11)
                cy = 6 + (var % 5)
                chim = pygame.Rect(int(cx), int(cy), 5, 7)
                roof.fill((*tint(accent_rgb, 26), alpha), chim)
                pygame.draw.rect(roof, (*outline_rgb, alpha), chim, 1)
                roof.fill((*tint(accent_rgb, 42), alpha), pygame.Rect(chim.x + 1, chim.y + 1, chim.w - 2, 1))
        elif style == 6:
            # High-rise: service roof (vents + access hatches).
            for y in range(0, h_px, 6):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            vent = (*accent_rgb, alpha)
            hatch = (*tint(accent_rgb, 22), alpha)
            for uy in range(6 + (var % 3), h_px - 6, 12):
                for ux in range(6 + ((var >> 2) % 4), w_px - 6, 12):
                    if ((ux + uy + var) % 3) != 0:
                        continue
                    roof.fill(vent, pygame.Rect(ux, uy, 4, 3))
                    roof.fill(vent, pygame.Rect(ux + 1, uy + 1, 2, 1))
            if w_px >= 26 and h_px >= 18:
                hx = int(w_px - 14)
                hy = int(6 + (var % 5))
                pygame.draw.rect(roof, hatch, pygame.Rect(hx, hy, 10, 6), border_radius=1)
                pygame.draw.rect(
                    roof,
                    (*tint(outline_rgb, 24), alpha),
                    pygame.Rect(hx, hy, 10, 6),
                    1,
                    border_radius=1,
                )
            # Elevator core / roof access (makes high-rises instantly recognizable).
            if w_px >= 28 and h_px >= 28:
                cx = int(w_px // 2 - 6)
                cy = int(h_px // 2 - 6)
                core = pygame.Rect(cx, cy, 12, 12)
                roof.fill((*tint(accent_rgb, 30), alpha), core)
                pygame.draw.rect(roof, (*outline_rgb, alpha), core, 1, border_radius=1)
                roof.fill((*tint(accent_rgb, 52), alpha), pygame.Rect(core.x + 3, core.y + 3, core.w - 6, 1))
        elif style == 7:
            # Bookstore: flat roof + small skylights.
            for y in range(0, h_px, 5):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            if w_px >= 22 and h_px >= 16:
                glass = (110, 140, 168, alpha)
                frame = (*outline_rgb, alpha)
                for uy in range(6 + (var % 4), h_px - 8, 14):
                    for ux in range(6 + ((var >> 2) % 4), w_px - 10, 18):
                        if ((ux + uy + var) % 3) != 0:
                            continue
                        sky = pygame.Rect(int(ux), int(uy), 8, 4)
                        roof.fill(glass, sky)
                        pygame.draw.rect(roof, frame, sky, 1, border_radius=1)
        elif style == 8:
            # Chinese-style tiled roof + ridge line.
            for y in range(0, h_px, 3):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)
            if w_px >= 16 and h_px >= 12:
                ry = int(h_px // 2)
                ridge = (*tint(accent_rgb, 18), alpha)
                pygame.draw.line(roof, ridge, (2, ry), (w_px - 3, ry), 1)
                eave = (*tint(accent_rgb, 34), alpha)
                roof.fill(eave, pygame.Rect(1, 1, 2, 2))
                roof.fill(eave, pygame.Rect(w_px - 3, 1, 2, 2))
                roof.fill(eave, pygame.Rect(1, h_px - 3, 2, 2))
                roof.fill(eave, pygame.Rect(w_px - 3, h_px - 3, 2, 2))
        else:
            # Default subtle stripes.
            for y in range(0, h_px, 4):
                pygame.draw.line(roof, (*stripe_rgb, alpha), (0, y), (w_px - 1, y), 1)

        # Big visual identifiers per building type so roofs don't all read the same.
        if alpha > 0:
            if style == 3 and w_px >= 22 and h_px >= 22:
                # Hospital: central cross pad.
                cx = int(w_px // 2)
                cy = int(h_px // 2)
                pad = pygame.Rect(int(cx - 8), int(cy - 8), 17, 17)
                pygame.draw.rect(roof, (232, 232, 238, alpha), pad, border_radius=2)
                pygame.draw.rect(roof, (*outline_rgb, alpha), pad, 1, border_radius=2)
                red = (190, 56, 56, alpha)
                roof.fill(red, pygame.Rect(int(cx - 2), int(cy - 7), 5, 15))
                roof.fill(red, pygame.Rect(int(cx - 7), int(cy - 2), 15, 5))
                ring = (210, 210, 220, alpha)
                pygame.draw.circle(roof, ring, (int(cx), int(cy)), 9, 1)
            elif style == 2 and w_px >= 22 and h_px >= 14:
                # Supermarket: bold stripe + logo block.
                band = pygame.Rect(0, int(h_px // 2 - 2), int(w_px), 5)
                roof.fill((34, 92, 56, alpha), band)
                roof.fill((170, 54, 54, alpha), pygame.Rect(0, int(band.y), int(w_px), 1))
                logo = pygame.Rect(int(w_px // 2 - 4), int(h_px // 2 - 4), 8, 8)
                pygame.draw.rect(roof, (232, 190, 88, alpha), logo, border_radius=2)
                pygame.draw.rect(roof, (*outline_rgb, alpha), logo, 1, border_radius=2)
            elif style == 7 and w_px >= 18 and h_px >= 18:
                # Bookstore: simple "book" mark.
                cx = int(w_px // 2)
                cy = int(h_px // 2)
                ink = (232, 200, 120, alpha)
                pygame.draw.rect(roof, ink, pygame.Rect(cx - 8, cy - 5, 7, 10), 1, border_radius=1)
                pygame.draw.rect(roof, ink, pygame.Rect(cx + 1, cy - 5, 7, 10), 1, border_radius=1)
                roof.fill((10, 10, 12, alpha), pygame.Rect(cx - 1, cy - 5, 2, 10))
            elif style == 8 and w_px >= 18 and h_px >= 18:
                # Chinese roof: central ridge ornament.
                cx = int(w_px // 2)
                cy = int(h_px // 2)
                gold = (210, 180, 110, alpha)
                roof.fill(gold, pygame.Rect(cx - 1, cy - 7, 2, 15))
                roof.fill(gold, pygame.Rect(cx - 5, cy - 1, 10, 2))
            elif style == 4 and w_px >= 18 and h_px >= 18:
                # Prison: perimeter wire marks.
                wire = (*tint(stripe_rgb, 28), alpha)
                pygame.draw.rect(roof, wire, pygame.Rect(1, 1, w_px - 2, h_px - 2), 1)
                for x in range(3, w_px - 3, 5):
                    roof.fill(wire, pygame.Rect(x, 1, 1, 1))
                    roof.fill(wire, pygame.Rect(x, h_px - 2, 1, 1))
                for y in range(3, h_px - 3, 5):
                    roof.fill(wire, pygame.Rect(1, y, 1, 1))
                    roof.fill(wire, pygame.Rect(w_px - 2, y, 1, 1))
                # Corner watch-tower blocks.
                tower = (*tint(accent_rgb, 24), alpha)
                for ox, oy in ((2, 2), (w_px - 8, 2), (2, h_px - 8), (w_px - 8, h_px - 8)):
                    if ((ox + oy + var) % 2) != 0:
                        continue
                    r = pygame.Rect(int(ox), int(oy), 6, 6)
                    roof.fill(tower, r)
                    pygame.draw.rect(roof, (*outline_rgb, alpha), r, 1, border_radius=1)
                    roof.fill((*tint(accent_rgb, 54), alpha), pygame.Rect(r.x + 2, r.y + 2, 2, 2))
            elif style == 5 and w_px >= 22 and h_px >= 16:
                # School: a bright gym roof patch.
                gym = pygame.Rect(4, 4, w_px - 8, max(6, h_px // 2 - 2))
                pygame.draw.rect(roof, (*tint(accent_rgb, 34), alpha), gym, border_radius=2)
                pygame.draw.rect(roof, (*outline_rgb, alpha), gym, 1, border_radius=2)
                # Tiny "S" mark to make schools readable even in clusters.
                if gym.w >= 18 and gym.h >= 10:
                    sx = int(gym.centerx)
                    sy = int(gym.centery)
                    ink = (*outline_rgb, alpha)
                    roof.fill(ink, pygame.Rect(sx - 5, sy - 3, 10, 2))
                    roof.fill(ink, pygame.Rect(sx - 5, sy + 1, 10, 2))
                    roof.fill(ink, pygame.Rect(sx - 5, sy - 1, 2, 2))
                    roof.fill(ink, pygame.Rect(sx + 3, sy - 1, 2, 2))

        # Hint storefront/facade signage to make city blocks read more like the reference.
        if alpha > 0 and w_px >= 18 and h_px >= 14:
            face = int(var) & 3  # 0=top,1=bottom,2=left,3=right
            if style in (2, 3):
                sign_rgb = tint(accent_rgb, 26)
                text_rgb = tint(accent_rgb, 44)
            elif style == 6:
                sign_rgb = tint(accent_rgb, 18)
                text_rgb = tint(accent_rgb, 34)
            else:
                sign_rgb = tint(stripe_rgb, 18)
                text_rgb = tint(stripe_rgb, 34)

            sw = int(min(w_px - 6, 34 + (var % 10)))
            sh = 5
            if face == 0:
                sign = pygame.Rect(3, 3, sw, sh)
            elif face == 1:
                sign = pygame.Rect(3, h_px - 3 - sh, sw, sh)
            elif face == 2:
                sign = pygame.Rect(3, 3, sh, int(min(h_px - 6, 26 + (var % 8))))
            else:
                sign = pygame.Rect(w_px - 3 - sh, 3, sh, int(min(h_px - 6, 26 + (var % 8))))
            roof.fill((*sign_rgb, alpha), sign)

            # Tiny pseudo-text pixels.
            if sign.w >= sign.h:
                ty = sign.y + 2
                for x in range(sign.x + 2, sign.right - 2, 3):
                    if ((x + var) % 5) == 0:
                        roof.fill((*text_rgb, alpha), pygame.Rect(x, ty, 2, 1))
            else:
                tx = sign.x + 2
                for y in range(sign.y + 2, sign.bottom - 2, 3):
                    if ((y + var) % 5) == 0:
                        roof.fill((*text_rgb, alpha), pygame.Rect(tx, y, 1, 2))

        # Extra rooftop props for high-rises.
        if style == 6 and w_px >= 20 and h_px >= 20:
            tank = (*accent_rgb, alpha)
            for ox, oy in ((6, 6), (w_px - 11, 6), (6, h_px - 11), (w_px - 11, h_px - 11)):
                if ((ox + oy + var) % 3) != 0:
                    continue
                pygame.draw.rect(roof, tank, pygame.Rect(int(ox), int(oy), 5, 5), border_radius=1)

        # Roof outline + subtle highlight edge so different roof palettes pop.
        if alpha > 0 and w_px >= 3 and h_px >= 3:
            pygame.draw.rect(roof, (*outline_rgb, alpha), roof.get_rect(), 1)
            hi = (*tint(base_rgb, 22), alpha)
            roof.fill(hi, pygame.Rect(1, h_px - 2, w_px - 2, 1))
            roof.fill(hi, pygame.Rect(w_px - 2, 1, 1, h_px - 2))
        self._roof_cache[key] = roof
        return roof

    def _building_roof_style_var(self, roof_kind: int) -> tuple[int, int]:
        roof_kind = int(roof_kind) & 0xFFFFFFFF
        style = int((roof_kind >> 8) & 0xFF)
        var = int(roof_kind & 0xFF)
        return style, var

    def _building_face_height_px(self, *, style: int, w: int, h: int, var: int, floors: int = 0) -> int:
        style = int(style)
        w = int(w)
        h = int(h)
        var = int(var)
        floors = int(floors)
        area = int(w) * int(h)
        # Wall extrusion height: makes building height readable at a glance.
        # High-rises should read "very tall" even at small internal resolution.
        if style == 6:
            f = int(floors) if int(floors) > 0 else 10
            f = int(clamp(int(f), 6, 30))
            base = 26 + int(f) * 3
            if area >= 520:
                base += 14
            elif area >= 360:
                base += 10
            elif area >= 240:
                base += 6
            base += (int(var) % 11) - 5

            roof_h = int(max(1, (int(h) - 2) * int(self.TILE_SIZE)))
            max_face = int(max(36, int(roof_h - 18)))
            return int(clamp(int(base), 36, int(min(max_face, 140))))

        base = {
            1: 3,  # 住宅
            2: 5,  # 超市/商店
            3: 5,  # 医院
            4: 6,  # 监狱
            5: 5,  # 学校
            7: 5,  # 书店
            8: 4,  # 中式建筑
        }.get(style, 3)
        if area >= 520:
            base += 3
        elif area >= 320:
            base += 2
        elif area >= 200:
            base += 1
        base += (int(var) % 5) - 2
        return int(clamp(int(base), 3, 16))

    def _building_wall_palette(self, *, style: int, var: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        style = int(style)
        var = int(var)
        # (front, side, trim, shadow) – tuned for a dirty city feel like the reference.
        front = (88, 84, 78)
        side = (64, 62, 58)
        trim = (140, 140, 146)
        shadow = (14, 14, 18)
        if style == 1:  # 住宅
            front, side, trim = (118, 102, 82), (92, 78, 62), (170, 148, 118)
        elif style == 2:  # 超市/商店
            front, side, trim = (86, 88, 94), (64, 66, 72), (130, 132, 140)
        elif style == 3:  # 医院
            front, side, trim = (184, 186, 196), (148, 150, 160), (236, 236, 242)
        elif style == 4:  # 监狱
            front, side, trim = (58, 58, 66), (40, 40, 46), (98, 98, 110)
        elif style == 5:  # 学校
            front, side, trim = (132, 102, 68), (102, 76, 50), (186, 146, 96)
        elif style == 6:  # 高层住宅
            front, side, trim = (88, 92, 122), (66, 70, 94), (142, 150, 186)
        elif style == 7:  # 书店
            front, side, trim = (72, 80, 108), (52, 58, 78), (210, 180, 110)
        elif style == 8:  # 中式建筑
            front, side, trim = (148, 58, 46), (112, 42, 34), (210, 180, 110)
        dv = (int(var) % 7) - 3
        dv2 = (int(var) % 5) - 2
        front = self._tint(front, add=(dv * 2, dv * 2, dv * 2))
        side = self._tint(side, add=(dv2 * 2, dv2 * 2, dv2 * 2))
        trim = self._tint(trim, add=(dv, dv, dv))
        return front, side, trim, shadow

    def _find_building_exterior_door(self, tx0: int, ty0: int, w: int, h: int) -> tuple[str, list[tuple[int, int]]] | None:
        tx0 = int(tx0)
        ty0 = int(ty0)
        w = int(w)
        h = int(h)
        doors: list[tuple[int, int]] = []
        # Perimeter only (skip interior doors).
        for x in range(tx0, tx0 + w):
            if int(self.world.peek_tile(x, ty0)) == int(self.T_DOOR):
                doors.append((x, ty0))
            if int(self.world.peek_tile(x, ty0 + h - 1)) == int(self.T_DOOR):
                doors.append((x, ty0 + h - 1))
        for y in range(ty0 + 1, ty0 + h - 1):
            if int(self.world.peek_tile(tx0, y)) == int(self.T_DOOR):
                doors.append((tx0, y))
            if int(self.world.peek_tile(tx0 + w - 1, y)) == int(self.T_DOOR):
                doors.append((tx0 + w - 1, y))
        if not doors:
            return None
        side_votes = {"N": 0, "S": 0, "W": 0, "E": 0}
        for x, y in doors:
            if y == ty0:
                side_votes["N"] += 1
            elif y == ty0 + h - 1:
                side_votes["S"] += 1
            elif x == tx0:
                side_votes["W"] += 1
            elif x == tx0 + w - 1:
                side_votes["E"] += 1
        side = max(side_votes.items(), key=lambda kv: kv[1])[0]
        return side, doors

    def _draw_facade_furniture_silhouette(
        self,
        surface: pygame.Surface,
        *,
        kind: str,
        x: int,
        floor_y: int,
        w: int,
        max_h: int,
        seed: int,
        depth: int = 0,
    ) -> None:
        kind = str(kind)
        x = int(x)
        floor_y = int(floor_y)
        w = int(w)
        max_h = int(max_h)
        seed = int(seed) & 0xFFFFFFFF
        depth = int(depth)

        if w <= 2 or max_h <= 2:
            return

        outline = (10, 10, 12)
        shade = int(-58 - depth * 10)
        add = (shade, shade, shade)
        hi_add = (shade + 18, shade + 18, shade + 18)
        lo_add = (shade - 14, shade - 14, shade - 14)

        if kind == "table":
            wood = self._tint((118, 92, 66), add=add)
            wood_hi = self._tint((118, 92, 66), add=hi_add)
            wood_lo = self._tint((118, 92, 66), add=lo_add)

            top_h = 2 if max_h >= 6 else 1
            legs_len = int(clamp(max_h - top_h - 1, 2, 4))
            top_y = int(floor_y - (legs_len + top_h))
            top = pygame.Rect(int(x), int(top_y), int(w), int(top_h))

            surface.fill(wood, top)
            surface.fill(wood_hi, pygame.Rect(top.x, top.y, top.w, 1))
            pygame.draw.rect(surface, outline, top, 1)

            leg_h = int(floor_y - top.bottom)
            if leg_h > 0:
                lx0 = int(x + 2)
                lx1 = int(x + w - 3)
                for lx in (lx0, lx1):
                    if int(x) <= lx < int(x + w):
                        surface.fill(wood_lo, pygame.Rect(lx, top.bottom, 1, leg_h))
                        surface.fill(outline, pygame.Rect(lx, top.bottom, 1, leg_h))
                if leg_h >= 3 and lx1 - lx0 >= 3:
                    yb = int(floor_y - 2)
                    surface.fill(wood_lo, pygame.Rect(int(lx0), yb, int(lx1 - lx0 + 1), 1))

            # Chair silhouette (legs visible).
            if w >= 7 and max_h >= 6 and (seed & 3) == 0:
                seat_c = self._tint((92, 120, 154), add=add)
                seat_hi = self._tint((92, 120, 154), add=hi_add)
                seat_lo = self._tint((92, 120, 154), add=lo_add)
                cw = int(clamp(w - 2, 5, 7))
                cx = int(x + (1 if (seed & 4) == 0 else (w - cw - 1)))
                seat_y = int(floor_y - 3)
                seat_r = pygame.Rect(cx, seat_y, cw, 2)
                surface.fill(seat_c, seat_r)
                surface.fill(seat_hi, pygame.Rect(seat_r.x, seat_r.y, seat_r.w, 1))
                pygame.draw.rect(surface, outline, seat_r, 1)

                back_h = 3
                if (seed & 4) == 0:
                    back = pygame.Rect(seat_r.x, seat_r.y - back_h, 2, back_h)
                else:
                    back = pygame.Rect(seat_r.right - 2, seat_r.y - back_h, 2, back_h)
                surface.fill(seat_lo, back)
                pygame.draw.rect(surface, outline, back, 1)

                leg_h = int(floor_y - seat_r.bottom)
                if leg_h > 0:
                    for lx in (seat_r.x + 1, seat_r.right - 2):
                        surface.fill(seat_lo, pygame.Rect(int(lx), int(seat_r.bottom), 1, leg_h))
                        surface.fill(outline, pygame.Rect(int(lx), int(seat_r.bottom), 1, leg_h))

            return

        if kind == "shelf":
            cab = self._tint((98, 96, 112), add=add)
            cab_hi = self._tint((98, 96, 112), add=hi_add)
            cab_lo = self._tint((98, 96, 112), add=lo_add)

            h = int(clamp(max_h, 4, 8))
            y0 = int(floor_y - h)
            body = pygame.Rect(int(x), int(y0), int(w), int(h))
            surface.fill(cab, body)
            surface.fill(cab_hi, pygame.Rect(body.x, body.y, body.w, 1))
            pygame.draw.rect(surface, outline, body, 1)

            for sy in range(body.y + 2, body.bottom - 2, 2):
                surface.fill(cab_lo, pygame.Rect(body.x + 1, sy, max(1, body.w - 2), 1))

            # Feet.
            surface.fill(cab_lo, pygame.Rect(body.x + 1, floor_y - 1, 1, 1))
            surface.fill(cab_lo, pygame.Rect(body.right - 2, floor_y - 1, 1, 1))

            # Items: small colored pixels on shelves (deterministic).
            if body.w >= 6 and body.h >= 5:
                for i in range(3):
                    # IMPORTANT: don't use screen coordinates here, otherwise it flickers while walking.
                    hh = int(
                        self._hash2_u32(
                            int((seed ^ 0xA1F3C9D7) + i * 131),
                            int((seed >> 8) + i * 313),
                            int(seed ^ 0x9E3779B9),
                        )
                    )
                    ix = int(body.x + 2 + (hh % max(1, body.w - 4)))
                    iy = int(body.y + 2 + ((hh >> 5) % max(1, body.h - 4)))
                    col = (200, 86, 86) if (hh & 1) else (86, 210, 130)
                    col = self._tint(col, add=(shade + 10, shade + 10, shade + 10))
                    surface.fill(col, pygame.Rect(ix, iy, 1, 1))
            return

        if kind == "bed":
            matt = self._tint((156, 150, 170), add=add)
            matt_hi = self._tint((156, 150, 170), add=hi_add)
            matt_lo = self._tint((156, 150, 170), add=lo_add)
            wood = self._tint((118, 92, 66), add=lo_add)

            h = int(clamp(max_h, 4, 7))
            y0 = int(floor_y - h)
            bed = pygame.Rect(int(x), int(y0), int(w), int(h))
            surface.fill(matt, bed)
            surface.fill(matt_hi, pygame.Rect(bed.x, bed.y, bed.w, 1))
            pygame.draw.rect(surface, outline, bed, 1)

            if (seed & 1) == 0:
                hb = pygame.Rect(bed.x, bed.y, 2, bed.h)
            else:
                hb = pygame.Rect(bed.right - 2, bed.y, 2, bed.h)
            surface.fill(wood, hb)
            pygame.draw.rect(surface, outline, hb, 1)

            px = bed.x + 3 if hb.x == bed.x else bed.right - 6
            py = bed.y + 2
            pillow = pygame.Rect(int(px), int(py), 3, 2)
            surface.fill(matt_hi, pillow)
            pygame.draw.rect(surface, outline, pillow, 1)

            if bed.w >= 6 and bed.h >= 5:
                fold_y = bed.y + bed.h // 2
                surface.fill(matt_lo, pygame.Rect(bed.x + 2, fold_y, max(1, bed.w - 4), 1))

            surface.fill(matt_lo, pygame.Rect(bed.x + 1, floor_y - 1, 1, 1))
            surface.fill(matt_lo, pygame.Rect(bed.right - 2, floor_y - 1, 1, 1))
            return

    def _draw_building_facades(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
        *,
        min_ground_y: int | None = None,
    ) -> None:
        # Expand by a full chunk in each direction so facades from large
        # buildings don't pop in/out at the screen edges while walking.
        chunk_pad = 1
        start_cx = start_tx // self.CHUNK_SIZE - chunk_pad
        end_cx = end_tx // self.CHUNK_SIZE + chunk_pad
        start_cy = start_ty // self.CHUNK_SIZE - chunk_pad
        end_cy = end_ty // self.CHUNK_SIZE + chunk_pad

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue
                for b in getattr(chunk, "buildings", []):
                    tx0, ty0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    roof_kind = int(b[4]) if len(b) > 4 else 0
                    style, var = self._building_roof_style_var(int(roof_kind))

                    floors = int(b[5]) if len(b) > 5 else 0
                    face_h = int(self._building_face_height_px(style=style, w=w, h=h, var=var, floors=floors))
                    # Only draw the *front* facade (bottom/S) as requested.
                    # Keep visibility checks stable to avoid popping at edges while walking.
                    bx = int(tx0 * self.TILE_SIZE - cam_x)
                    by = int(ty0 * self.TILE_SIZE - cam_y)
                    bw = int(w * self.TILE_SIZE)
                    bh = int(h * self.TILE_SIZE)
                    if bx > INTERNAL_W or by > INTERNAL_H:
                        continue
                    if (bx + bw) < 0 or (by + bh) < 0:
                        continue

                    front, side, trim, shadow = self._building_wall_palette(style=style, var=var)
                    outline = (10, 10, 12)

                    # Door detection (used for storefront detail).
                    door_info = self._find_building_exterior_door(tx0, ty0, w, h)
                    door_side = ""
                    door_tiles: list[tuple[int, int]] = []
                    if door_info is not None:
                        door_side, door_tiles = door_info

                    door_px: pygame.Rect | None = None
                    if door_tiles and door_side == "S":
                        xs = [p[0] for p in door_tiles]
                        ys = [p[1] for p in door_tiles]
                        dx0 = int(min(xs))
                        dx1 = int(max(xs))
                        dy = int(max(ys))
                        door_px = pygame.Rect(
                            int(dx0 * self.TILE_SIZE - cam_x),
                            int(dy * self.TILE_SIZE - cam_y),
                            int((dx1 - dx0 + 1) * self.TILE_SIZE),
                            int(self.TILE_SIZE + face_h),
                        )

                    ground_y = int(by + bh)
                    # Optional occlusion pass: only draw facades that are "in front"
                    # of a given screen-space y (used to keep vehicles from looking
                    # like they're on top of the building when behind it).
                    if min_ground_y is not None and int(ground_y) <= int(min_ground_y):
                        continue
                    # South (bottom) face: draw the "front facade".
                    # For high-rises, extend the facade upward so multiple
                    # window rows can read as "tall" (roof is clipped for
                    # style==6 to avoid overlap).
                    south_y = int(ground_y - self.TILE_SIZE)
                    south_h = int(self.TILE_SIZE)
                    if style == 6:
                        south_y = int(ground_y - self.TILE_SIZE - int(face_h))
                        south_h = int(self.TILE_SIZE + int(face_h))
                    south = pygame.Rect(int(bx), int(south_y), int(bw), int(south_h))
                    pygame.draw.rect(surface, front, south)
                    pygame.draw.rect(surface, outline, south, 1)
                    # Ground shadow under the facade.
                    shadow_h = int(clamp(2 + int(face_h) // 10, 2, 5))
                    sh = pygame.Rect(int(south.x + 2), int(south.bottom), int(south.w - 2), int(shadow_h))
                    pygame.draw.rect(surface, shadow, sh)
                    # Lower-half shade for depth (independent of "building height").
                    shade_h = int(max(2, int(south.h) // 2))
                    shade = self._tint(front, add=(-18, -18, -18))
                    surface.fill(shade, pygame.Rect(int(south.x), int(south.bottom - shade_h), int(south.w), int(shade_h)))

                    # Trim line under the roof.
                    surface.fill(trim, pygame.Rect(south.x, south.y + 1, south.w, 2))

                    # Facade details by building type.
                    store_header_bottom = int(south.y + 3)
                    if style in (2, 3, 5, 7):  # shop/hospital/school/bookstore: sign + band
                        sign_h = int(clamp(int(south.h // 3), 5, 7))
                        sign = pygame.Rect(south.x + 3, south.y + 2, south.w - 6, sign_h)
                        if style == 2:
                            base = (78, 118, 176) if ((var >> 1) & 1) == 0 else (176, 132, 78)
                            sign_bg = self._tint(base, add=(-10, -10, -10))
                        elif style == 7:
                            base = (46, 70, 110)
                            sign_bg = self._tint(base, add=(-8, -8, -8))
                        elif style == 5:
                            base = (168, 132, 92)
                            sign_bg = self._tint(base, add=(-18, -14, -10))
                        elif style == 3:
                            sign_bg = self._tint(trim, add=(-4, -4, -4))
                        else:
                            sign_bg = self._tint(trim, add=(-22, -22, -22))
                        pygame.draw.rect(surface, sign_bg, sign, border_radius=2)
                        pygame.draw.rect(surface, outline, sign, 1, border_radius=2)
                        # Tiny pseudo text blocks
                        for xx in range(sign.x + 4 + (var % 3), sign.right - 6, 4):
                            # IMPORTANT: use local x (not screen coordinates) so it doesn't flicker while walking.
                            if (((xx - sign.x) + var) % 5) == 0:
                                continue
                            surface.fill(self._tint(sign_bg, add=(38, 38, 42)), pygame.Rect(xx, sign.y + 2, 2, 1))
                        if style == 3 and sign.w >= 18:
                            # Red cross
                            cx = sign.centerx
                            cy = sign.centery
                            red = (190, 60, 60)
                            surface.fill(red, pygame.Rect(cx - 1, cy - 3, 2, 7))
                            surface.fill(red, pygame.Rect(cx - 3, cy - 1, 7, 2))
                            pygame.draw.rect(surface, outline, pygame.Rect(cx - 3, cy - 3, 7, 7), 1)
                        if style == 7 and sign.w >= 18:
                            # Tiny book icon.
                            cx = sign.centerx
                            cy = sign.centery
                            ink = (232, 200, 120)
                            pygame.draw.rect(surface, ink, pygame.Rect(cx - 6, cy - 3, 5, 7), 1, border_radius=1)
                            pygame.draw.rect(surface, ink, pygame.Rect(cx + 1, cy - 3, 5, 7), 1, border_radius=1)
                            surface.fill(outline, pygame.Rect(cx, cy - 3, 1, 7))

                        # Awning stripes just below sign for shops.
                        store_header_bottom = int(sign.bottom)
                        if style == 2:
                            awn_h = 5 if int(south.h) >= 16 else 4
                            awn = pygame.Rect(south.x + 3, sign.bottom + 1, south.w - 6, awn_h)
                            a1 = (190, 190, 196)
                            a2 = (170, 80, 80)
                            for i in range(awn.w):
                                col = a1 if ((i + var) % 6) < 3 else a2
                                surface.fill(col, pygame.Rect(awn.x + i, awn.y, 1, awn.h))
                            pygame.draw.rect(surface, outline, awn, 1)
                            store_header_bottom = int(awn.bottom)
                            # Small hanging placard on the right.
                            if awn.w >= 34:
                                plac = pygame.Rect(int(awn.right - 14), int(awn.bottom + 2), 10, 10)
                                pygame.draw.rect(surface, self._tint(sign_bg, add=(18, 18, 18)), plac, border_radius=2)
                                pygame.draw.rect(surface, outline, plac, 1, border_radius=2)
                                surface.fill((240, 220, 140), pygame.Rect(plac.x + 3, plac.y + 4, 4, 1))

                    if style == 3:
                        # Hospital: clean stripe band.
                        stripe = (92, 138, 206)
                        sy = int(store_header_bottom + 2)
                        if sy + 2 < south.bottom - 4:
                            surface.fill(stripe, pygame.Rect(int(south.x + 3), int(sy), int(south.w - 6), 2))

                    if style == 5:
                        # School: brick columns.
                        col = self._tint(front, add=(-26, -20, -16))     
                        if south.w >= 26 and south.h >= 14:
                            surface.fill(col, pygame.Rect(int(south.x + 2), int(south.y + 3), 3, int(south.h - 6)))
                            surface.fill(col, pygame.Rect(int(south.right - 5), int(south.y + 3), 3, int(south.h - 6)))
                            # Small banner tabs.
                            surface.fill(self._tint(trim, add=(-30, -30, -30)), pygame.Rect(int(sign.x + 6), int(sign.bottom), 2, 3))
                            surface.fill(self._tint(trim, add=(-30, -30, -30)), pygame.Rect(int(sign.right - 8), int(sign.bottom), 2, 3))

                    if style == 8:
                        # Chinese: red pillars + a small plaque.
                        if south.w >= 24 and south.h >= 14:
                            pillar = self._tint(front, add=(18, 0, 0))
                            surface.fill(pillar, pygame.Rect(int(south.x + 2), int(south.y + 3), 3, int(south.h - 6)))
                            surface.fill(pillar, pygame.Rect(int(south.right - 5), int(south.y + 3), 3, int(south.h - 6)))
                            plaque = pygame.Rect(int(south.centerx - 8), int(south.y + 2), 16, 6)
                            bg = self._tint(trim, add=(-26, -26, -26))
                            pygame.draw.rect(surface, bg, plaque, border_radius=2)
                            pygame.draw.rect(surface, outline, plaque, 1, border_radius=2)
                            ink = self._tint(trim, add=(22, 18, 0))
                            for xx in range(int(plaque.x + 4), int(plaque.right - 4), 3):
                                if ((xx + var) % 4) == 0:
                                    continue
                                surface.fill(ink, pygame.Rect(int(xx), int(plaque.y + 3), 2, 1))

                    if style == 4:
                        # Prison: barred windows.
                        bar = self._tint(trim, add=(-30, -30, -30))      
                        for xx in range(south.x + 6, south.right - 6, 6):
                            surface.fill(bar, pygame.Rect(xx, south.y + 3, 1, south.h - 6))

                    if style in (1, 6):
                        # Residential/high-rise windows.
                        win = self._tint(trim, add=(-40, -40, -46))
                        frame = outline
                        start_x = int(south.x + 6 + (var % 4))
                        if style == 6:
                            # Multiple rows so height reads as "high-rise".
                            step_x = 10
                            step_y = 8
                            y0w = int(south.y + 4)
                            y1w = int(south.bottom - 7)
                            for row_i, yy in enumerate(range(y0w, y1w, step_y)):
                                for col_i, xx in enumerate(range(start_x, south.right - 10, step_x)):
                                    # IMPORTANT: use local indices (not screen coords) so it doesn't flicker.
                                    if ((col_i + row_i + var) % 4) == 0:
                                        continue
                                    r = pygame.Rect(int(xx), int(yy), 6, 4)
                                    pygame.draw.rect(surface, win, r, border_radius=1)
                                    pygame.draw.rect(surface, frame, r, 1, border_radius=1)
                        else:
                            # Single row for small residential buildings.
                            step_x = 12
                            y1 = int(south.y + 4)
                            for i, xx in enumerate(range(start_x, south.right - 10, step_x)):
                                # IMPORTANT: use local window index (not screen coordinates) so it doesn't flicker.
                                if ((i + var) % 3) == 0:
                                    continue
                                r = pygame.Rect(int(xx), int(y1), 6, 4)
                                pygame.draw.rect(surface, win, r, border_radius=1)
                                pygame.draw.rect(surface, frame, r, 1, border_radius=1)

                    # "Front interior" storefront hint (like the reference): show a cutout window strip with
                    # silhouettes of shelves/props synced to the actual interior tiles.
                    if style in (2, 3, 5, 7):
                        open_y = int(store_header_bottom + 1)
                        open_h = int(south.bottom - 2 - open_y)
                        open_x = int(south.x + 4)
                        open_w = int(south.w - 8)
                        if open_w >= int(self.TILE_SIZE * 2) and open_h >= 4:
                            open_rect = pygame.Rect(int(open_x), int(open_y), int(open_w), int(open_h))
                            interior = self._tint(side, add=(-30, -30, -30))
                            if style == 3:
                                interior = self._tint(front, add=(-34, -34, -30))
                            elif style == 5:
                                interior = self._tint(front, add=(-36, -30, -26))
                            pygame.draw.rect(surface, interior, open_rect, border_radius=2)
                            pygame.draw.rect(surface, outline, open_rect, 1, border_radius=2)

                            # Glass highlight + mullions.
                            glass_hi = self._tint(trim, add=(48, 48, 54))
                            surface.fill(
                                glass_hi,
                                pygame.Rect(open_rect.x + 1, open_rect.y + 1, max(1, open_rect.w - 2), 1),
                            )
                            mull = self._tint(interior, add=(18, 18, 18))
                            step = 12 if open_rect.w >= 80 else 10
                            for xx in range(open_rect.x + 8 + (var % 5), open_rect.right - 6, step):
                                surface.fill(mull, pygame.Rect(int(xx), open_rect.y + 2, 1, max(1, open_rect.h - 4)))

                            prev_clip = surface.get_clip()
                            surface.set_clip(open_rect.inflate(-1, -1))

                            # Interior silhouettes near the front wall.
                            if open_rect.h >= 6 and int(w) >= 4 and int(h) >= 4:
                                sample_ys = [int(ty0 + h - 2), int(ty0 + h - 3)]
                                seed = int(self.seed) ^ 0x51F2A1B3
                                for row_i, sy in enumerate(sample_ys):
                                    if not (int(ty0) < int(sy) < int(ty0 + h - 1)):
                                        continue
                                    y_off = int(row_i)
                                    for tx in range(int(tx0 + 1), int(tx0 + w - 1)):
                                        px = int(tx * self.TILE_SIZE - cam_x)
                                        if px + self.TILE_SIZE <= open_rect.x or px >= open_rect.right:
                                            continue
                                        tid = int(self.world.peek_tile(tx, sy))
                                        if tid == int(self.T_FLOOR):
                                            if style != 2:
                                                continue
                                            hh = int(self._hash2_u32(int(tx), int(sy), seed))
                                            if (hh & 15) != 0:
                                                continue
                                            crate = (132, 92, 62) if ((hh >> 4) & 1) else (92, 70, 52)
                                            rr = pygame.Rect(
                                                int(px + 2),
                                                int(open_rect.bottom - 3 - y_off),
                                                max(1, int(self.TILE_SIZE - 4)),
                                                2,
                                            )
                                            surface.fill(crate, rr)
                                            pygame.draw.rect(surface, outline, rr, 1)
                                            if rr.w >= 5:
                                                g1 = (196, 120, 86) if ((hh >> 6) & 1) else (86, 190, 120)
                                                surface.fill(g1, pygame.Rect(rr.x + 2, rr.y, 1, 1))
                                            continue

                                        if tid in (int(self.T_TABLE), int(self.T_SHELF), int(self.T_BED)):
                                            # More detailed facade furniture silhouettes (legs, shelves, pillows).
                                            fh = int(self._hash2_u32(int(tx), int(sy), seed ^ 0x9E3779B9))
                                            floor_y = int(open_rect.bottom - 2 - y_off)
                                            max_h = int(max(3, open_rect.h - 3))
                                            if tid == int(self.T_TABLE):
                                                self._draw_facade_furniture_silhouette(
                                                    surface,
                                                    kind="table",
                                                    x=int(px + 1),
                                                    floor_y=floor_y,
                                                    w=int(self.TILE_SIZE - 2),
                                                    max_h=max_h,
                                                    seed=fh,
                                                    depth=y_off,
                                                )
                                            elif tid == int(self.T_SHELF):
                                                self._draw_facade_furniture_silhouette(
                                                    surface,
                                                    kind="shelf",
                                                    x=int(px + 1),
                                                    floor_y=floor_y,
                                                    w=int(self.TILE_SIZE - 2),
                                                    max_h=max_h,
                                                    seed=fh,
                                                    depth=y_off,
                                                )
                                            else:
                                                self._draw_facade_furniture_silhouette(
                                                    surface,
                                                    kind="bed",
                                                    x=int(px + 1),
                                                    floor_y=floor_y,
                                                    w=int(self.TILE_SIZE - 2),
                                                    max_h=max_h,
                                                    seed=fh,
                                                    depth=y_off,
                                                )
                                            continue

                                            # Keep facade interior silhouettes readable even if minimap hides interiors.
                                            if tid == int(self.T_TABLE):
                                                c = (104, 82, 62)
                                            elif tid == int(self.T_SHELF):
                                                c = (92, 92, 104)
                                            else:
                                                c = (126, 126, 164)
                                            c = self._tint(c, add=(-56, -56, -56))
                                            bh = 3 if tid != int(self.T_BED) else 4
                                            rr = pygame.Rect(
                                                int(px + 1),
                                                int(open_rect.bottom - 2 - bh - y_off),
                                                max(1, int(self.TILE_SIZE - 2)),
                                                int(bh),
                                            )
                                            surface.fill(c, rr)
                                            pygame.draw.rect(surface, outline, rr, 1)
                                            if tid == int(self.T_SHELF) and rr.w >= 6 and rr.h >= 3:
                                                hh = int(self._hash2_u32(int(tx), int(sy), seed ^ 0x9E3779B9))
                                                g1 = (200, 86, 86) if ((hh >> 1) & 1) else (86, 210, 130)
                                                g2 = (210, 200, 120) if ((hh >> 2) & 1) else (140, 140, 220)
                                                surface.fill(g1, pygame.Rect(rr.x + 2, rr.y + 1, 1, 1))
                                                surface.fill(g2, pygame.Rect(rr.right - 3, rr.y + 1, 1, 1))

                            # Type accents for quick read.
                            if style == 5 and open_rect.w >= 22 and open_rect.h >= 7:
                                board = pygame.Rect(open_rect.x + 3, open_rect.y + 2, min(18, open_rect.w - 6), 4)
                                surface.fill((22, 34, 26), board)
                                pygame.draw.rect(surface, outline, board, 1)
                            if style == 3 and open_rect.w >= 18 and open_rect.h >= 7:
                                cx = open_rect.x + 8
                                cy = open_rect.y + 4
                                red = (190, 60, 60)
                                surface.fill(red, pygame.Rect(int(cx - 1), int(cy - 3), 2, 7))
                                surface.fill(red, pygame.Rect(int(cx - 3), int(cy - 1), 7, 2))
                                pygame.draw.rect(surface, outline, pygame.Rect(int(cx - 3), int(cy - 3), 7, 7), 1)

                            surface.set_clip(prev_clip)

                    # Re-draw the exterior door if our *front* facade covers it (south only).
                    if isinstance(door_px, pygame.Rect):
                        # Door visual should not scale with the full facade height (especially for high-rises).
                        # Anchor to the ground (bottom of facade) and clamp size by building type.
                        door_open_w = int(door_px.w)
                        if style == 1:  # house
                            door_w = int(clamp(door_open_w - 10, 10, 14))       
                        elif style in (2, 3, 4, 7):  # shop/hospital/prison/bookstore
                            door_w = int(clamp(door_open_w - 2, 16, 20))        
                        elif style in (5, 8):  # school/chinese
                            door_w = int(clamp(door_open_w - 4, 14, 18))        
                        elif style == 6:  # high-rise lobby
                            door_w = int(clamp(door_open_w - 4, 14, 18))        
                        else:
                            door_w = int(clamp(door_open_w - 6, 12, 18))        

                        if style == 6:
                            # High-rise lobby: keep it single-story (do not scale with the whole facade height).
                            door_h = int(clamp(16 + ((int(var) >> 2) % 3) - 1, 14, 18))
                        else:
                            door_h = int(clamp(10 + int(face_h) // 3, 12, 18))
                        door_h = int(min(int(door_h), max(10, int(south.h - 6))))

                        dr = pygame.Rect(0, 0, int(door_w), int(door_h))
                        # Door bottom line should align with the tile baseline (no "floating").
                        dr.midbottom = (int(door_px.centerx), int(south.bottom))
                        min_x = int(south.x + 2)
                        max_x = int(south.right - dr.w - 2)
                        if max_x >= min_x:
                            dr.x = int(clamp(int(dr.x), min_x, max_x))

                        if style == 2:
                            # Shop: glass sliding doors.
                            door_bg = self._tint(trim, add=(-78, -78, -84))     
                            pygame.draw.rect(surface, door_bg, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            hi = self._tint(trim, add=(48, 48, 54))
                            surface.fill(hi, pygame.Rect(dr.x + 2, dr.y + 3, max(1, dr.w - 4), 1))
                            mid = int(dr.centerx)
                            surface.fill(outline, pygame.Rect(mid, dr.y + 2, 1, max(1, dr.h - 4)))
                            handle = (230, 230, 236)
                            surface.fill(handle, pygame.Rect(mid - 3, dr.centery - 1, 1, 2))
                            surface.fill(handle, pygame.Rect(mid + 2, dr.centery - 1, 1, 2))
                        elif style == 7:
                            # Bookstore: glass door + small book decal.
                            door_bg = self._tint(trim, add=(-82, -82, -90))
                            pygame.draw.rect(surface, door_bg, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            hi = self._tint(trim, add=(52, 52, 56))
                            surface.fill(hi, pygame.Rect(dr.x + 2, dr.y + 3, max(1, dr.w - 4), 1))
                            mid = int(dr.centerx)
                            surface.fill(outline, pygame.Rect(mid, dr.y + 2, 1, max(1, dr.h - 4)))
                            ink = (232, 200, 120)
                            bx = int(dr.centerx)
                            by = int(dr.y + 5)
                            pygame.draw.rect(surface, ink, pygame.Rect(bx - 5, by, 4, 6), 1, border_radius=1)
                            pygame.draw.rect(surface, ink, pygame.Rect(bx + 2, by, 4, 6), 1, border_radius=1)
                            surface.fill(outline, pygame.Rect(bx, by, 1, 6))
                        elif style == 3:
                            # Hospital: glass door + red cross.
                            door_bg = self._tint(front, add=(-46, -46, -40))
                            pygame.draw.rect(surface, door_bg, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            hi = self._tint(trim, add=(52, 52, 56))
                            surface.fill(hi, pygame.Rect(dr.x + 2, dr.y + 3, max(1, dr.w - 4), 1))
                            mid = int(dr.centerx)
                            surface.fill(outline, pygame.Rect(mid, dr.y + 2, 1, max(1, dr.h - 4)))
                            red = (190, 60, 60)
                            cx = int(dr.x + min(10, max(6, dr.w // 2)))
                            cy = int(clamp(int(dr.y + dr.h // 2), int(dr.y + 4), int(dr.bottom - 4)))
                            surface.fill(red, pygame.Rect(cx - 1, cy - 3, 2, 7))
                            surface.fill(red, pygame.Rect(cx - 3, cy - 1, 7, 2))
                            pygame.draw.rect(surface, outline, pygame.Rect(cx - 3, cy - 3, 7, 7), 1)
                        elif style == 4:
                            # Prison: heavy gate + bars + caution band.
                            metal = self._tint(front, add=(-22, -22, -28))
                            pygame.draw.rect(surface, metal, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            bar = self._tint(trim, add=(-46, -46, -52))
                            for xx in range(int(dr.x + 2), int(dr.right - 2), 3):
                                surface.fill(bar, pygame.Rect(int(xx), int(dr.y + 2), 1, max(1, int(dr.h - 4))))
                            # Caution stripe at the top of the gate.
                            band = pygame.Rect(int(dr.x + 2), int(dr.y + 2), max(1, int(dr.w - 4)), 4)
                            a = (210, 190, 90)
                            bcol = (50, 50, 56)
                            for i in range(int(band.w)):
                                col = a if ((i + var) % 6) < 3 else bcol
                                surface.fill(col, pygame.Rect(int(band.x + i), int(band.y), 1, int(band.h)))
                            pygame.draw.rect(surface, outline, band, 1)
                        elif style == 5:
                            # School: wooden double doors.
                            wood = self._tint(front, add=(-36, -26, -18))
                            pygame.draw.rect(surface, wood, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            for xx in range(int(dr.x + 3), int(dr.right - 3), 4):
                                surface.fill(self._tint(wood, add=(18, 12, 6)), pygame.Rect(int(xx), int(dr.y + 2), 1, max(1, int(dr.h - 4))))
                            mid = int(dr.centerx)
                            surface.fill(outline, pygame.Rect(mid, dr.y + 2, 1, max(1, dr.h - 4)))
                            knob = (18, 18, 22)
                            surface.fill(knob, pygame.Rect(mid - 3, dr.centery, 1, 1))
                            surface.fill(knob, pygame.Rect(mid + 2, dr.centery, 1, 1))
                        elif style == 8:
                            # Chinese: red double door + gold studs.
                            red = self._tint(front, add=(10, -4, -6))
                            pygame.draw.rect(surface, red, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            mid = int(dr.centerx)
                            surface.fill(outline, pygame.Rect(mid, dr.y + 2, 1, max(1, dr.h - 4)))
                            gold = (210, 180, 110)
                            for yy in range(int(dr.y + 4), int(dr.bottom - 4), 4):
                                surface.fill(gold, pygame.Rect(int(mid - 3), int(yy), 1, 1))
                                surface.fill(gold, pygame.Rect(int(mid + 2), int(yy), 1, 1))
                            surface.fill(gold, pygame.Rect(int(mid - 3), int(dr.centery), 1, 1))
                            surface.fill(gold, pygame.Rect(int(mid + 2), int(dr.centery), 1, 1))
                        elif style == 1:
                            # Residential: simple wood door + small window.
                            wood = self._tint(front, add=(-34, -22, -14))
                            pygame.draw.rect(surface, wood, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            winr = pygame.Rect(int(dr.centerx - 3), int(dr.y + 4), 6, 4)
                            pygame.draw.rect(surface, self._tint(trim, add=(-30, -30, -34)), winr, border_radius=1)
                            pygame.draw.rect(surface, outline, winr, 1, border_radius=1)
                            pygame.draw.circle(surface, (18, 18, 22), (dr.right - 4, dr.centery), 1)
                            # Doormat.
                            mat = pygame.Rect(int(dr.x + 1), int(dr.bottom - 4), max(1, int(dr.w - 2)), 2)
                            surface.fill(self._tint(front, add=(-44, -44, -44)), mat)
                        elif style == 6:
                            # High-rise lobby: big glass door.
                            door_bg = self._tint(trim, add=(-86, -86, -92))
                            pygame.draw.rect(surface, door_bg, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            hi = self._tint(trim, add=(52, 52, 56))
                            surface.fill(hi, pygame.Rect(dr.x + 2, dr.y + 3, max(1, dr.w - 4), 1))
                            for xx in range(int(dr.x + 3), int(dr.right - 3), 4):
                                surface.fill(outline, pygame.Rect(int(xx), int(dr.y + 2), 1, max(1, int(dr.h - 4))))
                        else:
                            # Default: simple door.
                            door_bg = self._tint(front, add=(-34, -30, -24))
                            pygame.draw.rect(surface, door_bg, dr, border_radius=2)
                            pygame.draw.rect(surface, outline, dr, 1, border_radius=2)
                            surface.fill(self._tint(door_bg, add=(18, 14, 10)), pygame.Rect(dr.x + 2, dr.y + 2, max(1, dr.w - 4), 3))
                            pygame.draw.circle(surface, (18, 18, 22), (dr.right - 4, dr.y + dr.h // 2), 1)

                        # Threshold shadow for depth.
                        surface.fill(self._tint(front, add=(-26, -26, -30)), pygame.Rect(dr.x + 1, dr.bottom - 2, max(1, dr.w - 2), 1))

    def _draw_roofs(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
    ) -> None:
        ptx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
        pty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
        p_tile = self.world.peek_tile(ptx, pty)
        can_be_inside = p_tile in (self.T_FLOOR, self.T_DOOR)

        start_cx = start_tx // self.CHUNK_SIZE
        end_cx = end_tx // self.CHUNK_SIZE
        start_cy = start_ty // self.CHUNK_SIZE
        end_cy = end_ty // self.CHUNK_SIZE

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue
                for b in chunk.buildings:
                    tx0, ty0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    roof_kind = int(b[4]) if len(b) > 4 else 0
                    style, var = self._building_roof_style_var(int(roof_kind))
                    floors = int(b[5]) if len(b) > 5 else 0
                    face_h = int(self._building_face_height_px(style=style, w=w, h=h, var=var, floors=floors))
                    inside = bool(can_be_inside and (tx0 <= ptx < tx0 + w) and (ty0 <= pty < ty0 + h))
                    if inside:
                        # When the player is inside a building, hide that building's roof completely.
                        continue
                    alpha = 255
                    rw = max(1, int(w) - 2)
                    rh = max(1, int(h) - 2)
                    roof = self._roof_surface(rw, rh, alpha, roof_kind=roof_kind)
                    roof_draw = roof
                    if style == 6:
                        # High-rise: the facade is tall; clip a chunk off the
                        # roof bottom so the facade isn't covered by the roof.
                        cut = int(max(0, int(face_h) - 2))
                        min_vis = 18
                        max_cut = max(0, int(roof.get_height()) - int(min_vis))
                        cut = int(min(int(cut), int(max_cut)))
                        if cut > 0:
                            roof_draw = roof.subsurface(pygame.Rect(0, 0, int(roof.get_width()), int(roof.get_height() - cut)))

                    sx = (int(tx0) + 1) * self.TILE_SIZE - int(cam_x)
                    sy = (int(ty0) + 1) * self.TILE_SIZE - int(cam_y)
                    # Slight roof offset to reveal the "front" (bottom/right) walls like MiniDayZ.
                    roof_off_x = -2
                    roof_off_y = -2
                    rx = int(sx + roof_off_x)
                    ry = int(sy + roof_off_y)
                    if rx > INTERNAL_W or ry > INTERNAL_H:
                        continue
                    if rx + roof_draw.get_width() < 0 or ry + roof_draw.get_height() < 0:
                        continue
                    if not inside:
                        sh = pygame.Surface((roof_draw.get_width(), roof_draw.get_height()), pygame.SRCALPHA)
                        sh.fill((0, 0, 0, 70))
                        surface.blit(sh, (int(rx + 2), int(ry + 2)))
                    surface.blit(roof_draw, (int(rx), int(ry)))

    def _draw_zombies(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        for z in self.zombies:
            p = pygame.Vector2(z.pos) - pygame.Vector2(cam_x, cam_y)
            sx = iround(float(p.x))
            sy = iround(float(p.y))

            kind = str(getattr(z, "kind", "walker"))
            by_kind = self._MONSTER_FRAMES.get(kind) or self._MONSTER_FRAMES.get("walker")
            if not by_kind:
                continue
            d = str(getattr(z, "dir", "down"))
            frames = by_kind.get(d) or by_kind.get("down")
            if not frames:
                continue

            moving = z.vel.length_squared() > 1.0
            idx = 0 if not moving else 1 + (int(float(getattr(z, "anim", 0.0))) % 2)
            idx = max(0, min(int(idx), len(frames) - 1))
            spr = frames[idx]

            # Wobble when walking: visual-only sway (doesn't affect collisions).
            wob_x = 0
            wob_y = 0
            if moving:
                phase = float(getattr(z, "anim", 0.0))
                if kind == "walker":
                    freq = 2.2
                    sway_amp = 2.2
                    bob_amp = 1.2
                elif kind == "screamer":
                    freq = 2.4
                    sway_amp = 2.0
                    bob_amp = 1.1
                else:
                    freq = 2.8
                    sway_amp = 1.4
                    bob_amp = 0.8
                sway = int(round(math.sin(phase * float(freq)) * float(sway_amp)))
                bob = int(round(math.cos(phase * float(freq)) * float(bob_amp)))
                if d in ("left", "right"):
                    wob_y = int(sway)
                    wob_x = int(bob)
                else:
                    wob_x = int(sway)
                    wob_y = int(bob)
            sx2 = int(sx + wob_x)
            sy2 = int(sy + wob_y)

            rect = spr.get_rect()
            rect.midbottom = (int(sx2), iround(float(sy2) + float(z.h) / 2.0))

            shadow = pygame.Rect(0, 0, max(6, rect.w - 4), 4)
            shadow.center = (int(sx2), iround(float(sy2) + float(z.h) / 2.0 - 1.0))
            pygame.draw.ellipse(surface, (0, 0, 0), shadow)
            surface.blit(spr, rect)

            if self._debug:
                mdef = self._MONSTER_DEFS.get(kind, self._MONSTER_DEFS["walker"])
                max_hp = max(1, int(mdef.hp) + 6)
                hp = int(clamp(int(z.hp), 0, max_hp))
                bar = pygame.Rect(rect.left, rect.top - 4, rect.w, 3)
                pygame.draw.rect(surface, (10, 10, 14), bar)
                fill_w = int((bar.w - 1) * (hp / max(1, max_hp)))
                if fill_w > 0:
                    pygame.draw.rect(surface, (220, 90, 90), pygame.Rect(bar.x + 1, bar.y + 1, fill_w, bar.h - 1))

    def _draw_bullets(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        for b in self.bullets:
            p = pygame.Vector2(b.pos) - pygame.Vector2(cam_x, cam_y)
            x = iround(float(p.x))
            y = iround(float(p.y))
            if 0 <= x < INTERNAL_W and 0 <= y < INTERNAL_H:
                surface.set_at((x, y), (250, 250, 250))

    def _draw_inventory_ui(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        surface.blit(overlay, (0, 0))

        cols = max(1, int(self.inventory.cols))
        rows = int(math.ceil(len(self.inventory.slots) / cols))
        slot = 30
        gap = 4
        grid_w = cols * slot + (cols - 1) * gap
        grid_h = rows * slot + (rows - 1) * gap
        x0 = (INTERNAL_W - grid_w) // 2
        y0 = (INTERNAL_H - grid_h) // 2 - 10

        panel = pygame.Rect(x0 - 10, y0 - 26, grid_w + 20, grid_h + 58)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=10)
        pygame.draw.rect(surface, (80, 80, 96), panel, 2, border_radius=10)
        draw_text(surface, self.app.font_m, "背包", (panel.centerx, panel.top + 14), pygame.Color(240, 240, 240), anchor="center")

        for i, st in enumerate(self.inventory.slots):
            cx = i % cols
            cy = i // cols
            r = pygame.Rect(x0 + cx * (slot + gap), y0 + cy * (slot + gap), slot, slot)
            selected = i == int(self.inv_index)
            bg = (44, 44, 52) if selected else (26, 26, 32)
            border = (220, 220, 240) if selected else (90, 90, 110)
            pygame.draw.rect(surface, bg, r, border_radius=6)
            pygame.draw.rect(surface, border, r, 2, border_radius=6)
            if st is None:
                continue
            img: pygame.Surface | None = None
            spr = self._ITEM_SPRITES.get(st.item_id)
            if spr is not None:
                factor = 2 if (spr.get_width() * 2 <= r.w - 2 and spr.get_height() * 2 <= r.h - 2) else 1
                if factor != 1:
                    img = pygame.transform.scale(spr, (spr.get_width() * factor, spr.get_height() * factor))
                else:
                    img = spr
            else:
                icon = self._ITEM_ICONS.get(st.item_id)
                if icon is None:
                    idef = self._ITEMS.get(st.item_id)
                    col = idef.color if idef is not None else (255, 0, 255)
                    icon_rect = pygame.Rect(0, 0, 10, 10)
                    icon_rect.center = r.center
                    pygame.draw.rect(surface, col, icon_rect)
                else:
                    img = icon
                    if img.get_width() <= 8:
                        img = pygame.transform.scale(img, (img.get_width() * 2, img.get_height() * 2))
            if img is not None:
                surface.blit(img, img.get_rect(center=r.center))
            draw_text(surface, self.app.font_s, str(int(st.qty)), (r.right - 2, r.bottom - 2), pygame.Color(240, 240, 240), anchor="topright")

        sel = self.inventory.slots[int(self.inv_index)]
        if sel is None:
            desc = "空"
        else:
            idef = self._ITEMS.get(sel.item_id)
            name = idef.name if idef is not None else sel.item_id
            desc = f"{name} x{int(sel.qty)}"
        draw_text(surface, self.app.font_s, desc, (panel.centerx, panel.bottom - 22), pygame.Color(200, 200, 210), anchor="center")
        draw_text(surface, self.app.font_s, "方向键/WASD移动格子  Enter使用/装备  Q丢弃  Tab关闭", (panel.centerx, panel.bottom - 8), pygame.Color(160, 160, 175), anchor="center")

    def _draw_sprite_gallery_ui(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        surface.blit(overlay, (0, 0))

        panel = pygame.Rect(10, 10, INTERNAL_W - 20, INTERNAL_H - 20)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        page = int(getattr(self, "_gallery_page", 0)) % 2
        title = "Sprite Gallery: Cars" if page == 0 else "Sprite Gallery: Bike"
        draw_text(surface, self.app.font_m, title, (panel.centerx, panel.top + 12), pygame.Color(240, 240, 240), anchor="center")

        if page == 0:
            ids = list(self._CAR_MODELS.keys())
            if not ids:
                draw_text(surface, self.app.font_s, "No car models", (panel.centerx, panel.centery), pygame.Color(200, 200, 210), anchor="center")
            else:
                maxw = 1
                maxh = 1
                for mid in ids:
                    base = self._CAR_BASE.get((str(mid), 0, 0))
                    if base is None:
                        continue
                    maxw = max(maxw, int(base.get_width()))
                    maxh = max(maxh, int(base.get_height()))

                scale = 3
                cols = 3
                cell_pad = 10
                label_h = 14
                cell_w = maxw * scale + cell_pad
                cell_h = maxh * scale + label_h + cell_pad
                rows = int((len(ids) + cols - 1) // cols)
                grid_w = cols * cell_w
                grid_h = rows * cell_h
                x0 = int(panel.centerx - grid_w // 2)
                y0 = int(panel.top + 30)
                if y0 + grid_h > panel.bottom - 16:
                    y0 = max(int(panel.top + 30), int(panel.bottom - 16 - grid_h))

                for idx, mid in enumerate(ids):
                    base = self._CAR_BASE.get((str(mid), 0, 0))
                    if base is None:
                        continue
                    spr = pygame.transform.scale(base, (int(base.get_width()) * scale, int(base.get_height()) * scale))
                    cx = idx % cols
                    cy = idx // cols
                    cell_x = x0 + cx * cell_w
                    cell_y = y0 + cy * cell_h
                    bg = pygame.Rect(cell_x + 4, cell_y + 2, cell_w - 8, maxh * scale + 4)
                    pygame.draw.rect(surface, (24, 24, 30), bg, border_radius=10)
                    pygame.draw.rect(surface, (50, 50, 64), bg, 1, border_radius=10)
                    x = cell_x + (cell_w - spr.get_width()) // 2
                    y = cell_y + 4
                    surface.blit(spr, (x, y))
                    draw_text(surface, self.app.font_s, str(mid), (cell_x + cell_w // 2, cell_y + maxh * scale + 8), pygame.Color(200, 200, 210), anchor="midtop")
        else:
            dirs = ["up", "down", "left", "right"]
            scale = 5
            cell = 18 * scale
            x0 = int(panel.centerx - (len(dirs) * cell) // 2)
            y0 = int(panel.top + 34)

            for di, d in enumerate(dirs):
                for fi in range(2):
                    b = self._BIKE_FRAMES.get(d, self._BIKE_FRAMES["right"])[fi]
                    rider_frames = getattr(self, "cyclist_frames", getattr(self, "_CYCLIST_FRAMES", {})).get(d)
                    if rider_frames is None:
                        pf = getattr(self, "player_frames", self._PLAYER_FRAMES)
                        rider_frames = pf.get(d, pf["down"])[:2]
                    r = rider_frames[fi % len(rider_frames)]

                    bs = pygame.transform.scale(b, (int(b.get_width()) * scale, int(b.get_height()) * scale))
                    rs = pygame.transform.scale(r, (int(r.get_width()) * scale, int(r.get_height()) * scale))
                    x = x0 + di * cell + (cell - bs.get_width()) // 2
                    y = y0 + fi * cell + (cell - bs.get_height()) // 2
                    surface.blit(bs, (x, y))
                    rr = rs.get_rect()
                    off = (5 if d in ("up", "down") else 4) * scale
                    rr.midbottom = (x + bs.get_width() // 2, y + bs.get_height() // 2 + off)
                    surface.blit(rs, rr)

                draw_text(surface, self.app.font_s, d, (x0 + di * cell + cell // 2, y0 + 2 * cell - 10), pygame.Color(200, 200, 210), anchor="center")

        draw_text(surface, self.app.font_s, "Tab/←/→ 翻页 | F2/Esc 关闭", (panel.centerx, panel.bottom - 8), pygame.Color(160, 160, 175), anchor="center")

    def _draw_rv_interior_ui(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)     
        overlay.fill((0, 0, 0, 190))
        surface.blit(overlay, (0, 0))

        panel = pygame.Rect(16, 14, INTERNAL_W - 32, INTERNAL_H - 28)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        draw_text(surface, self.app.font_m, "房车内部", (panel.centerx, panel.top + 14), pygame.Color(240, 240, 240), anchor="center")

        # Floor plan (tile map).
        tile_px = 14
        map_w = int(self._RV_INT_W) * tile_px
        map_h = int(self._RV_INT_H) * tile_px
        map_x = int(panel.centerx - map_w // 2)
        map_y = int(panel.top + 28)
        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        pygame.draw.rect(surface, (24, 24, 30), map_rect.inflate(8, 8), border_radius=10)
        pygame.draw.rect(surface, (60, 60, 76), map_rect.inflate(8, 8), 2, border_radius=10)

        layout = getattr(self, "rv_int_layout", self._RV_INT_LAYOUT)
        for y, row in enumerate(layout):
            for x, ch in enumerate(row[: int(self._RV_INT_W)]):
                tid = int(self._RV_INT_LEGEND.get(ch, int(self.T_FLOOR)))
                tdef = self._TILES.get(tid)
                col = tdef.color if tdef is not None else (255, 0, 255)
                if tid == int(self.T_FLOOR):
                    col = (62, 60, 56)
                elif tid == int(self.T_WALL):
                    col = (22, 22, 26)
                r = pygame.Rect(map_x + x * tile_px, map_y + y * tile_px, tile_px, tile_px)
                pygame.draw.rect(surface, col, r)
                pygame.draw.rect(surface, (10, 10, 14), r, 1)
                # Tiny glyph for special tiles.
                if ch == "D":
                    pygame.draw.line(surface, (240, 220, 140), (r.left + 3, r.centery), (r.right - 3, r.centery), 2)
                elif ch == "B":
                    pygame.draw.rect(surface, (240, 240, 240), pygame.Rect(r.left + 3, r.top + 3, r.w - 6, 3))
                elif ch == "S":
                    pygame.draw.rect(surface, (200, 200, 210), pygame.Rect(r.left + 3, r.top + 3, r.w - 6, r.h - 6), 1)
                elif ch == "T":
                    pygame.draw.rect(surface, (12, 12, 16), pygame.Rect(r.left + 3, r.top + 5, r.w - 6, r.h - 10))
                elif ch == "V":
                    g = pygame.Rect(r.left + 3, r.top + 4, r.w - 6, r.h - 8)
                    pygame.draw.rect(surface, (90, 140, 190), g, 1)
                    pygame.draw.line(surface, (180, 210, 240), (g.left + 1, g.top + 1), (g.right - 2, g.top + 1), 1)

        focus = str(getattr(self, "rv_ui_focus", "map"))
        cx, cy = self.rv_ui_map
        if 0 <= int(cx) < int(self._RV_INT_W) and 0 <= int(cy) < int(self._RV_INT_H):
            cur_r = pygame.Rect(map_x + int(cx) * tile_px, map_y + int(cy) * tile_px, tile_px, tile_px)
            col = (240, 240, 240) if focus == "map" else (130, 130, 150)
            pygame.draw.rect(surface, col, cur_r, 2)

        # Inventories.
        slot = 22
        gap = 4

        pcols = max(1, int(self.inventory.cols))
        prows = int(math.ceil(len(self.inventory.slots) / pcols))
        pgrid_w = pcols * slot + (pcols - 1) * gap
        pgrid_h = prows * slot + (prows - 1) * gap

        scols = max(1, int(self.rv_storage.cols))
        srows = int(math.ceil(len(self.rv_storage.slots) / scols))
        sgrid_w = scols * slot + (scols - 1) * gap
        sgrid_h = srows * slot + (srows - 1) * gap

        inv_y = map_rect.bottom + 22
        left_x = panel.left + 14
        right_x = panel.right - 14 - sgrid_w

        draw_text(surface, self.app.font_s, "背包", (left_x, inv_y - 14), pygame.Color(200, 200, 210), anchor="topleft")
        draw_text(surface, self.app.font_s, "房车储物", (right_x, inv_y - 14), pygame.Color(200, 200, 210), anchor="topleft")

        def draw_inv(inv: HardcoreSurvivalState._Inventory, *, x0: int, y0: int, cols: int, sel_idx: int, focused: bool) -> None:
            for i, st in enumerate(inv.slots):
                cx = i % cols
                cy = i // cols
                r = pygame.Rect(x0 + cx * (slot + gap), y0 + cy * (slot + gap), slot, slot)
                selected = focused and i == int(sel_idx)
                bg = (44, 44, 52) if selected else (26, 26, 32)
                border = (220, 220, 240) if selected else (90, 90, 110)
                pygame.draw.rect(surface, bg, r, border_radius=6)
                pygame.draw.rect(surface, border, r, 2, border_radius=6)
                if st is None:
                    continue
                img: pygame.Surface | None = None
                spr = self._ITEM_SPRITES.get(st.item_id)
                if spr is not None:
                    img = spr
                    max_w = r.w - 4
                    max_h = r.h - 4
                    if img.get_width() > max_w or img.get_height() > max_h:
                        scale = min(max_w / max(1, img.get_width()), max_h / max(1, img.get_height()))
                        iw = max(1, int(round(img.get_width() * scale)))
                        ih = max(1, int(round(img.get_height() * scale)))
                        img = pygame.transform.scale(img, (iw, ih))
                else:
                    icon = self._ITEM_ICONS.get(st.item_id)
                    if icon is None:
                        idef = self._ITEMS.get(st.item_id)
                        col = idef.color if idef is not None else (255, 0, 255)
                        icon_rect = pygame.Rect(0, 0, 10, 10)
                        icon_rect.center = r.center
                        pygame.draw.rect(surface, col, icon_rect)
                    else:
                        img = icon
                        if img.get_width() <= 8:
                            img = pygame.transform.scale(img, (img.get_width() * 2, img.get_height() * 2))
                if img is not None:
                    surface.blit(img, img.get_rect(center=r.center))
                draw_text(surface, self.app.font_s, str(int(st.qty)), (r.right - 2, r.bottom - 2), pygame.Color(240, 240, 240), anchor="topright")

        draw_inv(self.inventory, x0=left_x, y0=inv_y, cols=pcols, sel_idx=int(self.rv_ui_player_index), focused=(focus == "player"))
        draw_inv(self.rv_storage, x0=right_x, y0=inv_y, cols=scols, sel_idx=int(self.rv_ui_storage_index), focused=(focus == "storage"))

        # Description / help.
        desc = ""
        if focus == "map":
            ch = self._rv_ui_char_at(int(cx), int(cy))
            if ch == "D":
                desc = "门：Enter 离开"
            elif ch == "B":
                desc = "床：Enter 休息（体力恢复/心态提升）"
            elif ch == "S":
                desc = "储物柜：Enter 打开（转移物品）"
            elif ch == "T":
                desc = "工作台：占位（后续做加工/修理）"
            elif ch == "C":
                desc = "驾驶舱：占位"
            else:
                desc = "走动区域"
        elif focus == "player":
            st = self.inventory.slots[int(self.rv_ui_player_index)]
            if st is None:
                desc = "背包：空"
            else:
                idef = self._ITEMS.get(st.item_id)
                name = idef.name if idef is not None else st.item_id
                desc = f"背包：{name} x{int(st.qty)}（Enter 转移到储物）"
        else:
            st = self.rv_storage.slots[int(self.rv_ui_storage_index)]
            if st is None:
                desc = "储物：空"
            else:
                idef = self._ITEMS.get(st.item_id)
                name = idef.name if idef is not None else st.item_id
                desc = f"储物：{name} x{int(st.qty)}（Enter 转移到背包）"

        if self.rv_ui_status:
            draw_text(surface, self.app.font_s, self.rv_ui_status, (panel.centerx, panel.bottom - 24), pygame.Color(255, 220, 140), anchor="center")
        draw_text(surface, self.app.font_s, desc, (panel.centerx, panel.bottom - 12), pygame.Color(200, 200, 210), anchor="center")
        draw_text(surface, self.app.font_s, "Tab切换区域 | Enter互动/转移 | Esc关闭", (panel.centerx, panel.bottom - 2), pygame.Color(160, 160, 175), anchor="center")

    def _draw_home_storage_ui(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        surface.blit(overlay, (0, 0))

        panel = pygame.Rect(16, 14, INTERNAL_W - 32, INTERNAL_H - 28)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        storage = self._home_ui_storage_inv()
        storage_label = self._home_ui_storage_label()
        draw_text(surface, self.app.font_m, self._home_ui_storage_title(), (panel.centerx, panel.top + 14), pygame.Color(240, 240, 240), anchor="center")

        focus = str(getattr(self, "home_ui_focus", "storage"))
        slot = 22
        gap = 4

        pcols = max(1, int(self.inventory.cols))
        prows = int(math.ceil(len(self.inventory.slots) / pcols))
        pgrid_w = pcols * slot + (pcols - 1) * gap
        pgrid_h = prows * slot + (prows - 1) * gap

        scols = max(1, int(storage.cols))
        srows = int(math.ceil(len(storage.slots) / scols))
        sgrid_w = scols * slot + (scols - 1) * gap
        sgrid_h = srows * slot + (srows - 1) * gap

        inv_y = panel.top + 46
        left_x = panel.left + 14
        right_x = panel.right - 14 - sgrid_w

        content_h = max(pgrid_h, sgrid_h) + 20
        inv_y = int(clamp(inv_y, panel.top + 42, panel.bottom - 52 - content_h))

        draw_text(surface, self.app.font_s, "背包", (left_x, inv_y - 14), pygame.Color(200, 200, 210), anchor="topleft")
        draw_text(surface, self.app.font_s, storage_label, (right_x, inv_y - 14), pygame.Color(200, 200, 210), anchor="topleft")

        def draw_inv(
            inv: HardcoreSurvivalState._Inventory,
            *,
            x0: int,
            y0: int,
            cols: int,
            sel_idx: int,
            focused: bool,
        ) -> None:
            for i, st in enumerate(inv.slots):
                cx = i % cols
                cy = i // cols
                r = pygame.Rect(x0 + cx * (slot + gap), y0 + cy * (slot + gap), slot, slot)
                selected = focused and i == int(sel_idx)
                bg = (44, 44, 52) if selected else (26, 26, 32)
                border = (220, 220, 240) if selected else (90, 90, 110)
                pygame.draw.rect(surface, bg, r, border_radius=6)
                pygame.draw.rect(surface, border, r, 2, border_radius=6)
                if st is None:
                    continue
                img: pygame.Surface | None = None
                spr = self._ITEM_SPRITES.get(st.item_id)
                if spr is not None:
                    img = spr
                    max_w = r.w - 4
                    max_h = r.h - 4
                    if img.get_width() > max_w or img.get_height() > max_h:
                        scale = min(max_w / max(1, img.get_width()), max_h / max(1, img.get_height()))
                        iw = max(1, int(round(img.get_width() * scale)))
                        ih = max(1, int(round(img.get_height() * scale)))
                        img = pygame.transform.scale(img, (iw, ih))
                else:
                    icon = self._ITEM_ICONS.get(st.item_id)
                    if icon is None:
                        idef = self._ITEMS.get(st.item_id)
                        col = idef.color if idef is not None else (255, 0, 255)
                        icon_rect = pygame.Rect(0, 0, 10, 10)
                        icon_rect.center = r.center
                        pygame.draw.rect(surface, col, icon_rect)
                    else:
                        img = icon
                        if img.get_width() <= 8:
                            img = pygame.transform.scale(img, (img.get_width() * 2, img.get_height() * 2))
                if img is not None:
                    surface.blit(img, img.get_rect(center=r.center))
                draw_text(surface, self.app.font_s, str(int(st.qty)), (r.right - 2, r.bottom - 2), pygame.Color(240, 240, 240), anchor="topright")

        draw_inv(
            self.inventory,
            x0=left_x,
            y0=inv_y,
            cols=pcols,
            sel_idx=int(getattr(self, "home_ui_player_index", 0)),
            focused=(focus == "player"),
        )
        draw_inv(
            storage,
            x0=right_x,
            y0=inv_y,
            cols=scols,
            sel_idx=int(getattr(self, "home_ui_storage_index", 0)),
            focused=(focus == "storage"),
        )

        desc = ""
        if focus == "player":
            st = self.inventory.slots[int(getattr(self, "home_ui_player_index", 0))]
            if st is None:
                desc = "背包：空"
            else:
                idef = self._ITEMS.get(st.item_id)
                name = idef.name if idef is not None else st.item_id
                desc = f"背包: {name} x{int(st.qty)} (Enter → {storage_label})"
        else:
            st = storage.slots[int(getattr(self, "home_ui_storage_index", 0))]
            if st is None:
                desc = f"{storage_label}：空"
            else:
                idef = self._ITEMS.get(st.item_id)
                name = idef.name if idef is not None else st.item_id
                desc = f"{storage_label}: {name} x{int(st.qty)} (Enter → 背包)"

        if getattr(self, "home_ui_status", ""):
            draw_text(surface, self.app.font_s, str(self.home_ui_status), (panel.centerx, panel.bottom - 24), pygame.Color(255, 220, 140), anchor="center")
        draw_text(surface, self.app.font_s, desc, (panel.centerx, panel.bottom - 12), pygame.Color(200, 200, 210), anchor="center")
        draw_text(surface, self.app.font_s, f"Tab 切换 背包/{storage_label} | Enter 转移 | Esc 关闭", (panel.centerx, panel.bottom - 2), pygame.Color(160, 160, 175), anchor="center")

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((0, 0, 0))
        if getattr(self, "sch_interior", False):
            self._hover_tooltip = None
            self._draw_sch_interior_scene(surface)
            self._draw_day_night_overlay(surface, in_rv=True)
            self._draw_survival_ui(surface)
            if getattr(self, "world_map_open", False):
                self._draw_world_map_ui(surface)
                return
            self._draw_hover_tooltip(surface)
            if getattr(self, "sch_elevator_ui_open", False):
                self._draw_sch_elevator_ui(surface)
            elif self.inv_open:
                self._draw_inventory_ui(surface)
            if getattr(self, "_gallery_open", False):
                self._draw_sprite_gallery_ui(surface)
            return
        if getattr(self, "hr_interior", False):
            self._hover_tooltip = None
            self._draw_hr_interior_scene(surface)
            self._draw_day_night_overlay(surface, in_rv=True)
            self._draw_survival_ui(surface)
            if getattr(self, "world_map_open", False):
                self._draw_world_map_ui(surface)
                return
            if getattr(self, "home_ui_open", False):
                self._draw_home_storage_ui(surface)
                return
            self._draw_hover_tooltip(surface)
            if getattr(self, "hr_elevator_ui_open", False):
                self._draw_hr_elevator_ui(surface)
            elif self.inv_open:
                self._draw_inventory_ui(surface)
            if getattr(self, "_gallery_open", False):
                self._draw_sprite_gallery_ui(surface)
            return
        if getattr(self, "rv_interior", False):
            self._hover_tooltip = None
            self._draw_rv_interior_scene(surface)
            self._draw_day_night_overlay(surface, in_rv=True)
            self._draw_weather_effects(surface, in_rv=True)
            self._draw_survival_ui(surface)
            if getattr(self, "world_map_open", False):
                self._draw_world_map_ui(surface)
                return
            self._draw_hover_tooltip(surface)
            if getattr(self, "rv_ui_open", False):
                self._draw_rv_interior_ui(surface)
            elif self.inv_open:
                self._draw_inventory_ui(surface)
            if getattr(self, "_gallery_open", False):
                self._draw_sprite_gallery_ui(surface)
            return
        cam_x = int(getattr(self, "cam_x", 0))
        cam_y = int(getattr(self, "cam_y", 0))

        start_tx = int(math.floor(cam_x / self.TILE_SIZE)) - 1
        start_ty = int(math.floor(cam_y / self.TILE_SIZE)) - 1
        end_tx = int(math.floor((cam_x + INTERNAL_W) / self.TILE_SIZE)) + 1     
        end_ty = int(math.floor((cam_y + INTERNAL_H) / self.TILE_SIZE)) + 1     

        # Hover tooltip is populated during world draws (items / vehicles).
        self._hover_tooltip = None

        for ty in range(start_ty, end_ty + 1):
            py = ty * self.TILE_SIZE - cam_y
            if py >= INTERNAL_H:
                break
            for tx in range(start_tx, end_tx + 1):
                px = tx * self.TILE_SIZE - cam_x
                if px >= INTERNAL_W:
                    break
                tile = self.world.peek_tile(tx, ty)
                rect = pygame.Rect(int(px), int(py), self.TILE_SIZE, self.TILE_SIZE)
                self._draw_world_tile(surface, rect, tx=tx, ty=ty, tile_id=int(tile))

        # Include a small off-screen margin so facade extrusions (face_h) don't pop at screen edges.
        self._draw_building_facades(surface, cam_x, cam_y, start_tx - 3, end_tx + 3, start_ty - 3, end_ty + 3)
        self._draw_vehicle_props(surface, cam_x, cam_y, start_tx, end_tx, start_ty, end_ty)
        self._draw_world_props(surface, cam_x, cam_y, start_tx, end_tx, start_ty, end_ty)
        self._draw_world_items(surface, cam_x, cam_y, start_tx, end_tx, start_ty, end_ty)
        self._draw_rv(surface, cam_x, cam_y)
        self._draw_bike(surface, cam_x, cam_y)
        self._draw_zombies(surface, cam_x, cam_y)
        self._draw_bullets(surface, cam_x, cam_y)
        if self.mount is None:
            self._draw_player(surface, cam_x, cam_y)
        # Occlude the mounted vehicle by front facades when it is behind buildings.
        self._occlude_mounted_vehicle_with_facades(surface, cam_x, cam_y, start_tx - 3, end_tx + 3, start_ty - 3, end_ty + 3)
        self._draw_roofs(surface, cam_x, cam_y, start_tx, end_tx, start_ty, end_ty)
        self._draw_day_night_overlay(surface, in_rv=False)
        self._draw_weather_effects(surface, in_rv=False)
        self._draw_survival_ui(surface)
        if getattr(self, "world_map_open", False):
            self._draw_world_map_ui(surface)
            return

        self._draw_hover_tooltip(surface)

        if self._debug:
            self._draw_debug(surface, cam_x, cam_y)
        if getattr(self, "rv_ui_open", False):
            self._draw_rv_interior_ui(surface)
        elif self.inv_open:
            self._draw_inventory_ui(surface)
        if getattr(self, "_gallery_open", False):
            self._draw_sprite_gallery_ui(surface)

    def _occlude_mounted_vehicle_with_facades(
        self,
        surface: pygame.Surface,
        cam_x: int,
        cam_y: int,
        start_tx: int,
        end_tx: int,
        start_ty: int,
        end_ty: int,
    ) -> None:
        mount = getattr(self, "mount", None)
        if mount == "bike":
            if not hasattr(self, "bike") or self.bike is None:
                return
            w = int(getattr(self.bike, "w", 14))
            h = int(getattr(self.bike, "h", 10))
            rect = self.bike.rect().move(-int(cam_x), -int(cam_y)).inflate(36, 46)
            ground_y = int(round((float(self.bike.pos.y) - float(cam_y)) + float(h) / 2.0))
        elif mount == "rv":
            if not hasattr(self, "rv") or self.rv is None:
                return
            rect = self.rv.rect().move(-int(cam_x), -int(cam_y)).inflate(72, 62)
            ground_y = int(round((float(self.rv.pos.y) - float(cam_y)) + float(self.rv.h) / 2.0))
        else:
            return

        clip = rect.clip(pygame.Rect(0, 0, INTERNAL_W, INTERNAL_H))
        if clip.w <= 0 or clip.h <= 0:
            return
        prev_clip = surface.get_clip()
        surface.set_clip(clip)
        try:
            self._draw_building_facades(
                surface,
                int(cam_x),
                int(cam_y),
                int(start_tx),
                int(end_tx),
                int(start_ty),
                int(end_ty),
                min_ground_y=int(ground_y) + 1,
            )
        finally:
            surface.set_clip(prev_clip)

    def _draw_player(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        p = self.player.pos - pygame.Vector2(cam_x, cam_y)
        px = iround(float(p.x))
        py = iround(float(p.y))

        face = pygame.Vector2(self.player.facing)
        if self.gun is not None:
            face = pygame.Vector2(self.aim_dir)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)

        prev_d = str(getattr(self.player, "dir", "down"))
        ax = abs(float(face.x))
        ay = abs(float(face.y))
        bias = 1.12
        if ax <= 1e-6 and ay <= 1e-6:
            d = prev_d
        elif prev_d in ("up", "down"):
            # Stick to vertical unless X is clearly dominant.
            if ax > ay * float(bias):
                d = "right" if float(face.x) >= 0.0 else "left"
            else:
                d = "down" if float(face.y) >= 0.0 else "up"
        elif prev_d in ("left", "right"):
            # Stick to horizontal unless Y is clearly dominant.
            if ay > ax * float(bias):
                d = "down" if float(face.y) >= 0.0 else "up"
            else:
                d = "right" if float(face.x) >= 0.0 else "left"
        else:
            # Fallback: choose the dominant axis.
            if ay >= ax:
                d = "down" if float(face.y) >= 0.0 else "up"
            else:
                d = "right" if float(face.x) >= 0.0 else "left"
        self.player.dir = d

        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = pf.get(d, pf["down"])
        moving = self.player.vel.length_squared() > 1.0
        bob = 0
        walk_idx = 0
        idle_anim = True
        if not moving or len(frames) <= 1:
            spr = frames[0]
        else:
            idle_anim = False
            walk = frames[1:]
            phase = (float(self.player.walk_phase) % math.tau) / math.tau
            idx = int(phase * len(walk)) % len(walk)
            walk_idx = int(idx)
            spr = walk[idx]
            if is_run and idx in (1, 4):
                bob = 1
        rect = spr.get_rect()
        rect.midbottom = (px, iround(float(p.y) + float(self.player.h) / 2.0))
        rect.y += bob
        surface.blit(spr, rect)

        if self.gun is not None:
            aim = pygame.Vector2(self.aim_dir)
            if aim.length_squared() <= 0.001:
                aim = pygame.Vector2(1, 0)
            aim = aim.normalize()

            height_delta = 0
            av = getattr(self, "avatar", None)
            if av is not None:
                hidx = int(getattr(av, "height", 1))
                height_delta = 1 if hidx == 0 else -1 if hidx == 2 else 0
            sk = self._survivor_skeleton_nodes(
                d,
                int(walk_idx),
                idle=bool(idle_anim),
                height_delta=int(height_delta),
                run=bool(is_run),
            )
            hand_node = sk.get("r_hand")
            if hand_node is not None:
                hx, hy = hand_node
                base_hand = pygame.Vector2(int(rect.left + hx), int(rect.top + hy))
            else:
                base_hand = pygame.Vector2(rect.centerx, rect.centery + 3)
            hand = base_hand + aim * 4.0
            base = self._GUN_HAND_SPRITES.get(self.gun.gun_id)
            if base is not None:
                flip = float(aim.x) < 0.0
                if flip:
                    base = flip_x_pixel_sprite(base)
                deg = -math.degrees(math.atan2(float(aim.y), float(aim.x)))
                if flip:
                    deg -= 180.0
                if float(self.gun.reload_left) > 0.0 and float(self.gun.reload_total) > 0.0:
                    p = 1.0 - float(self.gun.reload_left) / max(1e-6, float(self.gun.reload_total))
                    deg += float(p) * 360.0
                gun_spr = rotate_pixel_sprite(base, deg, step_deg=5.0)
                grect = gun_spr.get_rect(center=(int(round(hand.x)), int(round(hand.y))))
                surface.blit(gun_spr, grect)
            else:
                tip = hand + aim * 12.0
                col = (240, 240, 240) if self.gun.reload_left <= 0.0 else (200, 200, 220)
                pygame.draw.line(surface, col, (int(hand.x), int(hand.y)), (int(tip.x), int(tip.y)), 2)

            if getattr(self, "muzzle_flash_left", 0.0) > 0.0 and self.gun.reload_left <= 0.0:
                muzzle = hand + aim * 13.0
                r = 3 if self.muzzle_flash_left > 0.03 else 2
                pygame.draw.circle(surface, (255, 240, 190), (int(round(muzzle.x)), int(round(muzzle.y))), r)
                pygame.draw.circle(surface, (255, 200, 120), (int(round(muzzle.x)), int(round(muzzle.y))), max(1, r - 1))

    def _draw_sun_moon_widget(self, surface: pygame.Surface, *, tday: float) -> None:
        rect = pygame.Rect(0, 0, 116, 22)
        rect.topright = (INTERNAL_W - 6, 34)
        pygame.draw.rect(surface, (18, 18, 22), rect, border_radius=8)
        pygame.draw.rect(surface, (70, 70, 86), rect, 2, border_radius=8)

        horizon_y = rect.bottom - 6
        pygame.draw.line(surface, (40, 40, 50), (rect.left + 6, horizon_y), (rect.right - 7, horizon_y), 1)

        amp = 9
        track_w = rect.w - 16
        sun_x = rect.left + 8 + int(round(float(track_w) * float(tday)))
        h_sun = float(math.sin((float(tday) - 0.25) * math.tau))
        sun_y = int(round(horizon_y - h_sun * float(amp)))

        moon_t = (float(tday) + 0.5) % 1.0
        moon_x = rect.left + 8 + int(round(float(track_w) * float(moon_t)))
        moon_y = int(round(horizon_y + h_sun * float(amp)))

        if h_sun > 0.0:
            pygame.draw.circle(surface, (240, 220, 140), (sun_x, sun_y), 4)
            pygame.draw.circle(surface, (0, 0, 0), (sun_x, sun_y), 4, 1)
            for dx, dy in ((-6, 0), (6, 0), (0, -6), (0, 6), (-4, -4), (4, -4), (-4, 4), (4, 4)):
                surface.set_at((int(clamp(sun_x + dx, rect.left + 2, rect.right - 3)), int(clamp(sun_y + dy, rect.top + 2, rect.bottom - 3))), (240, 220, 140))
        else:
            pygame.draw.circle(surface, (220, 220, 230), (moon_x, moon_y), 4)
            pygame.draw.circle(surface, (0, 0, 0), (moon_x, moon_y), 4, 1)
            cut_x = moon_x + 2
            cut_y = moon_y - 1
            pygame.draw.circle(surface, (18, 18, 22), (cut_x, cut_y), 4)

    def _draw_minimap(self, surface: pygame.Surface) -> pygame.Rect | None:
        if bool(getattr(self, "rv_interior", False)) or bool(getattr(self, "hr_interior", False)) or bool(getattr(self, "sch_interior", False)):
            return None

        # Keep the minimap panel size similar (player-centric).
        radius = 15  # tiles
        scale = 3  # px per tile
        tiles = int(radius) * 2 + 1
        map_px = int(tiles) * int(scale)
        pad = 6
        label_h = 12

        panel = pygame.Rect(0, 0, map_px + pad * 2 + 2, map_px + pad * 2 + label_h + 2)
        panel.topright = (INTERNAL_W - 6, 60)
        pygame.draw.rect(surface, (18, 18, 22), panel, border_radius=12)
        pygame.draw.rect(surface, (70, 70, 86), panel, 2, border_radius=12)

        tx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
        ty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
        season = int(self._season_index())
        key = (int(tx), int(ty), int(radius), int(scale), int(season))

        cached_key = getattr(self, "_minimap_cache_key", None)
        cached_scaled: pygame.Surface | None = getattr(self, "_minimap_cache_scaled", None)
        size_ok = isinstance(cached_scaled, pygame.Surface) and cached_scaled.get_size() == (map_px, map_px)
        need_rebuild = (cached_key != key) or (not size_ok)
        if need_rebuild:
            moving = False
            if getattr(self, "mount", None) == "rv" and getattr(self, "rv", None) is not None:
                moving = self.rv.vel.length_squared() > 0.1
            elif getattr(self, "mount", None) == "bike" and getattr(self, "bike", None) is not None:
                moving = self.bike.vel.length_squared() > 0.1
            else:
                moving = self.player.vel.length_squared() > 0.1

            # Throttle minimap rebuilds while moving to avoid stutter.
            now = float(time.monotonic())
            last_t = float(getattr(self, "_minimap_cache_t", 0.0))
            min_interval = 0.12 if moving else 0.0
            if size_ok and min_interval > 0.0 and (now - last_t) < float(min_interval):
                pass
            else:
                base = pygame.Surface((tiles, tiles))
                base.lock()
                for oy in range(-int(radius), int(radius) + 1):
                    wy = int(ty) + int(oy)
                    py = int(oy) + int(radius)
                    for ox in range(-int(radius), int(radius) + 1):
                        wx = int(tx) + int(ox)
                        px = int(ox) + int(radius)
                        tid = self.world.peek_tile(wx, wy)
                        base.set_at((px, py), self._minimap_color(tid))
                base.unlock()
                cached_scaled = pygame.transform.scale(base, (map_px, map_px))
                self._minimap_cache_key = key
                self._minimap_cache_scaled = cached_scaled
                self._minimap_cache_t = float(now)

        draw_text(surface, self.app.font_s, "小地图", (panel.left + pad, panel.top + 3), pygame.Color(200, 200, 210), anchor="topleft")

        home = getattr(self, "home_highrise_door", None)
        if isinstance(home, tuple) and len(home) == 2:
            hx, hy = int(home[0]), int(home[1])
            dx = float(hx - int(tx))
            dy = float(hy - int(ty))
            dist = float(math.hypot(dx, dy))
            if dist < 1000.0:
                dist_text = f"家 {int(round(dist))}m"
            else:
                dist_text = f"家 {dist / 1000.0:.1f}km"
            draw_text(surface, self.app.font_s, dist_text, (panel.right - pad, panel.top + 3), pygame.Color(255, 220, 140), anchor="topright")

        map_x0 = panel.left + pad
        map_y0 = panel.top + pad + label_h
        if isinstance(cached_scaled, pygame.Surface):
            surface.blit(cached_scaled, (map_x0, map_y0))

        # Border for the map area.
        pygame.draw.rect(surface, (10, 10, 12), pygame.Rect(map_x0 - 1, map_y0 - 1, map_px + 2, map_px + 2), 1)

        # Player marker.
        cx = int(map_x0 + int(radius) * int(scale) + int(scale) // 2)
        cy = int(map_y0 + int(radius) * int(scale) + int(scale) // 2)
        pygame.draw.circle(surface, (240, 240, 244), (cx, cy), 3)
        pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 3, 1)

        # Home marker (inside map) or arrow (off-screen).
        if isinstance(home, tuple) and len(home) == 2:
            hx, hy = int(home[0]), int(home[1])
            dx_t = int(hx - int(tx))
            dy_t = int(hy - int(ty))
            if abs(dx_t) <= int(radius) and abs(dy_t) <= int(radius):
                mx = int(map_x0 + (dx_t + int(radius)) * int(scale) + int(scale) // 2)
                my = int(map_y0 + (dy_t + int(radius)) * int(scale) + int(scale) // 2)
                pygame.draw.circle(surface, (255, 220, 140), (mx, my), 4)
                pygame.draw.circle(surface, (0, 0, 0), (mx, my), 4, 1)
                surface.fill((255, 245, 190), pygame.Rect(int(mx - 1), int(my - 1), 2, 2))
            else:
                m = max(abs(dx_t), abs(dy_t))
                if m <= 0:
                    ux = 0.0
                    uy = 0.0
                else:
                    ux = float(dx_t) / float(m)
                    uy = float(dy_t) / float(m)
                    l = float(math.hypot(ux, uy))
                    if l > 1e-6:
                        ux /= l
                        uy /= l
                inset = float(radius) - 0.8
                tip_x = float(map_x0) + (float(radius) + ux * inset) * float(scale) + float(scale) * 0.5
                tip_y = float(map_y0) + (float(radius) + uy * inset) * float(scale) + float(scale) * 0.5
                base_x = tip_x - ux * 8.0
                base_y = tip_y - uy * 8.0
                px = -uy
                py = ux
                left = (base_x + px * 4.0, base_y + py * 4.0)
                right = (base_x - px * 4.0, base_y - py * 4.0)
                tip = (tip_x, tip_y)
                pygame.draw.polygon(surface, (255, 220, 140), [tip, left, right])
                pygame.draw.polygon(surface, (0, 0, 0), [tip, left, right], 1)
                lx = int(round(base_x + px * 5.0))
                ly = int(round(base_y + py * 5.0))
                draw_text(surface, self.app.font_s, "家", (lx, ly), pygame.Color(240, 240, 244), anchor="center")

        def draw_edge_arrow(
            *,
            dx_t: int,
            dy_t: int,
            color: tuple[int, int, int],
            label: str | None = None,
        ) -> None:
            m = max(abs(int(dx_t)), abs(int(dy_t)))
            if m <= 0:
                return
            ux = float(dx_t) / float(m)
            uy = float(dy_t) / float(m)
            l = float(math.hypot(ux, uy))
            if l > 1e-6:
                ux /= l
                uy /= l
            inset = float(radius) - 0.8
            tip_x = float(map_x0) + (float(radius) + ux * inset) * float(scale) + float(scale) * 0.5
            tip_y = float(map_y0) + (float(radius) + uy * inset) * float(scale) + float(scale) * 0.5
            base_x = tip_x - ux * 7.0
            base_y = tip_y - uy * 7.0
            px = -uy
            py = ux
            left = (base_x + px * 3.5, base_y + py * 3.5)
            right = (base_x - px * 3.5, base_y - py * 3.5)
            tip = (tip_x, tip_y)
            pygame.draw.polygon(surface, tuple(color), [tip, left, right])
            pygame.draw.polygon(surface, (0, 0, 0), [tip, left, right], 1)
            if label:
                lx = int(round(base_x + px * 5.0))
                ly = int(round(base_y + py * 5.0))
                draw_text(surface, self.app.font_s, label, (lx, ly), pygame.Color(240, 240, 240), anchor="center")

        # Player-made markers from the big map (dot inside radius, arrow when off-screen).
        markers = getattr(self, "world_map_markers", None)
        if isinstance(markers, list) and markers:
            inside: list[tuple[int, int, tuple[int, int, int], str]] = []
            outside: list[tuple[int, int, float, tuple[int, int, int], str]] = []
            for m in markers:
                mx_t = int(getattr(m, "tx", 0))
                my_t = int(getattr(m, "ty", 0))
                dx_t = int(mx_t - int(tx))
                dy_t = int(my_t - int(ty))
                col = tuple(getattr(m, "color", (255, 220, 140)))
                lab = str(getattr(m, "label", ""))
                if abs(int(dx_t)) <= int(radius) and abs(int(dy_t)) <= int(radius):
                    inside.append((dx_t, dy_t, col, lab))
                else:
                    d2 = float(dx_t * dx_t + dy_t * dy_t)
                    outside.append((dx_t, dy_t, d2, col, lab))

            for dx_t, dy_t, col, _lab in inside[:24]:
                mx = int(map_x0 + (int(dx_t) + int(radius)) * int(scale) + int(scale) // 2)
                my = int(map_y0 + (int(dy_t) + int(radius)) * int(scale) + int(scale) // 2)
                pygame.draw.circle(surface, tuple(col), (mx, my), 2)
                pygame.draw.circle(surface, (0, 0, 0), (mx, my), 2, 1)

            outside.sort(key=lambda t: t[2])
            for dx_t, dy_t, _d2, col, lab in outside[:3]:
                draw_edge_arrow(dx_t=int(dx_t), dy_t=int(dy_t), color=tuple(col), label=str(lab) if lab else None)

        return panel

    def _toggle_world_map(self, *, open: bool | None = None) -> None:
        if open is None:
            open = not bool(getattr(self, "world_map_open", False))
        self.world_map_open = bool(open)
        if not self.world_map_open:
            return

        # Close other overlays when opening the big map.
        self.inv_open = False
        self.rv_ui_open = False
        self.hr_elevator_ui_open = False
        self._gallery_open = False
        self.world_map_center = self._player_tile()
        self._world_map_cache_key = None
        self.world_map_dragging = False

    def _world_map_tile_from_point(self, x: int, y: int) -> tuple[int, int] | None:
        rect = getattr(self, "_world_map_draw_rect", None)
        if not isinstance(rect, pygame.Rect):
            return None
        if not rect.collidepoint(int(x), int(y)):
            return None
        scale = max(1, int(getattr(self, "world_map_scale", 1)))
        start_tx, start_ty = getattr(self, "_world_map_start_tile", (0, 0))
        tx = int(start_tx) + int((int(x) - int(rect.x)) // scale)
        ty = int(start_ty) + int((int(y) - int(rect.y)) // scale)
        return int(tx), int(ty)

    def _add_world_map_marker(self, tx: int, ty: int) -> None:
        tx = int(tx)
        ty = int(ty)
        markers = getattr(self, "world_map_markers", None)
        if not isinstance(markers, list):
            self.world_map_markers = []
            markers = self.world_map_markers
        # Replace any marker on the same tile.
        self.world_map_markers = [m for m in markers if not (int(m.tx) == tx and int(m.ty) == ty)]

        idx = int(getattr(self, "world_map_marker_next", 0))
        self.world_map_marker_next = idx + 1
        label = chr(ord("A") + (idx % 26)) if idx < 26 else str(idx + 1)
        palette = [(255, 220, 140), (140, 220, 255), (220, 140, 255), (220, 255, 140)]
        color = palette[idx % len(palette)]
        self.world_map_markers.append(HardcoreSurvivalState._MapMarker(tx=tx, ty=ty, label=label, color=color))
        self._set_hint(f"已标记 {label}", seconds=1.2)

    def _remove_world_map_marker_near(self, tx: int, ty: int) -> None:
        tx = int(tx)
        ty = int(ty)
        markers = getattr(self, "world_map_markers", None)
        if not isinstance(markers, list) or not markers:
            return
        best_i = -1
        best_d2 = 999999
        for i, m in enumerate(markers):
            d2 = (int(m.tx) - tx) ** 2 + (int(m.ty) - ty) ** 2
            if d2 < best_d2:
                best_d2 = int(d2)
                best_i = int(i)
        if best_i < 0 or best_d2 > 36:
            return
        label = str(getattr(markers[best_i], "label", "?"))
        try:
            del markers[best_i]
        except Exception:
            self.world_map_markers = [m for i, m in enumerate(markers) if i != best_i]
        self._set_hint(f"删除标记 {label}", seconds=1.0)

    def _draw_world_map_icon(self, surface: pygame.Surface, cx: int, cy: int, kind: str, *, scale: int) -> None:
        scale = max(1, int(scale))
        s = int(max(5, min(15, 3 + scale * 3)))
        half = s // 2
        x = int(cx)
        y = int(cy)

        if kind == "hospital":
            col = (220, 70, 70)
            w = max(1, int(round(s / 5)))
            pygame.draw.line(surface, (0, 0, 0), (x, y - half), (x, y + half), w + 2)
            pygame.draw.line(surface, (0, 0, 0), (x - half, y), (x + half, y), w + 2)
            pygame.draw.line(surface, col, (x, y - half), (x, y + half), w)
            pygame.draw.line(surface, col, (x - half, y), (x + half, y), w)
            return

        if kind == "market":
            col = (220, 200, 120)
            r = pygame.Rect(0, 0, s, s)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, col, r.inflate(-2, -2))
            # Handle
            pygame.draw.line(surface, (0, 0, 0), (r.left + 2, r.top + 3), (r.right - 3, r.top + 3), 2)
            pygame.draw.line(surface, col, (r.left + 2, r.top + 3), (r.right - 3, r.top + 3), 1)
            return

        if kind == "bookstore":
            col = (232, 200, 120)
            r = pygame.Rect(0, 0, s, s)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, (80, 100, 150), r.inflate(-2, -2))
            # Book
            bx = int(r.centerx)
            by = int(r.centery)
            pygame.draw.rect(surface, col, pygame.Rect(bx - 5, by - 4, 4, 8), 1, border_radius=1)
            pygame.draw.rect(surface, col, pygame.Rect(bx + 1, by - 4, 4, 8), 1, border_radius=1)
            surface.fill((0, 0, 0), pygame.Rect(bx, by - 4, 1, 8))
            return

        if kind == "chinese":
            col = (210, 180, 110)
            r = pygame.Rect(0, 0, s, s)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, (150, 60, 50), r.inflate(-2, -2))
            # Roof + pillars
            pygame.draw.line(surface, col, (r.left + 2, r.top + 3), (r.right - 3, r.top + 3), 1)
            pygame.draw.line(surface, col, (r.left + 3, r.top + 2), (r.right - 4, r.top + 2), 1)
            surface.fill(col, pygame.Rect(r.left + 3, r.top + 4, 1, max(1, s - 7)))
            surface.fill(col, pygame.Rect(r.right - 4, r.top + 4, 1, max(1, s - 7)))
            return

        if kind == "school":
            col = (110, 170, 240)
            r = pygame.Rect(0, 0, s, s)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, col, r.inflate(-2, -2))
            # Flag
            pygame.draw.line(surface, (0, 0, 0), (r.centerx - 2, r.top + 1), (r.centerx - 2, r.top + 5), 1)
            pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(r.centerx - 1, r.top + 1, 3, 2))
            pygame.draw.rect(surface, (250, 240, 200), pygame.Rect(r.centerx, r.top + 2, 1, 1))
            return

        if kind == "prison":
            col = (150, 150, 160)
            r = pygame.Rect(0, 0, s, s)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, col, r.inflate(-2, -2))
            for i in range(r.left + 2, r.right - 2, 3):
                pygame.draw.line(surface, (0, 0, 0), (i, r.top + 2), (i, r.bottom - 3), 1)
            return

        if kind == "highrise":
            col = (210, 210, 220)
            w = max(3, s // 2)
            h = s + max(2, s // 3)
            r = pygame.Rect(0, 0, w, h)
            r.center = (x, y)
            pygame.draw.rect(surface, (0, 0, 0), r)
            pygame.draw.rect(surface, col, r.inflate(-2, -2))
            # Windows
            for wy in range(r.top + 3, r.bottom - 2, 3):
                surface.fill((0, 0, 0), pygame.Rect(r.centerx, wy, 1, 1))
            return

    def _draw_world_map_pois(
        self,
        surface: pygame.Surface,
        *,
        draw_x: int,
        draw_y: int,
        start_tx: int,
        start_ty: int,
        view_w: int,
        view_h: int,
        scale: int,
    ) -> None:
        scale = max(1, int(scale))
        start_tx = int(start_tx)
        start_ty = int(start_ty)
        view_w = int(max(1, view_w))
        view_h = int(max(1, view_h))

        end_tx = start_tx + view_w - 1
        end_ty = start_ty + view_h - 1
        start_cx = start_tx // self.CHUNK_SIZE
        end_cx = end_tx // self.CHUNK_SIZE
        start_cy = start_ty // self.CHUNK_SIZE
        end_cy = end_ty // self.CHUNK_SIZE

        style_to_kind = {
            2: "market",
            3: "hospital",
            4: "prison",
            5: "school",
            6: "highrise",
            7: "bookstore",
            8: "chinese",
        }

        for cy in range(int(start_cy), int(end_cy) + 1):
            for cx in range(int(start_cx), int(end_cx) + 1):
                chunk = self.world.peek_chunk(cx, cy)
                if chunk is None:
                    continue
                for b in getattr(chunk, "buildings", []):
                    if len(b) < 5:
                        continue
                    tx0, ty0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    roof_kind = int(b[4])
                    style = int((roof_kind >> 8) & 0xFF)
                    kind = style_to_kind.get(int(style))
                    if kind is None:
                        continue

                    c_tx = int(tx0 + max(1, w) // 2)
                    c_ty = int(ty0 + max(1, h) // 2)
                    if not (start_tx <= c_tx <= end_tx and start_ty <= c_ty <= end_ty):
                        continue
                    mx = int(draw_x + (c_tx - start_tx) * scale + scale // 2)
                    my = int(draw_y + (c_ty - start_ty) * scale + scale // 2)
                    self._draw_world_map_icon(surface, mx, my, str(kind), scale=scale)

    def _draw_world_map_legend(self, surface: pygame.Surface, *, map_area: pygame.Rect) -> None:
        legend = pygame.Rect(0, 0, 162, 132)
        legend.bottomright = (int(map_area.right) - 8, int(map_area.bottom) - 8)

        ui = pygame.Surface((legend.w, legend.h), pygame.SRCALPHA)
        ui.fill((12, 12, 16, 210))
        pygame.draw.rect(ui, (90, 90, 110, 220), ui.get_rect(), 1, border_radius=10)
        surface.blit(ui, legend.topleft)

        draw_text(surface, self.app.font_s, "图例", (legend.left + 10, legend.top + 6), pygame.Color(240, 240, 240), anchor="topleft")

        # Clickable toggle button (hide legend).
        btn = pygame.Rect(0, 0, 14, 12)
        btn.topright = (legend.right - 8, legend.top + 6)
        pygame.draw.rect(surface, (24, 24, 30), btn, border_radius=4)
        pygame.draw.rect(surface, (90, 90, 110), btn, 1, border_radius=4)
        draw_text(surface, self.app.font_s, "x", (btn.centerx, btn.centery - 1), pygame.Color(210, 210, 220), anchor="center")
        self._world_map_legend_toggle_rect = btn.copy()

        # Icons (fixed size independent of zoom).
        icon_scale = 2
        x0 = legend.left + 12
        y = legend.top + 22
        step = 14

        entries: list[tuple[str, str]] = [
            ("hospital", "医院"),
            ("market", "超市"),
            ("bookstore", "书店"),
            ("school", "学校"),
            ("prison", "监狱"),
            ("highrise", "高层"),
            ("chinese", "中式"),
        ]
        for kind, label in entries:
            self._draw_world_map_icon(surface, x0 + 6, y + 6, kind, scale=icon_scale)
            draw_text(surface, self.app.font_s, label, (x0 + 18, y + 1), pygame.Color(210, 210, 220), anchor="topleft")
            y += step

        # Player / home / RV / markers.
        y2 = legend.top + 22
        x1 = legend.left + 92
        pygame.draw.circle(surface, (240, 240, 244), (x1 + 6, y2 + 6), 3)
        pygame.draw.circle(surface, (0, 0, 0), (x1 + 6, y2 + 6), 3, 1)
        draw_text(surface, self.app.font_s, "你", (x1 + 16, y2 + 1), pygame.Color(210, 210, 220), anchor="topleft")

        y2 += step
        pygame.draw.circle(surface, (255, 220, 140), (x1 + 6, y2 + 6), 3)
        pygame.draw.circle(surface, (0, 0, 0), (x1 + 6, y2 + 6), 3, 1)
        draw_text(surface, self.app.font_s, "家", (x1 + 16, y2 + 1), pygame.Color(210, 210, 220), anchor="topleft")

        y2 += step
        pygame.draw.circle(surface, (140, 220, 255), (x1 + 6, y2 + 6), 3)
        pygame.draw.circle(surface, (0, 0, 0), (x1 + 6, y2 + 6), 3, 1)
        draw_text(surface, self.app.font_s, "房车", (x1 + 16, y2 + 1), pygame.Color(210, 210, 220), anchor="topleft")

        y2 += step
        pygame.draw.circle(surface, (220, 140, 255), (x1 + 6, y2 + 6), 3)
        pygame.draw.circle(surface, (0, 0, 0), (x1 + 6, y2 + 6), 3, 1)
        draw_text(surface, self.app.font_s, "标记", (x1 + 16, y2 + 1), pygame.Color(210, 210, 220), anchor="topleft")

    def _handle_world_map_key(self, key: int) -> None:
        key = int(key)
        if key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_TAB):
            self.world_map_open = False
            return
        if key in (pygame.K_l,):
            self.world_map_legend_open = not bool(getattr(self, "world_map_legend_open", True))
            return

        mods = int(pygame.key.get_mods())
        fast = bool(mods & pygame.KMOD_SHIFT)
        step = 40 if fast else 14
        cx, cy = getattr(self, "world_map_center", (0, 0))
        cx = int(cx)
        cy = int(cy)

        if key in (pygame.K_LEFT, pygame.K_a):
            cx -= step
        elif key in (pygame.K_RIGHT, pygame.K_d):
            cx += step
        elif key in (pygame.K_UP, pygame.K_w):
            cy -= step
        elif key in (pygame.K_DOWN, pygame.K_s):
            cy += step
        elif key in (pygame.K_HOME, pygame.K_0):
            cx, cy = self._player_tile()
        elif key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.world_map_scale = int(clamp(int(getattr(self, "world_map_scale", 1))) + 1, 1, 4)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.world_map_scale = int(clamp(int(getattr(self, "world_map_scale", 1))) - 1, 1, 4)
        else:
            return

        self.world_map_center = (int(cx), int(cy))
        self._world_map_cache_key = None

    def _handle_world_map_mouse(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            dy = int(getattr(event, "y", 0))
            if dy != 0:
                self.world_map_scale = int(clamp(int(getattr(self, "world_map_scale", 1)) + (1 if dy > 0 else -1), 1, 4))
                self._world_map_cache_key = None
            return

        if event.type == pygame.MOUSEBUTTONUP:
            btn = int(getattr(event, "button", 0))
            if btn == 3:
                self.world_map_dragging = False
            return

        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            return
        internal = self.app.screen_to_internal(getattr(event, "pos", (0, 0)))
        if internal is None:
            return
        ix = int(internal[0])
        iy = int(internal[1])

        if event.type == pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) == 1:
            toggle = getattr(self, "_world_map_legend_toggle_rect", None)
            if isinstance(toggle, pygame.Rect) and toggle.collidepoint(ix, iy):
                self.world_map_legend_open = not bool(getattr(self, "world_map_legend_open", True))
                return

        if event.type == pygame.MOUSEMOTION:
            if not bool(getattr(self, "world_map_dragging", False)):
                return
            scale = max(1, int(getattr(self, "world_map_scale", 1)))
            sx, sy = getattr(self, "world_map_drag_start", (ix, iy))
            cx0, cy0 = getattr(self, "world_map_drag_center", getattr(self, "world_map_center", (0, 0)))
            dx = int(ix) - int(sx)
            dy = int(iy) - int(sy)
            tcx = int(cx0) - int(round(float(dx) / float(scale)))
            tcy = int(cy0) - int(round(float(dy) / float(scale)))
            self.world_map_center = (int(tcx), int(tcy))
            self._world_map_cache_key = None
            return

        tx_ty = self._world_map_tile_from_point(ix, iy)
        if tx_ty is None:
            return
        tx, ty = tx_ty

        btn = int(getattr(event, "button", 0))
        if btn == 1:
            self._add_world_map_marker(tx, ty)
        elif btn == 3:
            mods = int(pygame.key.get_mods())
            if mods & pygame.KMOD_CTRL:
                self._remove_world_map_marker_near(tx, ty)
                return
            self.world_map_dragging = True
            self.world_map_drag_start = (int(ix), int(iy))
            self.world_map_drag_center = tuple(getattr(self, "world_map_center", (0, 0)))
        elif btn in (4, 5):
            self.world_map_scale = int(clamp(int(getattr(self, "world_map_scale", 1)) + (1 if btn == 4 else -1), 1, 4))
            self._world_map_cache_key = None

    def _draw_world_map_ui(self, surface: pygame.Surface) -> None:
        panel = pygame.Rect(10, 10, INTERNAL_W - 20, INTERNAL_H - 20)
        pygame.draw.rect(surface, (12, 12, 16), panel, border_radius=12)
        pygame.draw.rect(surface, (90, 90, 110), panel, 2, border_radius=12)

        title = "大地图"
        draw_text(surface, self.app.font_m, title, (panel.centerx, panel.top + 12), pygame.Color(240, 240, 240), anchor="center")
        hint = "左键标记  右键拖动  Ctrl+右键删除  滚轮缩放  WASD/方向键移动  L 图例  M/Esc关闭"
        draw_text(surface, self.app.font_s, hint, (panel.centerx, panel.bottom - 10), pygame.Color(170, 170, 180), anchor="center")

        map_area = pygame.Rect(panel.left + 10, panel.top + 26, panel.w - 20, panel.h - 44)
        pygame.draw.rect(surface, (18, 18, 22), map_area, border_radius=10)
        pygame.draw.rect(surface, (40, 40, 50), map_area, 1, border_radius=10)

        scale = max(1, int(getattr(self, "world_map_scale", 1)))
        view_w = max(1, map_area.w // scale)
        view_h = max(1, map_area.h // scale)
        cx, cy = getattr(self, "world_map_center", (0, 0))
        cx = int(cx)
        cy = int(cy)
        start_tx = int(cx - view_w // 2)
        start_ty = int(cy - view_h // 2)

        season = int(self._season_index())
        key = (start_tx, start_ty, view_w, view_h, int(scale), int(season))
        cached_scaled: pygame.Surface | None = getattr(self, "_world_map_cache_scaled", None)
        if getattr(self, "_world_map_cache_key", None) != key or cached_scaled is None:
            base = pygame.Surface((view_w, view_h))
            base.lock()
            for py in range(view_h):
                wy = start_ty + py
                for px in range(view_w):
                    wx = start_tx + px
                    tid = self.world.peek_tile(wx, wy)
                    base.set_at((px, py), self._minimap_color(tid))
            base.unlock()
            cached_scaled = pygame.transform.scale(base, (view_w * scale, view_h * scale))
            self._world_map_cache_key = key
            self._world_map_cache_scaled = cached_scaled

        draw_x = int(map_area.x + (map_area.w - cached_scaled.get_width()) // 2)
        draw_y = int(map_area.y + (map_area.h - cached_scaled.get_height()) // 2)
        self._world_map_draw_rect = pygame.Rect(draw_x, draw_y, cached_scaled.get_width(), cached_scaled.get_height())
        self._world_map_start_tile = (start_tx, start_ty)
        surface.blit(cached_scaled, (draw_x, draw_y))

        # POI icons (public buildings): hospital/market/school/etc.
        self._draw_world_map_pois(
            surface,
            draw_x=int(draw_x),
            draw_y=int(draw_y),
            start_tx=int(start_tx),
            start_ty=int(start_ty),
            view_w=int(view_w),
            view_h=int(view_h),
            scale=int(scale),
        )

        def draw_marker(tx: int, ty: int, col: tuple[int, int, int], label: str | None = None, *, r: int = 3) -> None:
            if not (start_tx <= int(tx) < start_tx + view_w and start_ty <= int(ty) < start_ty + view_h):
                return
            mx = int(draw_x + (int(tx) - start_tx) * scale + scale // 2)
            my = int(draw_y + (int(ty) - start_ty) * scale + scale // 2)
            rr = int(max(2, int(r)))
            pygame.draw.circle(surface, col, (mx, my), rr)
            pygame.draw.circle(surface, (0, 0, 0), (mx, my), rr, 1)
            if label:
                draw_text(surface, self.app.font_s, label, (mx + 6, my - 6), pygame.Color(240, 240, 240), anchor="topleft")

        # Player + key landmarks.
        ptx, pty = self._player_tile()
        draw_marker(ptx, pty, (240, 240, 244), None)

        rtx = int(math.floor(self.rv.pos.x / self.TILE_SIZE))
        rty = int(math.floor(self.rv.pos.y / self.TILE_SIZE))
        draw_marker(rtx, rty, (140, 220, 255), "房车")

        home = getattr(self, "home_highrise_door", None)
        if isinstance(home, tuple) and len(home) == 2:
            draw_marker(int(home[0]), int(home[1]), (255, 220, 140), "家", r=4)

        for m in getattr(self, "world_map_markers", []):
            draw_marker(int(getattr(m, "tx", 0)), int(getattr(m, "ty", 0)), tuple(getattr(m, "color", (255, 220, 140))), str(getattr(m, "label", "")))

        # Legend (icon key): bottom-right, toggleable.
        self._world_map_legend_toggle_rect = None
        if bool(getattr(self, "world_map_legend_open", True)):
            self._draw_world_map_legend(surface, map_area=map_area)
        else:
            btn = pygame.Rect(0, 0, 58, 14)
            btn.bottomright = (int(map_area.right) - 8, int(map_area.bottom) - 8)
            ui = pygame.Surface((btn.w, btn.h), pygame.SRCALPHA)
            ui.fill((12, 12, 16, 210))
            pygame.draw.rect(ui, (90, 90, 110, 220), ui.get_rect(), 1, border_radius=8)
            surface.blit(ui, btn.topleft)
            draw_text(surface, self.app.font_s, "图例(L)", (btn.centerx, btn.centery - 1), pygame.Color(210, 210, 220), anchor="center")
            self._world_map_legend_toggle_rect = btn.copy()

    def _draw_survival_ui(self, surface: pygame.Surface) -> None:
        in_rv = bool(getattr(self, "rv_interior", False))
        day = int(self.world_time_s / max(1e-6, self.DAY_LENGTH_S)) + 1
        season = self.SEASONS[self._season_index()]
        _daylight, tday = self._daylight_amount()
        hh = int(tday * 24) % 24
        mm = int(tday * 24 * 60) % 60
        w = self.WEATHER_NAMES.get(str(getattr(self, "weather_kind", "clear")), "?")
        info = f"DAY {day}  {season}  {hh:02d}:{mm:02d}  {w}"
        draw_text(surface, self.app.font_s, info, (6, 6), pygame.Color(230, 230, 240), anchor="topleft")

        def stat_line(y: int, label: str, value: float, *, fg: tuple[int, int, int]) -> None:
            value = float(clamp(value, 0.0, 100.0))
            draw_text(surface, self.app.font_s, f"{label} {int(round(value)):3d}", (6, y), pygame.Color(220, 220, 230), anchor="topleft")
            bar = pygame.Rect(56, y + 4, 74, 6)
            pygame.draw.rect(surface, (10, 10, 14), bar)
            pygame.draw.rect(surface, (70, 70, 86), bar, 1)
            fill_w = int((bar.w - 2) * (value / 100.0))
            if fill_w > 0:
                pygame.draw.rect(surface, fg, pygame.Rect(bar.x + 1, bar.y + 1, fill_w, bar.h - 2))

        stat_line(20, "HP", float(self.player.hp), fg=(220, 90, 90))
        stat_line(32, "饿", self.player.hunger, fg=(210, 170, 90))
        stat_line(44, "渴", self.player.thirst, fg=(120, 170, 230))
        stat_line(56, "健", self.player.condition, fg=(120, 210, 140))
        stat_line(68, "心", self.player.morale, fg=(200, 160, 240))
        stat_line(80, "体", self.player.stamina, fg=(240, 240, 240))

        if self.gun is None:
            draw_text(surface, self.app.font_s, "枪：无", (INTERNAL_W - 6, 6), pygame.Color(200, 200, 210), anchor="topright")
        else:
            gdef = self._GUNS.get(self.gun.gun_id)
            if gdef is None:
                draw_text(surface, self.app.font_s, "枪：未知", (INTERNAL_W - 6, 6), pygame.Color(200, 200, 210), anchor="topright")
            else:
                reserve = self.inventory.count(gdef.ammo_item)
                extra = " 换弹" if self.gun.reload_left > 0.0 else ""
                draw_text(
                    surface,
                    self.app.font_s,
                    f"{gdef.name} {int(self.gun.mag)}/{reserve}{extra}",
                    (INTERNAL_W - 6, 6),
                    pygame.Color(230, 230, 240),
                    anchor="topright",
                )
        draw_text(surface, self.app.font_s, f"房车油 {int(self.rv.fuel)}", (INTERNAL_W - 6, 20), pygame.Color(180, 180, 190), anchor="topright")
        if self.mount == "bike" and str(getattr(self.bike, "model_id", "bike")).startswith("moto"):
            draw_text(
                surface,
                self.app.font_s,
                f"摩托油 {int(getattr(self.bike, 'fuel', 0.0))}",
                (INTERNAL_W - 6, 32),
                pygame.Color(180, 180, 190),
                anchor="topright",
            )
        self._draw_sun_moon_widget(surface, tday=float(tday))

        if in_rv:
            tx, ty = self._rv_int_player_tile()
            prompt = ""
            for cx, cy in ((tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)):
                ch = self._rv_int_char_at(cx, cy)
                if ch == "D":
                    prompt = "E 出门"
                    break
                if ch == "B":
                    prompt = "E 睡觉/休息"
                    break
                if ch == "K":
                    prompt = "E 厨房"
                    break
                if ch == "H":
                    prompt = "E 工作台"
                    break
                if ch == "S":
                    prompt = "E 打开储物柜"
                    break
            if prompt:
                draw_text(surface, self.app.font_s, prompt, (6, 96), pygame.Color(220, 220, 230), anchor="topleft")

            if self.hint_text:
                draw_text(surface, self.app.font_s, self.hint_text, (INTERNAL_W // 2, INTERNAL_H - 30), pygame.Color(255, 220, 140), anchor="center")
            if getattr(self, "rv_edit_mode", False):
                help_text = "房车内部(摆放模式)  Tab选中  方向键移动  R退出  H/ESC退出"
            else:
                help_text = "房车内部  WASD移动  E互动  R摆放  H/ESC退出  Tab背包"
            draw_text(surface, self.app.font_s, help_text, (6, INTERNAL_H - 14), pygame.Color(170, 170, 180), anchor="topleft")
            return

        minimap_rect = self._draw_minimap(surface)
        self.minimap_rect = minimap_rect if isinstance(minimap_rect, pygame.Rect) else None

        _chunk, it = self._find_nearest_item(radius_px=20.0)
        if it is not None:
            idef = self._ITEMS.get(it.item_id)
            name = idef.name if idef is not None else it.item_id
            draw_text(surface, self.app.font_s, f"E 拾取：{name} x{int(it.qty)}", (6, 96), pygame.Color(220, 220, 230), anchor="topleft")

        if self.hint_text:
            draw_text(surface, self.app.font_s, self.hint_text, (INTERNAL_W // 2, INTERNAL_H - 30), pygame.Color(255, 220, 140), anchor="center")

        y0 = 34
        if isinstance(minimap_rect, pygame.Rect):
            y0 = max(int(y0), int(minimap_rect.bottom + 6))

        if self.mount is not None:
            kind = "房车" if self.mount == "rv" else self._two_wheel_name(getattr(self.bike, "model_id", "bike"))
            draw_text(surface, self.app.font_s, f"F 下车({kind})", (INTERNAL_W - 6, int(y0)), pygame.Color(220, 220, 230), anchor="topright")
            if self.mount == "rv":
                draw_text(surface, self.app.font_s, "H 房车内部", (INTERNAL_W - 6, int(y0 + 12)), pygame.Color(220, 220, 230), anchor="topright")       
                draw_text(surface, self.app.font_s, "V 换车型", (INTERNAL_W - 6, int(y0 + 24)), pygame.Color(180, 180, 190), anchor="topright")
        else:
            y = int(y0)
            near_rv = (self.player.pos - self.rv.pos).length_squared() <= (22.0 * 22.0)
            if near_rv:
                draw_text(surface, self.app.font_s, "F 上房车", (INTERNAL_W - 6, y), pygame.Color(220, 220, 230), anchor="topright")
                y += 12
                draw_text(surface, self.app.font_s, "H 房车内部", (INTERNAL_W - 6, y), pygame.Color(220, 220, 230), anchor="topright")
                y += 12
                draw_text(surface, self.app.font_s, "V 换车型", (INTERNAL_W - 6, y), pygame.Color(180, 180, 190), anchor="topright")
                y += 12
            if (self.player.pos - self.bike.pos).length_squared() <= (18.0 * 18.0):
                draw_text(surface, self.app.font_s, f"F 骑车({self._two_wheel_name(getattr(self.bike, 'model_id', 'bike'))})", (INTERNAL_W - 6, y), pygame.Color(220, 220, 230), anchor="topright")
        draw_text(surface, self.app.font_s, "WASD移动  Shift冲刺  E拾取  F载具  H房车  F3骨骼  Tab背包  Esc菜单", (6, INTERNAL_H - 14), pygame.Color(170, 170, 180), anchor="topleft")

    def _draw_debug(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        tx = int(math.floor(self.player.pos.x / self.TILE_SIZE))
        ty = int(math.floor(self.player.pos.y / self.TILE_SIZE))
        cx = tx // self.CHUNK_SIZE
        cy = ty // self.CHUNK_SIZE
        chunk = self.world.get_chunk(cx, cy)
        info = f"seed {self.seed}  tile {tx},{ty}  chunk {cx},{cy}  town {chunk.town_kind or '-'}"
        draw_text(surface, self.app.font_s, info, (6, INTERNAL_H - 6), pygame.Color(180, 180, 190), anchor="topleft")

        # Player skeleton overlay (helps tuning side-walk + gun anchors).
        p = self.player.pos - pygame.Vector2(cam_x, cam_y)
        px = int(round(p.x))
        py = int(round(p.y))

        face = pygame.Vector2(self.player.facing)
        if self.gun is not None:
            face = pygame.Vector2(self.aim_dir)
        if face.length_squared() <= 0.001:
            face = pygame.Vector2(0, 1)

        if abs(face.y) >= abs(face.x):
            d = "down" if face.y >= 0 else "up"
        else:
            d = "right" if face.x >= 0 else "left"

        pf_walk = getattr(self, "player_frames", self._PLAYER_FRAMES)
        pf_run = getattr(self, "player_frames_run", None)
        is_run = bool(getattr(self, "player_sprinting", False))
        pf = pf_run if (is_run and isinstance(pf_run, dict)) else pf_walk
        frames = pf.get(d, pf["down"])
        moving = self.player.vel.length_squared() > 1.0
        bob = 0
        walk_idx = 0
        idle_anim = True
        if moving and len(frames) > 1:
            idle_anim = False
            walk = frames[1:]
            phase = (float(self.player.walk_phase) % math.tau) / math.tau
            walk_idx = int(phase * len(walk)) % len(walk)
            spr = walk[walk_idx]
            if is_run and walk_idx in (1, 4):
                bob = 1
        else:
            spr = frames[0]

        rect = spr.get_rect()
        rect.midbottom = (px, iround(float(p.y) + float(self.player.h) / 2.0))
        rect.y += bob

        height_delta = 0
        av = getattr(self, "avatar", None)
        if av is not None:
            hidx = int(getattr(av, "height", 1))
            height_delta = 1 if hidx == 0 else -1 if hidx == 2 else 0

        sk = self._survivor_skeleton_nodes(
            d,
            int(walk_idx),
            idle=bool(idle_anim),
            height_delta=int(height_delta),
            run=bool(is_run),
        )
        line_col = (255, 0, 200)
        node_col = (255, 240, 120)
        for a, b in self._SURVIVOR_SKELETON_BONES:
            pa = sk.get(a)
            pb = sk.get(b)
            if pa is None or pb is None:
                continue
            ax, ay = pa
            bx, by = pb
            pygame.draw.line(surface, line_col, (rect.left + ax, rect.top + ay), (rect.left + bx, rect.top + by), 1)

        for key in ("head", "chest", "hip", "l_knee", "r_knee", "l_foot", "r_foot", "r_hand"):
            pt = sk.get(key)
            if pt is None:
                continue
            x, y = pt
            pygame.draw.rect(surface, node_col, pygame.Rect(rect.left + x - 1, rect.top + y - 1, 3, 3), 1)


class SettingsState(State):
    def on_enter(self) -> None:
        self.items = ["网格", "返回"]
        self.index = 0
        self.item_rects: list[pygame.Rect] = []

    def _activate(self) -> None:
        if self.index == 0:
            self.app.set_config(show_grid=not self.app.config.show_grid)
        else:
            self.app.set_state(MainMenuState(self.app))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE,):
                self.app.set_state(MainMenuState(self.app))
                return

            if event.key in (pygame.K_UP, pygame.K_w):
                self.index = (self.index - 1) % len(self.items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.index = (self.index + 1) % len(self.items)
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                if self.index == 0:
                    self.app.set_config(show_grid=not self.app.config.show_grid)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                if self.index == 0:
                    self.app.set_config(show_grid=not self.app.config.show_grid)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate()
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        for i, rect in enumerate(self.item_rects):
            if rect.collidepoint(internal):
                self.index = i
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    self._activate()
                break

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((16, 16, 20))
        draw_text(surface, self.app.font_l, "设置", (INTERNAL_W // 2, 30), pygame.Color(240, 240, 240), anchor="center")

        self.item_rects = []
        y0 = 80
        for i, label in enumerate(self.items):
            selected = i == self.index
            value = ""
            if label == "网格":
                value = "开" if self.app.config.show_grid else "关"
            text = f"{label}  {value}".rstrip()
            rect = pygame.Rect(0, 0, 220, 22)
            rect.center = (INTERNAL_W // 2, y0 + i * 28)
            self.item_rects.append(draw_button(surface, self.app.font_m, text, rect, selected=selected))

        draw_text(
            surface,
            self.app.font_s,
            "左右或点击切换；Esc返回",
            (INTERNAL_W // 2, INTERNAL_H - 22),
            pygame.Color(170, 170, 180),
            anchor="center",
        )


class CharacterSelectState(State):
    def on_enter(self) -> None:
        self.index = 0
        self.dot_rects: list[pygame.Rect] = []
        self.btn_back_rect = pygame.Rect(0, 0, 0, 0)
        self.btn_ok_rect = pygame.Rect(0, 0, 0, 0)
        self.panel_rect = pygame.Rect(0, 0, 0, 0)

    def _confirm(self) -> None:
        self.app.set_state(MapSelectState(self.app, character=CHARS[self.index]))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE,):
                self.app.set_state(MainMenuState(self.app))
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.index = (self.index - 1) % len(CHARS)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.index = (self.index + 1) % len(CHARS)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._confirm()
            elif pygame.K_1 <= event.key <= pygame.K_9:
                n = event.key - pygame.K_1
                if 0 <= n < len(CHARS):
                    self.index = n
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        for i, r in enumerate(self.dot_rects):
            if r.collidepoint(internal):
                self.index = i
                break

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            if self.btn_back_rect.collidepoint(internal):
                self.app.set_state(MainMenuState(self.app))
                return
            if self.btn_ok_rect.collidepoint(internal) or self.panel_rect.collidepoint(internal):
                self._confirm()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((14, 14, 18))
        draw_text(surface, self.app.font_l, "选择角色", (INTERNAL_W // 2, 18), pygame.Color(240, 240, 240), anchor="center")
        draw_text(
            surface,
            self.app.font_s,
            "左右/鼠标选择 | Enter/左键确认 | Esc返回",
            (INTERNAL_W // 2, 34),
            pygame.Color(180, 180, 190),
            anchor="center",
        )

        char_def = CHARS[self.index]
        panel = pygame.Rect(10, 44, INTERNAL_W - 20, 88)
        self.panel_rect = panel.copy()
        pygame.draw.rect(surface, (24, 24, 30), panel, border_radius=8)
        pygame.draw.rect(surface, (50, 50, 60), panel, 2, border_radius=8)

        spr_scale = 4
        spr = pygame.transform.scale(
            char_def.sprite,
            (char_def.sprite.get_width() * spr_scale, char_def.sprite.get_height() * spr_scale),
        )
        spr_pos = (panel.left + 10, panel.top + 10)
        surface.blit(spr, spr_pos)

        tx = spr_pos[0] + spr.get_width() + 10
        ty = panel.top + 8
        draw_text(surface, self.app.font_m, f"{char_def.name}", (tx, ty), pygame.Color(240, 240, 240))
        draw_text(surface, self.app.font_s, char_def.desc, (tx, ty + 18), pygame.Color(180, 180, 190))

        stats = [
            f"生命: {char_def.base_hp}",
            f"移速: {int(char_def.move_speed)}",
            f"起手武器: {WEAPON_DEFS[char_def.start_weapon].name}",
        ]
        for i, line in enumerate(stats):
            draw_text(surface, self.app.font_s, line, (tx, ty + 34 + i * 14), pygame.Color(200, 200, 210))

        self.dot_rects = []
        dot_w, dot_h, dot_gap = 14, 8, 10
        total_w = len(CHARS) * dot_w + max(0, (len(CHARS) - 1) * dot_gap)
        start_x = (INTERNAL_W - total_w) // 2
        dots_y = panel.bottom + 6
        for i in range(len(CHARS)):
            c = (240, 240, 240) if i == self.index else (90, 90, 100)
            r = pygame.Rect(start_x + i * (dot_w + dot_gap), dots_y, dot_w, dot_h)
            self.dot_rects.append(r)
            pygame.draw.rect(surface, c, r)

        self.btn_back_rect = pygame.Rect(0, 0, 120, 22)
        self.btn_back_rect.center = (INTERNAL_W // 2 - 70, 160)
        self.btn_ok_rect = pygame.Rect(0, 0, 120, 22)
        self.btn_ok_rect.center = (INTERNAL_W // 2 + 70, 160)
        draw_button(surface, self.app.font_m, "返回", self.btn_back_rect)
        draw_button(surface, self.app.font_m, "确认", self.btn_ok_rect)


class MapSelectState(State):
    def __init__(self, app: App, *, character: CharacterDef) -> None:
        super().__init__(app)
        self.character = character

    def on_enter(self) -> None:
        self.index = 0
        self.dot_rects: list[pygame.Rect] = []
        self.btn_back_rect = pygame.Rect(0, 0, 0, 0)
        self.btn_ok_rect = pygame.Rect(0, 0, 0, 0)
        self.panel_rect = pygame.Rect(0, 0, 0, 0)

    def _confirm(self) -> None:
        self.app.set_state(GameState(self.app, character=self.character, map_def=MAPS[self.index]))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE,):
                self.app.set_state(CharacterSelectState(self.app))
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.index = (self.index - 1) % len(MAPS)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.index = (self.index + 1) % len(MAPS)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._confirm()
            elif pygame.K_1 <= event.key <= pygame.K_9:
                n = event.key - pygame.K_1
                if 0 <= n < len(MAPS):
                    self.index = n
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        for i, r in enumerate(self.dot_rects):
            if r.collidepoint(internal):
                self.index = i
                break

        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            if self.btn_back_rect.collidepoint(internal):
                self.app.set_state(CharacterSelectState(self.app))
                return
            if self.btn_ok_rect.collidepoint(internal) or self.panel_rect.collidepoint(internal):
                self._confirm()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((14, 14, 18))
        draw_text(surface, self.app.font_l, "选择场景", (INTERNAL_W // 2, 18), pygame.Color(240, 240, 240), anchor="center")
        draw_text(surface, self.app.font_s, f"角色：{self.character.name}", (INTERNAL_W // 2, 34), pygame.Color(180, 180, 190), anchor="center")

        m = MAPS[self.index]
        panel = pygame.Rect(10, 44, INTERNAL_W - 20, 88)
        self.panel_rect = panel.copy()
        pygame.draw.rect(surface, (24, 24, 30), panel, border_radius=8)
        pygame.draw.rect(surface, (50, 50, 60), panel, 2, border_radius=8)

        preview = pygame.Surface((panel.width - 20, 38))
        preview.fill(m.base_color)
        rng = random.Random(hash(m.id) & 0xFFFF)
        for _ in range(140):
            x = rng.randrange(0, preview.get_width())
            y = rng.randrange(0, preview.get_height())
            preview.set_at((x, y), m.accent_color)
        surface.blit(preview, (panel.left + 10, panel.top + 10))

        draw_text(surface, self.app.font_m, m.name, (panel.left + 10, panel.top + 52), pygame.Color(240, 240, 240))
        draw_text(surface, self.app.font_s, m.desc, (panel.left + 10, panel.top + 68), pygame.Color(180, 180, 190))

        self.dot_rects = []
        dot_w, dot_h, dot_gap = 14, 8, 10
        total_w = len(MAPS) * dot_w + max(0, (len(MAPS) - 1) * dot_gap)
        start_x = (INTERNAL_W - total_w) // 2
        dots_y = panel.bottom + 6
        for i in range(len(MAPS)):
            c = (240, 240, 240) if i == self.index else (90, 90, 100)
            r = pygame.Rect(start_x + i * (dot_w + dot_gap), dots_y, dot_w, dot_h)
            self.dot_rects.append(r)
            pygame.draw.rect(surface, c, r)

        self.btn_back_rect = pygame.Rect(0, 0, 120, 22)
        self.btn_back_rect.center = (INTERNAL_W // 2 - 70, 160)
        self.btn_ok_rect = pygame.Rect(0, 0, 120, 22)
        self.btn_ok_rect.center = (INTERNAL_W // 2 + 70, 160)
        draw_button(surface, self.app.font_m, "返回", self.btn_back_rect)
        draw_button(surface, self.app.font_m, "开始", self.btn_ok_rect)


class GameState(State):
    def __init__(self, app: App, *, character: CharacterDef, map_def: MapDef) -> None:
        super().__init__(app)
        self.character = character
        self.map_def = map_def

        self.player = Player(character)
        self.enemies: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.swings: list[Swing] = []
        self.weapon_drops: list[WeaponDrop] = []
        self.damage_texts: list[DamageText] = []
        self.explosions: list[ExplosionFx] = []

        self.elapsed_s = 0.0
        self.kills = 0

        self.spawn_cd = 0.0
        self.rng = random.Random(time.time_ns() & 0xFFFFFFFF)
        self.background = self._make_background(map_def)

        self.paused = False
        self.pause_index = 0

        self.level_up_options: list[Upgrade] | None = None
        self.level_up_index = 0
        self.level_up_rects: list[pygame.Rect] = []
        self.pause_item_rects: list[pygame.Rect] = []
        self.grid_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.grid_btn_hover = False

    def on_enter(self) -> None:
        self.player.weapons = [WeaponInstance(WEAPON_DEFS[self.character.start_weapon])]
        self._arrange_weapon_angles()
        c = self.player.center()
        self.weapon_drops = [WeaponDrop("pistol", pygame.Vector2(c.x + 24, c.y))]

    def _arrange_weapon_angles(self) -> None:
        n = len(self.player.weapons)
        if n <= 0:
            return
        for idx, w in enumerate(self.player.weapons):
            w.angle = (idx * (math.tau / n)) % math.tau

    def _make_background(self, map_def: MapDef) -> pygame.Surface:
        bg = pygame.Surface((INTERNAL_W, INTERNAL_H))
        bg.fill(map_def.base_color)
        rng = random.Random(hash(map_def.id) & 0xFFFF)
        for _ in range(2200):
            x = rng.randrange(0, INTERNAL_W)
            y = rng.randrange(0, INTERNAL_H)
            bg.set_at((x, y), map_def.accent_color)
        return bg

    def closest_enemy_to(self, p: pygame.Vector2) -> Enemy | None:
        closest: Enemy | None = None
        best_d2 = 1e18
        for e in self.enemies:
            d2 = (e.center() - p).length_squared()
            if d2 < best_d2:
                best_d2 = d2
                closest = e
        return closest

    def add_damage_text(
        self,
        pos: pygame.Vector2,
        amount: float,
        *,
        color: tuple[int, int, int] = (240, 240, 240),
    ) -> None:
        self.damage_texts.append(DamageText(pos, amount, color=pygame.Color(*color)))

    def _explode(self, pos: pygame.Vector2, proj: Projectile) -> None:
        radius = float(proj.aoe_radius)
        if radius <= 0.0:
            return
        base_damage = float(proj.damage) * float(proj.aoe_damage_mult)
        kb = float(proj.knockback)
        self.explosions.append(ExplosionFx(pos, radius=radius, color=(255, 170, 90)))

        for e in self.enemies:
            d = e.center() - pos
            dist = d.length()
            if dist > radius:
                continue
            t = 1.0 if radius <= 0.001 else clamp(1.0 - dist / radius, 0.0, 1.0)
            dmg = base_damage * (0.6 + 0.4 * t)
            e.hp -= dmg
            self.add_damage_text(e.center(), dmg, color=(255, 210, 160))
            if kb > 0.0:
                if d.length_squared() > 0.01:
                    e.vel += d.normalize() * kb * (0.5 + 0.8 * t)
                else:
                    e.vel += pygame.Vector2(1, 0) * kb * 0.5

    def _spawn_enemy(self) -> None:
        enemy_id = self.rng.choice(self.map_def.enemy_pool)
        enemy_def = ENEMY_DEFS[enemy_id]

        side = self.rng.randrange(4)
        if side == 0:
            pos = pygame.Vector2(-enemy_def.size - 2, self.rng.randrange(0, INTERNAL_H - enemy_def.size))
        elif side == 1:
            pos = pygame.Vector2(INTERNAL_W + 2, self.rng.randrange(0, INTERNAL_H - enemy_def.size))
        elif side == 2:
            pos = pygame.Vector2(self.rng.randrange(0, INTERNAL_W - enemy_def.size), -enemy_def.size - 2)
        else:
            pos = pygame.Vector2(self.rng.randrange(0, INTERNAL_W - enemy_def.size), INTERNAL_H + 2)

        self.enemies.append(Enemy(enemy_def, pos))

    def _game_over(self) -> None:
        self.app.set_state(
            GameOverState(
                self.app,
                character=self.character,
                map_def=self.map_def,
                level=self.player.level,
                kills=self.kills,
                elapsed_s=self.elapsed_s,
            )
        )

    def _add_weapon(self, weapon_id: str) -> bool:
        if len(self.player.weapons) >= 6:
            return False
        self.player.weapons.append(WeaponInstance(WEAPON_DEFS[weapon_id]))
        self._arrange_weapon_angles()
        return True

    def _upgrade_weapon(self, weapon: WeaponInstance) -> None:
        weapon.level += 1

    def _generate_upgrades(self) -> list[Upgrade]:
        p = self.player
        pool: list[Upgrade] = []

        def add(u: Upgrade, weight: int = 1) -> None:
            for _ in range(weight):
                pool.append(u)

        def vitality(pl: Player, _g: GameState) -> None:
            pl.max_hp += 12
            pl.hp = min(pl.max_hp, pl.hp + 12)

        def power(pl: Player, _g: GameState) -> None:
            pl.damage_mult *= 1.10

        def haste(pl: Player, _g: GameState) -> None:
            pl.cooldown_mult *= 0.90

        def agility(pl: Player, _g: GameState) -> None:
            pl.move_speed *= 1.10

        def regen(pl: Player, _g: GameState) -> None:
            pl.regen += 0.25

        def xp_gain(pl: Player, _g: GameState) -> None:
            pl.xp_mult *= 1.15

        def heal(pl: Player, _g: GameState) -> None:
            pl.hp = min(pl.max_hp, pl.hp + pl.max_hp * 0.30)

        add(Upgrade("力量 +10%", "所有武器伤害提高。", power), weight=4)
        add(Upgrade("急速 -10%", "所有武器冷却降低。", haste), weight=4)
        add(Upgrade("体魄 +12", "最大生命提高并回复同等生命。", vitality), weight=3)
        add(Upgrade("敏捷 +10%", "移动速度提高。", agility), weight=3)
        add(Upgrade("回复 +0.25/s", "每秒恢复更多生命。", regen), weight=2)
        add(Upgrade("经验 +15%", "击杀获得更多经验。", xp_gain), weight=2)
        add(Upgrade("治疗 30%", "立刻回复 30% 最大生命。", heal), weight=2)

        if p.weapons:
            w = self.rng.choice(p.weapons)

            def up_weapon(_pl: Player, g: GameState, ww: WeaponInstance = w) -> None:
                g._upgrade_weapon(ww)

            add(Upgrade(f"升级：{w.weapon_def.name} +1", "提升该武器等级（伤害/冷却更强）。", up_weapon), weight=5)

        picked: list[Upgrade] = []
        seen: set[str] = set()
        for _ in range(200):
            if len(picked) >= 3 or not pool:
                break
            u = self.rng.choice(pool)
            if u.title in seen:
                continue
            seen.add(u.title)
            picked.append(u)

        while len(picked) < 3:
            picked.append(Upgrade("空选项", "（demo占位）", lambda _pl, _g: None))
        return picked

    def _level_up(self) -> None:
        self.level_up_options = self._generate_upgrades()
        self.level_up_index = 0

    def _pick_level_up(self) -> None:
        if self.level_up_options is None:
            return
        choice = self.level_up_options[self.level_up_index]
        choice.apply(self.player, self)
        self.level_up_options = None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if self.level_up_options is not None:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.level_up_index = (self.level_up_index - 1) % 3
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.level_up_index = (self.level_up_index + 1) % 3
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.level_up_index = event.key - pygame.K_1
                    self._pick_level_up()
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._pick_level_up()
                return

            if event.key in (pygame.K_ESCAPE,):
                self.paused = not self.paused
                self.pause_index = 0
                return

            if not self.paused:
                return

            if event.key in (pygame.K_UP, pygame.K_w):
                self.pause_index = (self.pause_index - 1) % 2
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.pause_index = (self.pause_index + 1) % 2
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.pause_index == 0:
                    self.paused = False
                else:
                    self.app.set_state(MainMenuState(self.app))
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        self.grid_btn_hover = self.grid_btn_rect.collidepoint(internal)
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and self.grid_btn_hover
        ):
            self.app.set_config(show_grid=not self.app.config.show_grid)
            return

        if self.level_up_options is not None:
            for i, rect in enumerate(self.level_up_rects):
                if rect.collidepoint(internal):
                    self.level_up_index = i
                    if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                        self._pick_level_up()
                    break
            return

        if not self.paused:
            return

        for i, rect in enumerate(self.pause_item_rects):
            if rect.collidepoint(internal):
                self.pause_index = i
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    if i == 0:
                        self.paused = False
                    else:
                        self.app.set_state(MainMenuState(self.app))
                break

    def update(self, dt: float) -> None:
        if self.paused or self.level_up_options is not None:
            return

        self.elapsed_s += dt

        p = self.player
        keys = pygame.key.get_pressed()
        move = pygame.Vector2(0, 0)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move.x += 1
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move.y += 1
        if move.length_squared() > 0:
            move = move.normalize()
            p.last_move_dir = pygame.Vector2(move)

        p.update_visual(dt, move)
        p.pos += move * p.move_speed * dt
        p.pos.x = clamp(p.pos.x, 0, INTERNAL_W - p.w)
        p.pos.y = clamp(p.pos.y, 0, INTERNAL_H - p.h)

        pr_pick = p.rect()
        for drop in list(self.weapon_drops):
            drop.bob_t += dt
            if drop.rect().colliderect(pr_pick):
                if self._add_weapon(drop.weapon_id):
                    self.weapon_drops.remove(drop)

        p.invuln_s = max(0.0, p.invuln_s - dt)
        p.hurt_s = max(0.0, p.hurt_s - dt)
        if p.regen > 0:
            p.hp = min(p.max_hp, p.hp + p.regen * dt)

        for w in p.weapons:
            w.update(dt, p, self)

        cap = 90
        self.spawn_cd -= dt
        if self.spawn_cd <= 0.0 and len(self.enemies) < cap:
            self._spawn_enemy()
            base = 0.7
            accel = min(0.55, self.elapsed_s * 0.01 + (p.level - 1) * 0.01)
            self.spawn_cd = max(0.12, base - accel)

        pr = p.rect()
        for e in self.enemies:
            e.update_visual(dt)
            to_p = (p.center() - e.center())
            v = pygame.Vector2(0, 0)
            if to_p.length_squared() > 0.01:
                v = to_p.normalize() * e.enemy_def.speed

            e.pos += (v + e.vel) * dt
            drag = math.pow(0.82, dt * 60.0)
            e.vel *= drag
            if e.vel.length_squared() < 0.01:
                e.vel.update(0.0, 0.0)
            if e.rect().colliderect(pr) and p.invuln_s <= 0.0:
                p.hp -= e.enemy_def.contact_damage
                p.invuln_s = 0.45
                p.hurt_s = 0.20
                if to_p.length_squared() > 0.01:
                    p.pos += to_p.normalize() * 10
                    p.pos.x = clamp(p.pos.x, 0, INTERNAL_W - p.w)
                    p.pos.y = clamp(p.pos.y, 0, INTERNAL_H - p.h)

        for s in list(self.swings):
            s.ttl_s -= dt
            if s.ttl_s <= 0:
                self.swings.remove(s)
                continue
            for e in self.enemies:
                if e.uid in s.hit_uids:
                    continue
                if s.rect.colliderect(e.rect()):
                    s.hit_uids.add(e.uid)
                    e.hp -= s.damage
                    self.add_damage_text(e.center(), s.damage, color=(255, 240, 200))

        for proj in list(self.projectiles):
            proj.ttl_s -= dt
            if proj.ttl_s <= 0:
                self.projectiles.remove(proj)
                continue

            proj.pos += proj.vel * dt
            if proj.pos.x < -10 or proj.pos.x > INTERNAL_W + 10 or proj.pos.y < -10 or proj.pos.y > INTERNAL_H + 10:
                self.projectiles.remove(proj)
                continue

            r = proj.rect()
            for e in self.enemies:
                if e.uid in proj.hit_uids:
                    continue
                if r.colliderect(e.rect()):
                    proj.hit_uids.add(e.uid)
                    if proj.aoe_radius > 0.0:
                        self._explode(e.center(), proj)
                        if proj in self.projectiles:
                            self.projectiles.remove(proj)
                        break
                    dmg = proj.damage
                    e.hp -= dmg
                    self.add_damage_text(e.center(), dmg, color=(255, 255, 255))
                    if proj.knockback > 0.0:
                        d = pygame.Vector2(proj.vel)
                        if d.length_squared() > 0.01:
                            d = d.normalize()
                        else:
                            d = pygame.Vector2(1, 0)
                        kb = proj.knockback * (1.25 if e.enemy_def.id == "slime" else 1.0)
                        e.vel += d * kb
                    if proj.pierce > 0:
                        proj.pierce -= 1
                    else:
                        self.projectiles.remove(proj)
                    break

        for t in list(self.damage_texts):
            t.ttl_s -= dt
            if t.ttl_s <= 0:
                self.damage_texts.remove(t)
                continue
            t.vel.y += 70.0 * dt
            t.pos += t.vel * dt

        for fx in list(self.explosions):
            fx.ttl_s -= dt
            if fx.ttl_s <= 0:
                self.explosions.remove(fx)

        for e in list(self.enemies):
            if e.hp > 0:
                continue
            self.enemies.remove(e)
            self.kills += 1
            p.xp += e.enemy_def.xp_value * p.xp_mult
            if len(self.weapon_drops) < 12 and self.rng.random() < 0.18:
                weapon_id = self.rng.choice(tuple(WEAPON_DEFS.keys()))
                self.weapon_drops.append(WeaponDrop(weapon_id, pygame.Vector2(e.center())))

        if p.hp <= 0:
            self._game_over()
            return

        if p.xp >= p.xp_to_next:
            p.xp -= p.xp_to_next
            p.level += 1
            p.xp_to_next = p.xp_to_next * 1.20 + 3.0
            self._level_up()

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.background, (0, 0))

        for drop in self.weapon_drops:
            weapon_def = WEAPON_DEFS.get(drop.weapon_id)
            color = (200, 200, 200) if weapon_def is None else weapon_def.color
            bob = math.sin(drop.bob_t * 4.0) * 1.5
            cx, cy = int(drop.pos.x), int(drop.pos.y + bob)
            if weapon_def is not None and weapon_def.sprite is not None:
                spr = weapon_def.sprite
                r = spr.get_rect(center=(cx, cy))
                bg = r.inflate(4, 4)
                pygame.draw.rect(surface, (20, 20, 24), bg)
                pygame.draw.rect(surface, (0, 0, 0), bg, 1)
                surface.blit(spr, r)
            else:
                r = pygame.Rect(0, 0, 8, 8)
                r.center = (cx, cy)
                pygame.draw.rect(surface, color, r)
                pygame.draw.rect(surface, (0, 0, 0), r, 1)

        for e in self.enemies:
            if e.sprite is not None:
                surface.blit(e.sprite, (int(e.pos.x), int(e.pos.y)))
            else:
                pygame.draw.rect(surface, e.enemy_def.color, e.rect())

        for proj in self.projectiles:
            r = proj.rect()
            pygame.draw.rect(surface, proj.color, r)
            pygame.draw.rect(surface, (0, 0, 0), r, 1)
            if proj.vel.length_squared() > 0.01:
                speed2 = proj.vel.length_squared()
                tail_len = 3.0
                if proj.aoe_radius > 0.0:
                    tail_len = 7.0
                elif proj.pierce >= 3 or speed2 >= 600.0 * 600.0:
                    tail_len = 10.0
                elif proj.size <= 1:
                    tail_len = 4.0
                tail = proj.vel.normalize() * -tail_len
                pygame.draw.line(
                    surface,
                    (255, 255, 255),
                    (int(proj.pos.x), int(proj.pos.y)),
                    (int(proj.pos.x + tail.x), int(proj.pos.y + tail.y)),       
                    1,
                )

        p = self.player
        ppos = (int(p.pos.x + p.draw_offset.x), int(p.pos.y + p.draw_offset.y))
        spr = p.sprite
        if p.hurt_s > 0.0:
            img = spr.copy()
            img.fill((140, 20, 20), special_flags=pygame.BLEND_RGB_ADD)
            surface.blit(img, ppos)
        elif p.invuln_s > 0.0 and int(p.invuln_s * 20) % 2 == 0:
            img = spr.copy()
            img.fill((90, 10, 10), special_flags=pygame.BLEND_RGB_ADD)
            surface.blit(img, ppos)
        else:
            surface.blit(spr, ppos)
        fx = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)     
        for boom in self.explosions:
            t = clamp(boom.ttl_s / 0.22, 0.0, 1.0)
            a0 = int(120 * t)
            a1 = int(220 * t)
            x, y = int(boom.pos.x), int(boom.pos.y)
            r = max(1, int(boom.radius))
            pygame.draw.circle(fx, (boom.color.r, boom.color.g, boom.color.b, a0), (x, y), r)
            pygame.draw.circle(fx, (255, 255, 255, a1), (x, y), max(1, int(r * 0.35)))
        for w in p.weapons:
            if w.weapon_def.kind != "melee" or len(w.trail) < 2:
                continue
            for i in range(1, len(w.trail)):
                p0, age0 = w.trail[i - 1]
                p1, age1 = w.trail[i]
                age = max(age0, age1)
                a = int(220 * (1.0 - age / 0.18))
                if a <= 0:
                    continue
                x0, y0 = int(p0.x), int(p0.y)
                x1, y1 = int(p1.x), int(p1.y)
                pygame.draw.line(fx, (255, 255, 255, max(0, a // 4)), (x0, y0), (x1, y1), 6)
                pygame.draw.line(fx, (255, 255, 255, a), (x0, y0), (x1, y1), 3)
        surface.blit(fx, (0, 0))

        for w in p.weapons:
            if w.weapon_def.kind == "melee":
                if w.sword_poly:
                    d = w.sword_tip - w.sword_base
                    if d.length_squared() > 0.01:
                        d = d.normalize()
                        pygame.draw.line(surface, (160, 120, 80), w.sword_base, w.sword_base + d * 3, 3)
                    pygame.draw.polygon(surface, w.weapon_def.color, w.sword_poly)
                    pygame.draw.polygon(surface, (0, 0, 0), w.sword_poly, 1)
                else:
                    pygame.draw.rect(surface, w.weapon_def.color, w.orbit_rect)
                    pygame.draw.rect(surface, (0, 0, 0), w.orbit_rect, 1)
            elif w.weapon_def.kind == "gun":
                center = pygame.Vector2(w.orbit_rect.centerx, w.orbit_rect.centery)
                ang = w.aim_angle + (w.reload_spin if w.reload_left > 0.0 else 0.0)
                aim_dir = pygame.Vector2(math.cos(w.aim_angle), math.sin(w.aim_angle))
                if aim_dir.length_squared() <= 0.0001:
                    aim_dir = pygame.Vector2(1, 0)
                spr = w.weapon_def.sprite
                if spr is not None:
                    flip = float(aim_dir.x) < 0.0
                    if flip:
                        spr = flip_x_pixel_sprite(spr)
                    deg = -math.degrees(ang)
                    if flip:
                        deg -= 180.0
                    step = 15.0 if w.reload_left > 0.0 else 5.0
                    rot = rotate_pixel_sprite(spr, deg, step_deg=step)
                    kick = 0.0
                    if w.muzzle_flash_s > 0.0 and w.reload_left <= 0.0:
                        flash_total = 0.10 if w.weapon_def.aoe_radius > 0.0 else 0.06
                        t = clamp(w.muzzle_flash_s / max(0.001, flash_total), 0.0, 1.0)
                        kick = 2.0 * t
                    draw_center = center - aim_dir * kick
                    surface.blit(rot, rot.get_rect(center=(int(round(draw_center.x)), int(round(draw_center.y)))))

                    if w.muzzle_flash_s > 0.0 and w.reload_left <= 0.0:
                        flash_total = 0.10 if w.weapon_def.aoe_radius > 0.0 else 0.06
                        t = clamp(w.muzzle_flash_s / max(0.001, flash_total), 0.0, 1.0)
                        tip_dist = max(6.0, spr.get_width() * 0.5 - 1.0)
                        tip = draw_center + aim_dir * tip_dist
                        x, y = int(round(tip.x)), int(round(tip.y))
                        rad = 1 + int(2 * t)
                        core = (255, 255, 255)
                        glow = (255, 220, 120) if w.weapon_def.aoe_radius <= 0.0 else (255, 160, 90)
                        pygame.draw.circle(surface, glow, (x, y), rad)
                        pygame.draw.circle(surface, core, (x, y), max(1, rad - 1))
                else:
                    pygame.draw.rect(surface, w.weapon_def.color, w.orbit_rect) 
                    pygame.draw.rect(surface, (0, 0, 0), w.orbit_rect, 1)       

                if w.reload_left > 0.0 and w.reload_total > 0.0:
                    t = 1.0 - (w.reload_left / max(0.001, w.reload_total))
                    t = clamp(t, 0.0, 1.0)
                    bar = pygame.Rect(0, 0, 16, 3)
                    bar.center = (int(center.x), int(center.y - 9))
                    pygame.draw.rect(surface, (10, 10, 12), bar)
                    fill = pygame.Rect(bar.x + 1, bar.y + 1, int((bar.w - 2) * t), bar.h - 2)
                    pygame.draw.rect(surface, (120, 220, 140), fill)
                    pygame.draw.rect(surface, (0, 0, 0), bar, 1)
            else:
                pygame.draw.rect(surface, w.weapon_def.color, w.orbit_rect)
                pygame.draw.rect(surface, (0, 0, 0), w.orbit_rect, 1)

        for t in self.damage_texts:
            x, y = int(t.pos.x), int(t.pos.y)
            draw_text(surface, self.app.font_s, t.text, (x + 1, y + 1), pygame.Color(0, 0, 0), anchor="center")
            draw_text(surface, self.app.font_s, t.text, (x, y), t.color, anchor="center")

        self._draw_ui(surface)
        if self.paused:
            self._draw_pause(surface)
        if self.level_up_options is not None:
            self._draw_level_up(surface)

    def _draw_ui(self, surface: pygame.Surface) -> None:
        p = self.player

        hp_w, hp_h = 150, 8
        x, y = 10, 10
        pygame.draw.rect(surface, (20, 20, 24), pygame.Rect(x, y, hp_w, hp_h))
        fill = int(hp_w * (p.hp / max(1, p.max_hp)))
        pygame.draw.rect(surface, (220, 70, 70), pygame.Rect(x, y, fill, hp_h))
        pygame.draw.rect(surface, (70, 70, 80), pygame.Rect(x, y, hp_w, hp_h), 1)
        draw_text(surface, self.app.font_s, f"HP {int(p.hp)}/{p.max_hp}", (x + hp_w + 8, y - 2), pygame.Color(230, 230, 240))

        xp_w, xp_h = 220, 6
        x2, y2 = 10, 22
        pygame.draw.rect(surface, (20, 20, 24), pygame.Rect(x2, y2, xp_w, xp_h))
        fill2 = int(xp_w * (p.xp / max(1.0, p.xp_to_next)))
        pygame.draw.rect(surface, (90, 190, 120), pygame.Rect(x2, y2, fill2, xp_h))
        pygame.draw.rect(surface, (70, 70, 80), pygame.Rect(x2, y2, xp_w, xp_h), 1)
        draw_text(surface, self.app.font_s, f"Lv {p.level}", (x2 + xp_w + 8, y2 - 3), pygame.Color(230, 230, 240))

        draw_text(surface, self.app.font_s, f"{self.map_def.name}", (INTERNAL_W - 10, 10), pygame.Color(230, 230, 240), anchor="topright")
        draw_text(surface, self.app.font_s, f"时间 {format_time(self.elapsed_s)}", (INTERNAL_W - 10, 26), pygame.Color(200, 200, 210), anchor="topright")       
        draw_text(surface, self.app.font_s, f"击杀 {self.kills}", (INTERNAL_W - 10, 42), pygame.Color(200, 200, 210), anchor="topright")
        self.grid_btn_rect = pygame.Rect(0, 0, 84, 18)
        self.grid_btn_rect.topright = (INTERNAL_W - 10, 54)
        draw_button(
            surface,
            self.app.font_s,
            f"网格:{'开' if self.app.config.show_grid else '关'}",
            self.grid_btn_rect,
            selected=self.grid_btn_hover or self.app.config.show_grid,
        )

        wx = 10
        wy = INTERNAL_H - 18
        for w in p.weapons[:6]:
            pygame.draw.rect(surface, w.weapon_def.color, pygame.Rect(wx, wy, 12, 8))
            pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(wx, wy, 12, 8), 1)
            draw_text(surface, self.app.font_s, f"{w.level}", (wx + 6, wy - 12), pygame.Color(240, 240, 240), anchor="center")
            wx += 18

    def _draw_pause(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        draw_text(surface, self.app.font_l, "暂停", (INTERNAL_W // 2, 40), pygame.Color(240, 240, 240), anchor="center")

        self.pause_item_rects = []
        items = ["继续", "返回主菜单"]
        y0 = 84
        for i, label in enumerate(items):
            rect = pygame.Rect(0, 0, 200, 22)
            rect.center = (INTERNAL_W // 2, y0 + i * 30)
            self.pause_item_rects.append(draw_button(surface, self.app.font_m, label, rect, selected=(i == self.pause_index)))

        draw_text(surface, self.app.font_s, "方向键/鼠标选择 | Enter/左键确认 | Esc返回", (INTERNAL_W // 2, INTERNAL_H - 12), pygame.Color(180, 180, 190), anchor="center")

    def _draw_level_up(self, surface: pygame.Surface) -> None:
        if self.level_up_options is None:
            return
        overlay = pygame.Surface((INTERNAL_W, INTERNAL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        draw_text(surface, self.app.font_l, "升级！选择一项", (INTERNAL_W // 2, 18), pygame.Color(240, 240, 240), anchor="center")

        card_w, card_h = INTERNAL_W - 20, 36
        x = 10
        y0 = 38
        gap = 8
        self.level_up_rects = []
        for i, u in enumerate(self.level_up_options):
            y = y0 + i * (card_h + gap)
            rect = pygame.Rect(x, y, card_w, card_h)
            self.level_up_rects.append(rect)
            bg = (34, 34, 42) if i == self.level_up_index else (24, 24, 30)
            pygame.draw.rect(surface, bg, rect, border_radius=8)
            border = (90, 90, 110) if i == self.level_up_index else (60, 60, 74)
            pygame.draw.rect(surface, border, rect, 2, border_radius=8)

            draw_text(surface, self.app.font_m, u.title, (rect.left + 10, rect.top + 4), pygame.Color(240, 240, 240))
            draw_text(surface, self.app.font_s, u.desc, (rect.left + 10, rect.top + 22), pygame.Color(200, 200, 210))
            draw_text(surface, self.app.font_s, f"[{i+1}]", (rect.right - 12, rect.centery), pygame.Color(160, 160, 170), anchor="center")

        draw_text(surface, self.app.font_s, "按 1/2/3 或点击选择", (INTERNAL_W // 2, INTERNAL_H - 12), pygame.Color(180, 180, 190), anchor="center")


class GameOverState(State):
    def __init__(self, app: App, *, character: CharacterDef, map_def: MapDef, level: int, kills: int, elapsed_s: float) -> None:
        super().__init__(app)
        self.character = character
        self.map_def = map_def
        self.level = level
        self.kills = kills
        self.elapsed_s = elapsed_s
        self.index = 0
        self.item_rects: list[pygame.Rect] = []

    def _activate(self) -> None:
        if self.index == 0:
            self.app.set_state(GameState(self.app, character=self.character, map_def=self.map_def))
        else:
            self.app.set_state(MainMenuState(self.app))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.index = (self.index - 1) % 2
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.index = (self.index + 1) % 2
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate()
            return

        if event.type not in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            return
        if not hasattr(event, "pos"):
            return

        internal = self.app.screen_to_internal(event.pos)
        if internal is None:
            return

        for i, rect in enumerate(self.item_rects):
            if rect.collidepoint(internal):
                self.index = i
                if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
                    self._activate()
                break

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        draw_text(surface, self.app.font_l, "游戏结束", (INTERNAL_W // 2, 26), pygame.Color(240, 240, 240), anchor="center")
        draw_text(
            surface,
            self.app.font_s,
            f"角色：{self.character.name}  场景：{self.map_def.name}",
            (INTERNAL_W // 2, 46),
            pygame.Color(180, 180, 190),
            anchor="center",
        )
        draw_text(surface, self.app.font_s, f"时间：{format_time(self.elapsed_s)}", (INTERNAL_W // 2, 62), pygame.Color(200, 200, 210), anchor="center")
        draw_text(surface, self.app.font_s, f"击杀：{self.kills}  等级：{self.level}", (INTERNAL_W // 2, 78), pygame.Color(200, 200, 210), anchor="center")

        self.item_rects = []
        items = ["再来一局", "返回主菜单"]
        y0 = 112
        for i, label in enumerate(items):
            rect = pygame.Rect(0, 0, 200, 22)
            rect.center = (INTERNAL_W // 2, y0 + i * 30)
            self.item_rects.append(draw_button(surface, self.app.font_m, label, rect, selected=(i == self.index)))

        draw_text(surface, self.app.font_s, "方向键/鼠标选择 | Enter/左键确认", (INTERNAL_W // 2, INTERNAL_H - 12), pygame.Color(170, 170, 180), anchor="center")


def main() -> int:
    try:
        App().run()
        return 0
    except Exception as exc:
        try:
            pygame.quit()
        except Exception:
            pass
        print("Fatal error:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
