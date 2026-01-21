from __future__ import annotations

import game


def main() -> int:
    # Compatibility entrypoint. The game now contains the xiaotou-style scaled
    # weapon overlay directly in `game.py`, so no runtime monkey-patching is
    # needed here.
    return int(game.main())


if __name__ == "__main__":
    raise SystemExit(main())
