"""Allow ``python -m memcore`` as an alias for ``python -m memcore.cli``."""

from __future__ import annotations

from memcore.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
