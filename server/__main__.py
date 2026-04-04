"""Entry point: uv run server  /  python -m server"""
from __future__ import annotations
import os
import sys

# Ensure project root is on path before importing app
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
