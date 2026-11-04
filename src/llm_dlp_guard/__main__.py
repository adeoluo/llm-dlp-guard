"""Entrypoint: `python -m llm_dlp_guard`."""

from __future__ import annotations

from llm_dlp_guard import __version__
from llm_dlp_guard.configuration import settings


def main() -> None:
    header = f"Llm Dlp Guard v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
