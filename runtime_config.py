"""Validated runtime configuration for the Gradio demo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860


@dataclass(frozen=True)
class LaunchConfig:
    server_name: str = DEFAULT_SERVER_NAME
    server_port: int = DEFAULT_SERVER_PORT
    share: bool = False


def _parse_boolean(name: str, raw_value: str) -> bool:
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, off"
    )


def _parse_port(raw_value: str) -> int:
    try:
        port = int(raw_value.strip())
    except ValueError as error:
        raise ValueError("DARIJA_DEMO_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("DARIJA_DEMO_PORT must be between 1 and 65535")
    return port


def load_launch_config(environ: Mapping[str, str] | None = None) -> LaunchConfig:
    """Read launch settings with private-by-default, validated values."""
    values = os.environ if environ is None else environ

    server_name = values.get("DARIJA_DEMO_HOST", DEFAULT_SERVER_NAME).strip()
    if not server_name:
        raise ValueError("DARIJA_DEMO_HOST must not be blank")

    server_port = _parse_port(
        values.get("DARIJA_DEMO_PORT", str(DEFAULT_SERVER_PORT))
    )
    share = _parse_boolean(
        "DARIJA_DEMO_SHARE", values.get("DARIJA_DEMO_SHARE", "false")
    )

    return LaunchConfig(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
