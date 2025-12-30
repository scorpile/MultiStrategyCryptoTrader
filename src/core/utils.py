"""Utility helpers (timezones, logging helpers, config loading, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration and validate required sections."""
    raise NotImplementedError("Configuration loading/parsing must be implemented.")


def resolve_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge environment variables (API keys, secrets) into the config."""
    raise NotImplementedError("Environment resolution logic is pending implementation.")


def setup_structured_logging(config: Dict[str, Any]) -> None:
    """Configure structured logging (JSON/text) based on config settings."""
    raise NotImplementedError("Logging setup will be defined in an integration phase.")


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    raise NotImplementedError("Timestamp helper has not been implemented yet.")
