"""
Configuration loader module for MP190 MLOps pipeline.

Reads and validates params.yaml using yaml.safe_load to provide
centralized, reproducible configurations across all stages.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Assuming config.py is inside src/
    return Path(__file__).resolve().parent.parent


def load_params(config_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load project configuration from params.yaml using yaml.safe_load.

    Args:
        config_path: Optional path to params.yaml. Defaults to repo root params.yaml.

    Returns:
        Dictionary containing project configuration parameters.
    """
    if config_path is None:
        config_path = get_project_root() / "params.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if not isinstance(params, dict):
        raise ValueError(f"Expected dictionary from {config_path}, got {type(params)}")

    return params


# Default global configuration instance
PARAMS = load_params()
PROJECT_ROOT = get_project_root()
