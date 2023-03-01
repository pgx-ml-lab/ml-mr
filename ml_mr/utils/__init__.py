# flake8: noqa
from typing import Tuple, Optional

from .temperature_scaling.temperature_scaling import temperature_scale
from .nn import MLP, build_mlp


def parse_project_and_run_name(s: str) -> Tuple[str, Optional[str]]:
    """Utility function to parse project and run name.
    
    This is used mostly for logging to trackers like WandB. The expected format
    is project:run_name.

    """
    try:
        project, run_name = s.split(":")
        return project, run_name
    except ValueError:
        return s, None  # Assume only project name is provided.
