# flake8: noqa
import sys
import argparse
from typing import Tuple, Optional

from .temperature_scaling.temperature_scaling import temperature_scale
from ..logging import critical


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


def default_validate_args(args: argparse.Namespace) -> None:
    if args.genotypes_backend is not None and args.sample_id_col is None:
        critical(
            "When providing a genotypes dataset for the instrument, a "
            "sample id column needs to be provided using --sample-id-col "
            "so that the individuals can be matched between the genotypes "
            "and data file."
        )
        sys.exit(1)

    if args.validation_proportion < 0 or args.validation_proportion > 1:
        critical("--validation-proportion should be between 0 and 1.")
        sys.exit(1)

    if args.genotypes_backend is None and len(args.instruments) == 0:
        critical("No instruments provided.")
        sys.exit(1)
