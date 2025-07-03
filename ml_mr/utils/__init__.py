# flake8: noqa
import sys
import os
import datetime
import subprocess
import argparse
from typing import Tuple, Optional, Dict, Any

import torch

from ..log_utils import critical


def _cat(*tensors) -> torch.Tensor:
    """Simple column concatenation of tensors with null checking."""
    return torch.hstack(
        [tens for tens in tensors if tens is not None and tens.numel() > 0]
    )


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


def get_git_hash() -> Optional[str]:
    """
    Detects if the package is tracked by git and returns the current git hash.
    If the package is not a git repository or git is unavailable, returns None.

    """
    try:
        # Get the top-level directory of the package
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # Check if the directory is part of a git repository
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=package_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # Get the current git commit hash
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package_dir, stderr=subprocess.DEVNULL
        )
        return git_hash.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Return None if git is not available or the directory is not tracked by git
        return None


def initialize_meta() -> Dict[str, Any]:
    """Used to initialize a model metadata dict with some generally useful
    values.

    """
    meta = {}
    meta["date"] = str(datetime.datetime.now())
    git_hash = get_git_hash()
    if git_hash is not None:
        meta["git_hash"] = git_hash

    return meta
