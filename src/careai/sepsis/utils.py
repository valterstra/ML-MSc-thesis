"""
Utility functions for the sepsis pipeline.
Adapted from ai_clinician/preprocessing/utils.py — only import path changed.
"""
import os
import pandas as pd
from careai.sepsis.columns import C_ICUSTAYID, DTYPE_SPEC, STAY_ID_OPTIONAL_DTYPE_SPEC


def load_csv(*file_paths, null_icustayid=False, **kwargs):
    """Load the first existing CSV from a list of candidate paths."""
    for path in file_paths:
        if os.path.exists(path):
            spec = DTYPE_SPEC if not null_icustayid else STAY_ID_OPTIONAL_DTYPE_SPEC
            return pd.read_csv(path, dtype=spec, **kwargs)
    raise FileNotFoundError(", ".join(file_paths))


def load_intermediate_or_raw_csv(data_dir, file_name):
    return load_csv(
        os.path.join(data_dir, "intermediates", file_name),
        os.path.join(data_dir, "raw_data", file_name),
    )
