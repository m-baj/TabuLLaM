"""Serialization utilities for tabular data."""

import pandas as pd
from typing import Dict, Any, Optional


def key_value_serialize(
    features: Dict[str, Any],
    label: Optional[Dict[str, Any]] = None
) -> str:
    """
    Serialize a row of features (and optionally a label) to a string.

    Parameters
    ----------
    features : dict
        Dictionary of feature name to value
    label : dict, optional
        Dictionary with target column name and value

    Returns
    -------
    str
        Serialized string in format "key1: val1 | key2: val2 => target: value"

    Examples
    --------
    >>> key_value_serialize({'age': 25, 'income': 50000})
    'age: 25 | income: 50000'

    >>> key_value_serialize({'age': 25}, {'churn': 'yes'})
    'age: 25 => churn: yes'
    """
    features_str = " | ".join(
        f"{k}: {v}" for k, v in features.items() if pd.notna(v)
    )

    if label:
        target_col, target_val = next(iter(label.items()))
        return f"{features_str} => {target_col}: {target_val}"

    return features_str
