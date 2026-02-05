"""
Dataset loading utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import train_test_split

from .config import DATASETS_INFO, get_dataset_info


def filter_rare_classes(
    X: pd.DataFrame,
    y: pd.Series,
    embeddings: Optional[np.ndarray] = None,
    min_samples: int = 2
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, np.ndarray]]:
    """
    Filter out classes that have fewer than min_samples instances.

    This is necessary for stratified train-test splitting, which requires
    at least 2 samples per class.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target labels
    embeddings : np.ndarray, optional
        Pre-computed embeddings aligned with X and y
    min_samples : int, default=2
        Minimum number of samples required per class

    Returns
    -------
    X_filtered, y_filtered [, embeddings_filtered]
        Filtered data with rare classes removed

    Warnings
    --------
    Prints a warning if any classes are filtered out.
    """
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    if rare_classes:
        print(f"  [WARNING] Filtering out classes with < {min_samples} samples: {rare_classes}")
        print(f"            Class counts: {dict(class_counts[rare_classes])}")

        mask = ~y.isin(rare_classes)
        X_filtered = X[mask].reset_index(drop=True)
        y_filtered = y[mask].reset_index(drop=True)

        if embeddings is not None:
            embeddings_filtered = embeddings[mask.values]
            return X_filtered, y_filtered, embeddings_filtered

        return X_filtered, y_filtered

    if embeddings is not None:
        return X, y, embeddings

    return X, y


def get_data_dir() -> Path:
    """Get the data directory path."""
    current = Path(__file__).parent.parent
    data_dir = current / 'data'
    if data_dir.exists():
        return data_dir
    raise FileNotFoundError(f"Data directory not found at {data_dir}")


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset by name from local parquet/csv file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load

    Returns
    -------
    df : pd.DataFrame
        The loaded dataframe
    target_column : str
        Name of the target column
    """
    info = get_dataset_info(dataset_name)
    if not info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_dir = get_data_dir()
    file_path = data_dir / f"{dataset_name}.parquet"

    if not file_path.exists():
        file_path = data_dir / f"{dataset_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found. "
            f"Expected file at data/{dataset_name}.parquet or data/{dataset_name}.csv"
        )

    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    print(f"[datasets] Loaded from {file_path}, {len(df)} rows")

    return df, info['target_column']


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    exclude_columns : list, optional
        Additional columns to exclude (e.g., embedding columns)

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    """
    exclude = {target_column}
    if exclude_columns:
        exclude.update(exclude_columns)

    for col in df.columns:
        if 'embedding' in col.lower():
            exclude.add(col)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df[target_column]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Automatically filters out classes with fewer than 2 samples
    to enable stratified splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_filtered, y_filtered = filter_rare_classes(X, y, min_samples=2)

    return train_test_split(
        X_filtered, y_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=y_filtered
    )


def sample_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sample n_samples from X, y with stratification.

    If n_samples >= len(X), returns original data unchanged.
    Automatically filters out classes with fewer than 2 samples
    to enable stratified sampling.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target labels
    n_samples : int
        Number of samples to select
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_sampled, y_sampled
    """
    X_filtered, y_filtered = filter_rare_classes(X, y, min_samples=2)

    if n_samples >= len(X_filtered):
        return X_filtered, y_filtered

    _, X_sampled, _, y_sampled = train_test_split(
        X_filtered, y_filtered,
        test_size=n_samples / len(X_filtered),
        random_state=random_state,
        stratify=y_filtered
    )
    return X_sampled, y_sampled


def extract_embeddings(
    df: pd.DataFrame,
    embedding_column: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Extract pre-computed embeddings from a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an embedding column
    embedding_column : str, optional
        Name of the embedding column. If None, auto-detects
        the first column with 'embedding' in its name.

    Returns
    -------
    embeddings : np.ndarray or None
        Extracted embeddings as (n_samples, embedding_dim) array,
        or None if no embedding column found
    """
    if embedding_column is None:
        embedding_cols = [c for c in df.columns if 'embedding' in c.lower()]
        if not embedding_cols:
            return None
        embedding_column = embedding_cols[0]
    elif embedding_column not in df.columns:
        return None

    raw = df[embedding_column]

    first_val = raw.iloc[0]
    if isinstance(first_val, str):
        import json
        embeddings = np.array([json.loads(v) for v in raw])
    elif isinstance(first_val, (list, np.ndarray)):
        embeddings = np.array(raw.tolist())
    else:
        return None

    return embeddings


def split_data_with_embeddings(
    X: pd.DataFrame,
    y: pd.Series,
    embeddings: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets, keeping embeddings aligned.

    Automatically filters out classes with fewer than 2 samples
    to enable stratified splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test, embeddings_train, embeddings_test
    """
    X_filtered, y_filtered, embeddings_filtered = filter_rare_classes(
        X, y, embeddings, min_samples=2
    )

    return train_test_split(
        X_filtered, y_filtered, embeddings_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=y_filtered
    )


def sample_stratified_with_embeddings(
    X: pd.DataFrame,
    y: pd.Series,
    embeddings: np.ndarray,
    n_samples: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Sample n_samples from X, y, embeddings with stratification.

    If n_samples >= len(X), returns original data unchanged.
    Automatically filters out classes with fewer than 2 samples
    to enable stratified sampling.
    """
    X_filtered, y_filtered, embeddings_filtered = filter_rare_classes(
        X, y, embeddings, min_samples=2
    )

    if n_samples >= len(X_filtered):
        return X_filtered, y_filtered, embeddings_filtered

    _, X_sampled, _, y_sampled, _, emb_sampled = train_test_split(
        X_filtered, y_filtered, embeddings_filtered,
        test_size=n_samples / len(X_filtered),
        random_state=random_state,
        stratify=y_filtered
    )
    return X_sampled, y_sampled, emb_sampled
