"""
Dataset loading utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine
from src.RAG.config import config

from .config import DATASETS_INFO, get_dataset_info


def get_data_dir() -> Path:
    """Get the data directory path."""
    # Look for data directory relative to project root
    current = Path(__file__).parent.parent
    data_dir = current / 'data'
    if data_dir.exists():
        return data_dir
    raise FileNotFoundError(f"Data directory not found at {data_dir}")


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset by name.

    Tries to load from local file first (parquet/csv),
    falls back to PostgreSQL if file not found.

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

    df = None
    source = None

    # Try loading from local file first
    try:
        data_dir = get_data_dir()
        file_path = data_dir / f"{dataset_name}.parquet"

        if not file_path.exists():
            file_path = data_dir / f"{dataset_name}.csv"

        if file_path.exists():
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            source = f"file: {file_path}"
    except FileNotFoundError:
        pass

    # Fallback to PostgreSQL
    if df is None:
        try:
            df = load_dataset_from_postgres(dataset_name)
            source = f"postgres: {dataset_name}"
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. "
                f"Tried local files and PostgreSQL. PostgreSQL error: {e}"
            )

    print(f"[datasets] Loaded from {source}, {len(df)} rows")

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

    # Also exclude any embedding columns
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

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
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
    if n_samples >= len(X):
        return X, y

    # Use train_test_split to get stratified sample
    # We want n_samples, so test_size = n_samples / len(X)
    _, X_sampled, _, y_sampled = train_test_split(
        X, y,
        test_size=n_samples / len(X),
        random_state=random_state,
        stratify=y
    )
    return X_sampled, y_sampled


def load_dataset_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Load a dataset directly from PostgreSQL database.

    Parameters
    ----------
    table_name : str
        Name of the table to load

    Returns
    -------
    df : pd.DataFrame
        The loaded dataframe with table_name stored in df.attrs
    """
    engine = create_engine(config.POSTGRES_URL)
    df = pd.read_sql(f"SELECT * FROM {table_name};", engine)

    df.sort_values(by='id').reset_index(drop=True, inplace=True)
    df.drop(columns=['id', 'serialized_row'], inplace=True, errors='ignore')
    df.attrs['table_name'] = table_name
    return df
