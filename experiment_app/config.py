"""
Configuration for experiment app.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import json


AVAILABLE_DATASETS = [
    'breast_cancer',
    'caesarian',
    'car_evaluation',
    'wine_quality'
]

AVAILABLE_MODELS = [
    'openai:gpt-5-mini',
    'openai:gpt-4o-mini',
    'openai:gpt-4o',
    'ollama:llama3.1:8b',
    'ollama:gpt-oss:20b',
]

AVAILABLE_MODES = [
    'zero_shot',
    'random_few_shot',
    'semantic_few_shot'
]

AVAILABLE_PREDICTION_MODES = [
    'predict',
    'predict_proba'
]

DEFAULT_SEEDS = [42, 7, 123]

DEFAULT_EMBEDDING_MODEL = 'sentence_transformers:all-MiniLM-L6-v2'

DATASETS_INFO = {
    'breast_cancer': {
        'name': 'breast_cancer',
        'target_column': 'Class',
        'description': "This dataset contains clinical records of breast cancer patients, including tumor size, number of involved lymph nodes, degree of malignancy, and menopause status. Your task is to classify whether a recurrence of cancer is likely to occur based on these medical attributes. The target variable is 'Class', which can be: 'recurrence-events' or 'no-recurrence-events'.",
        'positive_class': 'recurrence-events',
        'binary': True,
        'file': 'data/breast_cancer.parquet',
        'embedding_model': 'ollama:nomic-embed-text:latest',
        'embedding_column': 'embedding_nomic_embed_text'
    },
    'caesarian': {
        'name': 'caesarian',
        'target_column': 'Caesarian',
        'description': "This dataset contains information about caesarian section results of pregnant women with the most important characteristics of delivery problems in the medical field. Your task is to predict whether a caesarian section is required based on these medical factors.",
        'positive_class': 'yes',
        'binary': True,
        'file': 'data/caesarian.parquet',
        'embedding_model': 'ollama:nomic-embed-text:latest',
        'embedding_column': 'embedding_nomic_embed_text'
    },
    'car_evaluation': {
        'name': 'car_evaluation',
        'target_column': 'class',
        'description': "This dataset contains evaluations of cars based on various attributes such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety features. Your task is to classify the overall acceptability of a car into one of four categories: unacc (unacceptable), acc (acceptable), good, and vgood (very good).",
        'binary': False,
        'file': 'data/car_evaluation.parquet',
        'embedding_model': 'ollama:nomic-embed-text:latest',
        'embedding_column': 'embedding_nomic_embed_text'
    },
    'wine_quality': {
        'name': 'wine_quality',
        'target_column': 'quality',
        'description': "This dataset contains physicochemical tests of wines and their quality ratings. Your task is to predict the quality rating based on the given features.",
        'binary': False,
        'file': 'data/wine_quality.parquet',
        'embedding_model': 'ollama:nomic-embed-text:latest',
        'embedding_column': 'embedding_nomic_embed_text'
    }
}


def is_binary_dataset(dataset_name: str) -> bool:
    """Check if dataset is binary classification."""
    return DATASETS_INFO.get(dataset_name, {}).get('binary', False)


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get metadata for a dataset."""
    return DATASETS_INFO.get(dataset_name, {})


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    datasets: List[str]
    models: List[str]
    modes: List[str]
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    k_shots: int = 5
    max_samples: Optional[int] = 500
    test_size: float = 0.2
    prediction_mode: str = 'predict_proba'  # 'predict' or 'predict_proba'
    name: Optional[str] = None

    def __post_init__(self):
        """Validate and convert single values to lists."""
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        if isinstance(self.models, str):
            self.models = [self.models]
        if isinstance(self.modes, str):
            self.modes = [self.modes]
        if isinstance(self.seeds, int):
            self.seeds = [self.seeds]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'datasets': self.datasets,
            'models': self.models,
            'modes': self.modes,
            'seeds': self.seeds,
            'k_shots': self.k_shots,
            'max_samples': self.max_samples,
            'test_size': self.test_size,
            'prediction_mode': self.prediction_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)

    def get_total_runs(self) -> int:
        """Calculate total number of experiment runs."""
        return (
            len(self.datasets) *
            len(self.models) *
            len(self.modes) *
            len(self.seeds)
        )


@dataclass
class ExperimentSuite:
    """Collection of experiment configurations."""

    experiments: List[ExperimentConfig]

    def add_experiment(self, exp: ExperimentConfig):
        """Add an experiment to the suite."""
        self.experiments.append(exp)

    def get_total_runs(self) -> int:
        """Calculate total runs across all experiments."""
        return sum(exp.get_total_runs() for exp in self.experiments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiments': [exp.to_dict() for exp in self.experiments]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentSuite':
        """Create from dictionary."""
        experiments = [ExperimentConfig.from_dict(exp) for exp in data.get('experiments', [])]
        return cls(experiments=experiments)

    def to_yaml(self, path: Path):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentSuite':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> 'ExperimentSuite':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
