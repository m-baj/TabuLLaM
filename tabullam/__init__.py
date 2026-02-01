"""
TabularLLM - A scikit-learn compatible library for classifying tabular data with LLMs.

This library provides a simple interface for using Large Language Models to classify
tabular data, supporting zero-shot, static few-shot, and dynamic (RAG-based) few-shot
learning approaches.

Examples
--------
Basic zero-shot classification:

>>> from tabullam import TabularLLMClassifier
>>> clf = TabularLLMClassifier(llm='openai:gpt-4o-mini', mode='zero_shot')
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)

Semantic few-shot with custom embedder:

>>> from tabullam import TabularLLMClassifier
>>> from tabullam.embeddings import OllamaEmbedder
>>> embedder = OllamaEmbedder(model='nomic-embed-text')
>>> clf = TabularLLMClassifier(
...     llm='ollama:llama3.1:8b',
...     mode='semantic_few_shot',
...     k_shots=5,
...     embedder=embedder
... )
>>> clf.fit(X_train, y_train)
>>> probas = clf.predict_proba(X_test)

With scikit-learn pipeline:

>>> from sklearn.pipeline import Pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from tabullam import TabularLLMClassifier
>>> pipe = Pipeline([
...     ('scaler', StandardScaler()),
...     ('classifier', TabularLLMClassifier())
... ])
>>> pipe.fit(X_train, y_train)
"""

from .classifier import TabularLLMClassifier

# Base classes
from .base import (
    TaskMetadata,
    LLMResponse,
    LLMResponseDuration,
    BaseLLMBackend,
    BasePromptBuilder,
    BaseResponseParser,
    BaseVectorStore,
)

# Prompt builders
from .prompt_builders import (
    StandardPromptBuilder,
    ConfidencePromptBuilder,
    ProbabilitiesPromptBuilder,
    create_prompt_builder,
)

# Response parsers
from .response_parsers import (
    StandardParser,
    ConfidenceParser,
    ProbabilitiesParser,
    create_response_parser,
)

# LLM backend implementations
from .llm_backends import (
    LangchainBackend,
    OllamaBackend,
    OpenAIBackend,
    GoogleBackend,
    create_llm_backend,
    parse_langchain_response,
)

# Embeddings
from .embeddings import (
    BaseEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)

# Vector stores
from .vector_stores import (
    SklearnVectorStore,
)

# Exceptions
from .exceptions import (
    TabularLLMError,
    NotFittedError,
    EmbeddingError,
    LLMBackendError,
    ParseError,
    ConfigurationError,
)

__version__ = '0.1.0'

__all__ = [
    # Main classifier
    'TabularLLMClassifier',

    # Base classes
    'TaskMetadata',
    'LLMResponse',
    'LLMResponseDuration',
    'BaseLLMBackend',
    'BasePromptBuilder',
    'BaseResponseParser',
    'BaseVectorStore',

    # Prompt builders
    'StandardPromptBuilder',
    'ConfidencePromptBuilder',
    'ProbabilitiesPromptBuilder',
    'create_prompt_builder',

    # Response parsers
    'StandardParser',
    'ConfidenceParser',
    'ProbabilitiesParser',
    'create_response_parser',

    # LLM Backends
    'LangchainBackend',
    'OllamaBackend',
    'OpenAIBackend',
    'GoogleBackend',
    'create_llm_backend',
    'parse_langchain_response',

    # Embeddings
    'BaseEmbedder',
    'OllamaEmbedder',
    'OpenAIEmbedder',
    'SentenceTransformerEmbedder',

    # Vector stores
    'SklearnVectorStore',

    # Exceptions
    'TabularLLMError',
    'NotFittedError',
    'EmbeddingError',
    'LLMBackendError',
    'ParseError',
    'ConfigurationError',
]
