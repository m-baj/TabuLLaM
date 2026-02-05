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

from .base import (
    TaskMetadata,
    LLMResponse,
    LLMResponseDuration,
    BaseLLMBackend,
    BasePromptBuilder,
    BaseResponseParser,
    BaseVectorStore,
    BaseEmbedder,
)

from .prompt_builders import (
    StandardPromptBuilder,
    BinaryConfidencePromptBuilder,
    MulticlassConfidencePromptBuilder,
    create_prompt_builder,
)

from .response_parsers import (
    StandardParser,
    BinaryConfidenceParser,
    MulticlassConfidenceParser,
    create_response_parser,
)

from .llm_backends import (
    LangchainBackend,
    OllamaBackend,
    GoogleBackend,
    create_llm_backend,
    parse_langchain_response,
)

from .embeddings import (
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)

from .vector_stores import (
    SklearnVectorStore,
)

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
    'TabularLLMClassifier',

    'TaskMetadata',
    'LLMResponse',
    'LLMResponseDuration',
    'BaseLLMBackend',
    'BasePromptBuilder',
    'BaseResponseParser',
    'BaseVectorStore',

    'StandardPromptBuilder',
    'BinaryConfidencePromptBuilder',
    'MulticlassConfidencePromptBuilder',
    'create_prompt_builder',

    'StandardParser',
    'BinaryConfidenceParser',
    'MulticlassConfidenceParser',
    'create_response_parser',

    'LangchainBackend',
    'OllamaBackend',
    'GoogleBackend',
    'create_llm_backend',
    'parse_langchain_response',

    'BaseEmbedder',
    'OllamaEmbedder',
    'OpenAIEmbedder',
    'SentenceTransformerEmbedder',

    'SklearnVectorStore',

    'TabularLLMError',
    'NotFittedError',
    'EmbeddingError',
    'LLMBackendError',
    'ParseError',
    'ConfigurationError',
]
