"""
Pytest configuration and shared fixtures for TabularLLM tests.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Sample feature dictionary."""
    return {
        'age': 35,
        'income': 50000,
        'education': 'bachelor'
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55, 30],
        'income': [30000, 50000, 70000, 90000, 40000],
        'education': ['high_school', 'bachelor', 'master', 'phd', 'bachelor']
    })


@pytest.fixture
def sample_labels():
    """Sample labels for classification."""
    return np.array(['low', 'medium', 'high', 'high', 'medium'])


@pytest.fixture
def binary_labels():
    """Binary labels for classification."""
    return np.array(['no', 'yes', 'yes', 'yes', 'no'])


@pytest.fixture
def sample_embeddings():
    """Sample embeddings array."""
    return np.random.randn(5, 384)


# ============================================================================
# Task Metadata Fixtures
# ============================================================================

@pytest.fixture
def task_metadata():
    """TaskMetadata instance for testing."""
    from tabullam.base.prompt_builder import TaskMetadata
    return TaskMetadata(
        feature_names=['age', 'income', 'education'],
        class_labels=['low', 'medium', 'high'],
        target_name='spending_category',
        task_description='Predict customer spending category'
    )


@pytest.fixture
def binary_task_metadata():
    """TaskMetadata for binary classification."""
    from tabullam.base.prompt_builder import TaskMetadata
    return TaskMetadata(
        feature_names=['age', 'income'],
        class_labels=['no', 'yes'],
        target_name='purchase',
        task_description='Predict if customer will purchase'
    )


# ============================================================================
# Mock LLM Response Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLMResponse object."""
    from tabullam.base.llm_backend import LLMResponse, LLMResponseDuration
    return LLMResponse(
        content='medium',
        usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110},
        duration=LLMResponseDuration(
            prompt_eval_duration=50.0,
            eval_duration=30.0,
            total_duration=80.0
        ),
        reasoning=None,
        raw_response=None
    )


@pytest.fixture
def mock_langchain_response():
    """Mock Langchain response object."""
    response = Mock()
    response.content = 'predicted_class'
    response.response_metadata = {
        'prompt_eval_duration': 50000000,  # nanoseconds
        'eval_duration': 30000000,
        'total_duration': 80000000
    }
    response.usage_metadata = {
        'input_tokens': 100,
        'output_tokens': 10,
        'total_tokens': 110
    }
    return response


@pytest.fixture
def mock_langchain_response_with_reasoning():
    """Mock Langchain response with reasoning blocks."""
    response = Mock()
    response.content = [
        {'type': 'reasoning', 'summary': [{'text': 'Step 1'}, {'text': 'Step 2'}]},
        {'type': 'text', 'text': 'final_answer'}
    ]
    response.response_metadata = {
        'prompt_eval_duration': 50000000,
        'eval_duration': 30000000,
        'total_duration': 80000000
    }
    response.usage_metadata = {
        'input_tokens': 100,
        'output_tokens': 10,
        'total_tokens': 110
    }
    return response


# ============================================================================
# Mock OpenAI Response Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = 'predicted_class'
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 10
    response.usage.total_tokens = 110
    return response


# ============================================================================
# Mock Embedding Fixtures
# ============================================================================

@pytest.fixture
def mock_ollama_embedding_response():
    """Mock Ollama embedding API response."""
    return {'embedding': list(np.random.randn(768))}


@pytest.fixture
def mock_openai_embedding_response():
    """Mock OpenAI embedding API response."""
    response = Mock()
    response.data = [Mock()]
    response.data[0].embedding = list(np.random.randn(1536))
    return response


# ============================================================================
# Few-shot Examples Fixtures
# ============================================================================

@pytest.fixture
def few_shot_examples():
    """Sample few-shot examples."""
    return [
        {'features': {'age': 25, 'income': 30000}, 'label': 'low'},
        {'features': {'age': 45, 'income': 80000}, 'label': 'high'},
        {'features': {'age': 35, 'income': 50000}, 'label': 'medium'},
    ]
