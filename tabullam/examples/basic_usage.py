#!/usr/bin/env python3
"""
Basic usage example for TabularLLMClassifier.

This script demonstrates how to use the library for tabular data classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import from the library
import sys
sys.path.insert(0, '..')
from tabullam import TabularLLMClassifier

from dotenv import load_dotenv
load_dotenv() 


def example_zero_shot():
    """Zero-shot classification example."""
    print("=" * 60)
    print("Example 1: Zero-shot Classification")
    print("=" * 60)

    data = {
        'age': [25, 45, 35, 50, 23, 40, 60, 28],
        'income': [30000, 80000, 55000, 90000, 25000, 70000, 100000, 35000],
        'has_mortgage': ['no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no'],
    }
    labels = ['no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no']

    X = pd.DataFrame(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = TabularLLMClassifier(
        llm='openai:gpt-5-mini',
        mode='zero_shot',
        task_description='Predict if a customer will purchase a premium product',
        verbose=True
    )

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    probabilities = clf.predict_proba(X_test)

    print("\nResults:")
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


def example_semantic_few_shot():
    """Semantic few-shot classification with semantic similarity."""
    print("\n" + "=" * 60)
    print("Example 2: Semantic Few-shot Classification")
    print("=" * 60)

    from tabullam.embeddings import OllamaEmbedder

    # Create sample data
    data = {
        'age': [25, 45, 35, 50, 23, 40, 60, 28, 32, 55],
        'income': [30000, 80000, 55000, 90000, 25000, 70000, 100000, 35000, 45000, 85000],
        'education': ['bachelor', 'master', 'bachelor', 'phd', 'high_school', 'master', 'phd', 'bachelor', 'master', 'master'],
    }
    labels = ['low', 'high', 'medium', 'high', 'low', 'high', 'high', 'low', 'medium', 'high']

    X = pd.DataFrame(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create embedder and classifier
    embedder = OllamaEmbedder(model='nomic-embed-text')

    clf = TabularLLMClassifier(
        llm='ollama:llama3.1:8b',
        mode='semantic_few_shot',
        k_shots=3,
        embedder=embedder,
        task_description='Predict customer spending category based on demographics',
        verbose=True
    )

    clf.fit(X_train, y_train)

    # Get class predictions (uses standard prompt internally)
    predictions = clf.predict(X_test)

    # Get probability estimates (uses probabilities prompt internally)
    probas = clf.predict_proba(X_test)

    # Get full metadata for debugging/analysis
    results = clf.predict_with_metadata(X_test)

    print("\nSample result:")
    for key, value in results[0].items():
        if key != 'prompt':  # Skip long prompt
            print(f"  {key}: {value}")


def example_sklearn_pipeline():
    """Using TabularLLMClassifier with sklearn pipeline."""
    print("\n" + "=" * 60)
    print("Example 3: sklearn Pipeline Integration")
    print("=" * 60)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Create sample data with missing values
    data = {
        'feature1': [1.0, 2.0, None, 4.0, 5.0],
        'feature2': [10, 20, 30, None, 50],
    }
    labels = ['A', 'B', 'A', 'B', 'A']

    X = pd.DataFrame(data)
    y = np.array(labels)

    # Create classifier - no prompt_type needed!
    # predict() automatically uses standard prompt
    # predict_proba() automatically uses probabilities prompt
    clf = TabularLLMClassifier(
        llm='openai:gpt-4o-mini',
        mode='zero_shot',
        show_progress=False
    )

    print("Pipeline created successfully!")
    print(f"Classifier params: {clf.get_params()}")


def example_cross_validation():
    """Cross-validation example."""
    print("\n" + "=" * 60)
    print("Example 4: Cross-Validation")
    print("=" * 60)

    from sklearn.model_selection import cross_val_score

    # Create sample data
    data = {
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
    }
    labels = ['A' if x > 0 else 'B' for x in data['feature1']]

    X = pd.DataFrame(data)
    y = np.array(labels)

    clf = TabularLLMClassifier(
        llm='openai:gpt-4o-mini',
        mode='zero_shot',
        show_progress=False
    )

    # Note: This will make many API calls
    # scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    # print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("Cross-validation setup complete!")
    print("(Uncomment the actual cross_val_score call to run)")


if __name__ == "__main__":
    # Run examples (comment out if you don't have API keys set up)
    print("TabularLLMClassifier Usage Examples")
    print("=" * 60)
    print("\nNote: These examples require API keys to be configured.")
    print("Set OPENAI_API_KEY for OpenAI examples.")
    print("Run Ollama locally for Ollama examples.\n")

    # Uncomment to run:
    example_zero_shot()
    # example_semantic_few_shot()
    # example_sklearn_pipeline()
    # example_cross_validation()
