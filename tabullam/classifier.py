"""
TabularLLMClassifier - sklearn-compatible LLM-based classifier for tabular data.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from tqdm import tqdm
import logging

from .base.prompt_builder import TaskMetadata
from .prompt_builders import create_prompt_builder
from .response_parsers import create_response_parser
from .llm_backends import create_llm_backend
from .base.embedder import BaseEmbedder
from .embeddings.ollama import OllamaEmbedder
from .base.vector_store import BaseVectorStore
from .vector_stores.sklearn_store import SklearnVectorStore
from .utils.serialization import key_value_serialize
from .utils.validation import validate_mode
from .exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class TabularLLMClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible LLM-based classifier for tabular data.

    Note: ClassifierMixin must come before BaseEstimator to properly set
    _estimator_type = 'classifier' for sklearn's is_classifier() function.

    Parameters
    ----------
    llm : str, default='openai:gpt-4o-mini'
        LLM model identifier. Formats:
        - 'openai:<model_name>' for OpenAI models
        - 'ollama:<model_name>' for Ollama local models
        - 'google:<model_name>' for Google Gemini models

    mode : str, default='zero_shot'
        Classification mode:
        - 'zero_shot': No examples provided
        - 'random_few_shot': Random examples from training set
        - 'semantic_few_shot': Semantically similar examples (requires embedder)

    k_shots : int, default=5
        Number of examples for few-shot modes

    embedder : BaseEmbedder or None, default=None
        Embedder for semantic_few_shot mode. If None and mode='semantic_few_shot',
        will use default OllamaEmbedder('nomic-embed-text').
        Not needed when precomputed_embeddings is provided.

    precomputed_embeddings : ndarray of shape (n_samples, embedding_dim), optional
        Pre-computed embeddings for training data. When provided,
        the embedder is not used during fit() and the vector store
        is built directly from these embeddings.

    vector_store : str or BaseVectorStore, default='sklearn'
        Vector store backend: 'sklearn' or a BaseVectorStore instance

    task_description : str or None, default=None
        Optional description of the classification task for the prompt

    instruction : str or None, default=None
        Custom instruction for the LLM. If None, uses default instruction.

    positive_class : str or None, default=None
        Positive class for binary classification (used for ROC AUC calculation)

    random_state : int or None, default=None
        Random seed for reproducibility

    verbose : bool, default=False
        Whether to print progress information

    show_progress : bool, default=True
        Whether to show progress bar during prediction

    llm_kwargs : dict or None, default=None
        Additional keyword arguments passed to the LLM backend

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels

    n_features_in_ : int
        Number of features seen during fit

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit

    Examples
    --------
    >>> from tabullam import TabularLLMClassifier
    >>> clf = TabularLLMClassifier(llm='openai:gpt-4o-mini', mode='semantic_few_shot')
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> probabilities = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        llm: str = 'openai:gpt-4o-mini',
        mode: str = 'zero_shot',
        k_shots: int = 5,
        embedder: Optional[BaseEmbedder] = None,
        precomputed_embeddings: Optional[np.ndarray] = None,
        vector_store: Union[str, BaseVectorStore] = 'sklearn',
        task_description: Optional[str] = None,
        instruction: Optional[str] = None,
        positive_class: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        show_progress: bool = True,
        llm_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.llm = llm
        self.mode = mode
        self.k_shots = k_shots
        self.embedder = embedder
        self.precomputed_embeddings = precomputed_embeddings
        self.vector_store = vector_store
        self.task_description = task_description
        self.instruction = instruction
        self.positive_class = positive_class
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = show_progress
        self.llm_kwargs = llm_kwargs or {}

    def fit(self, X, y):
        """
        Fit the classifier.

        For zero_shot mode: only stores class labels and feature names.
        For random_few_shot: samples random examples from training data.
        For semantic_few_shot: computes embeddings and builds vector store.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted classifier
        """
        validate_mode(self.mode)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns.tolist())

        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if self.positive_class is None and len(self.classes_) == 2:
            self.positive_class = str(self.classes_[1])

        self._X_train = X.copy()
        self._y_train = y.copy()

        self._llm_backend = create_llm_backend(self.llm, **self.llm_kwargs)

        self._task_metadata = TaskMetadata(
            feature_names=self.feature_names_in_.tolist(),
            class_labels=[str(c) for c in self.classes_],
            target_name='target',
            task_description=self.task_description
        )

        self._prompt_builders = {}
        self._response_parsers = {}

        if self.mode == 'random_few_shot':
            self._setup_random_few_shot()
        elif self.mode == 'semantic_few_shot':
            self._setup_semantic_few_shot()

        return self

    def _get_prompt_builder(self, prompt_type: str):
        """Get or create prompt builder for given type."""
        if prompt_type not in self._prompt_builders:
            self._prompt_builders[prompt_type] = create_prompt_builder(
                prompt_type,
                self._task_metadata,
                self.instruction
            )
        return self._prompt_builders[prompt_type]

    def _get_response_parser(self, prompt_type: str):
        """Get or create response parser for given type."""
        if prompt_type not in self._response_parsers:
            self._response_parsers[prompt_type] = create_response_parser(
                prompt_type,
                [str(c) for c in self.classes_],
                self.positive_class
            )
        return self._response_parsers[prompt_type]

    def _setup_random_few_shot(self):
        """Setup for random few-shot mode: sample random examples."""
        rng = np.random.RandomState(self.random_state)
        n_samples = len(self._X_train)
        k = min(self.k_shots, n_samples)

        self._static_indices = rng.choice(n_samples, size=k, replace=False)

        if self.verbose:
            logger.info(f"Random few-shot: selected {k} random examples")

    def _setup_semantic_few_shot(self):
        """Setup for semantic few-shot mode: compute embeddings and build vector store."""
        if isinstance(self.vector_store, str):
            if self.vector_store == 'sklearn':
                self._vector_store = SklearnVectorStore()
            else:
                raise ConfigurationError(
                    f"Unknown vector_store: '{self.vector_store}'. Use 'sklearn' or provide a BaseVectorStore instance."
                )
        else:
            self._vector_store = self.vector_store

        if self.precomputed_embeddings is not None:
            embeddings = self.precomputed_embeddings
            if len(embeddings) != len(self._X_train):
                raise ConfigurationError(
                    f"precomputed_embeddings length ({len(embeddings)}) does not match "
                    f"training data length ({len(self._X_train)})"
                )
            if self.verbose:
                logger.info(f"Using pre-computed embeddings ({embeddings.shape[1]} dims)")
        else:
            if self.embedder is None:
                self.embedder = OllamaEmbedder(model='nomic-embed-text')
                if self.verbose:
                    logger.info("Using default OllamaEmbedder with nomic-embed-text")

            if self.verbose:
                logger.info("Computing embeddings for training data...")

            embeddings = self.embedder.embed_dataframe(
                self._X_train,
                columns=self.feature_names_in_.tolist()
            )

        metadata = []
        for i in range(len(self._X_train)):
            metadata.append({
                'index': i,
                'features': self._X_train.iloc[i].to_dict(),
                'label': str(self._y_train[i])
            })

        self._vector_store.add(embeddings, metadata)

        if self.verbose:
            logger.info(f"Added {len(embeddings)} embeddings to vector store")

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Uses 'standard' prompt type internally (plain text class prediction).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)

        results = self._predict_internal(X, prompt_type='standard')
        predictions = []

        for result in results:
            pred = result.get('prediction')
            if pred is None or pred not in [str(c) for c in self.classes_]:
                pred = str(self.classes_[0])
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Uses 'binary_confidence' or 'multiclass_confidence' prompt type internally.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)

        prompt_type = 'binary_confidence' if len(self.classes_) == 2 else 'multiclass_confidence'
        results = self._predict_internal(X, prompt_type=prompt_type)
        n_classes = len(self.classes_)
        probas = []

        for result in results:
            if 'probabilities' in result:
                proba = [
                    result['probabilities'].get(str(c), 1.0 / n_classes)
                    for c in self.classes_
                ]
            elif 'confidence' in result and n_classes == 2:
                confidence = result['confidence']
                prediction = result.get('prediction')
                proba = []
                for c in self.classes_:
                    if str(c) == prediction:
                        proba.append(confidence)
                    else:
                        proba.append(1.0 - confidence)
            else:
                proba = [1.0 / n_classes] * n_classes

            proba = np.array(proba)
            proba = proba / proba.sum()
            probas.append(proba)

        return np.array(probas)

    def predict_with_metadata(
        self,
        X,
        prompt_type: str = 'multiclass_confidence',
        precomputed_embeddings: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict with full metadata including retrieved examples and prompts.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Samples to predict
        prompt_type : str, default='multiclass_confidence'
            Prompt type to use: 'standard', 'binary_confidence', or 'multiclass_confidence'
        precomputed_embeddings : ndarray of shape (n_samples, embedding_dim), optional
            Pre-computed embeddings for test data. When provided,
            used as query vectors for semantic_few_shot instead of
            computing embeddings on-the-fly.

        Returns
        -------
        results : list of dict
            Each dict contains:
            - 'prediction': predicted class
            - 'probabilities': class probabilities (if applicable)
            - 'retrieved_examples': examples used for few-shot (if applicable)
            - 'prompt': full prompt sent to LLM
            - 'raw_response': LLM's raw response
            - 'usage': token usage statistics
            - 'duration': inference time
        """
        check_is_fitted(self)
        return self._predict_internal(
            X, prompt_type=prompt_type, include_metadata=True,
            precomputed_embeddings=precomputed_embeddings
        )

    def _predict_internal(
        self,
        X,
        prompt_type: str,
        include_metadata: bool = False,
        precomputed_embeddings: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Internal prediction method."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if precomputed_embeddings is not None and len(precomputed_embeddings) != len(X):
            raise ConfigurationError(
                f"precomputed_embeddings length ({len(precomputed_embeddings)}) "
                f"does not match X length ({len(X)})"
            )

        prompt_builder = self._get_prompt_builder(prompt_type)
        response_parser = self._get_response_parser(prompt_type)

        results = []
        iterator = range(len(X))

        if self.show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for i in iterator:
            row = X.iloc[i]
            query_features = row.to_dict()

            query_embedding = precomputed_embeddings[i] if precomputed_embeddings is not None else None

            examples = self._get_examples(query_features, query_embedding=query_embedding)

            prompt = prompt_builder.build(query_features, examples)

            llm_response = self._llm_backend.generate(prompt)

            parsed = response_parser.parse(llm_response.content)

            result = {
                **parsed,
                'usage': llm_response.usage,
                'duration': llm_response.duration_seconds,
            }

            if include_metadata:
                result['prompt'] = prompt
                result['raw_response'] = llm_response.content
                result['reasoning'] = llm_response.reasoning
                if examples:
                    result['retrieved_examples'] = examples

            results.append(result)

        return results

    def _get_examples(
        self,
        query_features: Dict[str, Any],
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get few-shot examples based on mode."""
        if self.mode == 'zero_shot':
            return None

        elif self.mode == 'random_few_shot':
            examples = []
            for idx in self._static_indices:
                examples.append({
                    'features': self._X_train.iloc[idx].to_dict(),
                    'label': str(self._y_train[idx])
                })
            return examples

        elif self.mode == 'semantic_few_shot':
            if query_embedding is None:
                query_text = key_value_serialize(query_features)
                query_embedding = self.embedder.embed([query_text])[0]

            indices, distances = self._vector_store.query(query_embedding, self.k_shots)
            metadata_list = self._vector_store.get_metadata(indices)

            examples = []
            for meta in metadata_list:
                examples.append({
                    'features': meta['features'],
                    'label': meta['label']
                })
            return examples

        return None

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'llm': self.llm,
            'mode': self.mode,
            'k_shots': self.k_shots,
            'embedder': self.embedder,
            'precomputed_embeddings': self.precomputed_embeddings,
            'vector_store': self.vector_store,
            'task_description': self.task_description,
            'instruction': self.instruction,
            'positive_class': self.positive_class,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'show_progress': self.show_progress,
            'llm_kwargs': self.llm_kwargs,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
