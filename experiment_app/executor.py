"""
Experiment executor using TabularLLMClassifier.

Results are saved in the following directory structure:
results/{dataset}/{model}/{mode}_{k_shots}/{timestamp}_{proba|standard}/
    config.json              - Experiment configuration
    run_1_seed_42_results.json
    run_2_seed_7_results.json
    ...
    metrics_summary.json     - Aggregated metrics across all seeds
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from tabullam.classifier import TabularLLMClassifier
from tabullam.embeddings.sentence_transformers import SentenceTransformerEmbedder
from tabullam.embeddings.ollama import OllamaEmbedder
from tabullam.utils.serialization import key_value_serialize

from .config import ExperimentConfig, ExperimentSuite, get_dataset_info, is_binary_dataset, DEFAULT_EMBEDDING_MODEL
from .datasets import (
    load_dataset, prepare_features, split_data, sample_stratified,
    extract_embeddings, split_data_with_embeddings, sample_stratified_with_embeddings
)

console = Console()


def serialize_example(example: Dict[str, Any]) -> str:
    """Serialize a few-shot example to a readable string."""
    return key_value_serialize(example['features'], {'label': example['label']})


def save_to_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def setup_experiment_directory(
    base_output_dir: Path,
    dataset_name: str,
    model: str,
    mode: str,
    k_shots: int,
    prediction_mode: str = 'predict_proba'
) -> Path:
    """
    Setup experiment directory with proper structure.

    Structure: results/{dataset}/{model}/{mode}_{k_shots}/{timestamp}_{prediction_mode}/
    """
    dataset_clean = dataset_name.replace(" ", "_").replace("/", "-")
    model_clean = model.replace(':', '_')

    if mode in ['semantic_few_shot', 'random_few_shot']:
        mode_dir = f"{mode}_{k_shots}-shot"
    else:
        mode_dir = mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_suffix = "proba" if prediction_mode == "predict_proba" else "standard"
    leaf_dir = f"{timestamp}_{pred_suffix}"

    output_dir = base_output_dir / dataset_clean / model_clean / mode_dir / leaf_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_experiment_config(
    output_dir: Path,
    dataset_name: str,
    model: str,
    mode: str,
    k_shots: int,
    prediction_mode: str,
    task_description: str,
    positive_class: Optional[str],
    possible_classes: List[str],
    max_samples: Optional[int],
    test_size: float
) -> None:
    """Save experiment configuration to config.json."""
    config = {
        'dataset_name': dataset_name,
        'llm_name': model,
        'experiment_mode': mode,
        'k_shots': k_shots if mode != 'zero_shot' else None,
        'prediction_mode': prediction_mode,
        'task_description': task_description,
        'positive_class': positive_class,
        'possible_classes': possible_classes,
        'max_samples': max_samples,
        'test_size': test_size,
        'timestamp': datetime.now().isoformat()
    }
    save_to_json(config, output_dir / 'config.json')


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    is_binary: bool = True,
    positive_class: Optional[str] = None,
    classes: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            if is_binary and y_proba.ndim == 2:
                pos_proba = y_proba[:, 1]
                y_true_bin = (y_true == positive_class).astype(int) if positive_class else y_true
                metrics['roc_auc'] = float(roc_auc_score(y_true_bin, pos_proba))
                metrics['pr_auc'] = float(average_precision_score(y_true_bin, pos_proba))
            elif not is_binary and classes is not None:
                present_classes = sorted(set(y_true))
                if len(present_classes) < 2:
                    metrics['roc_auc'] = None
                    metrics['pr_auc'] = None
                else:
                    present_indices = [i for i, c in enumerate(classes) if c in present_classes]
                    y_proba_filtered = y_proba[:, present_indices]
                    # Use label_binarize with explicit classes to ensure correct shape
                    # even when there are only 2 present classes
                    y_true_onehot = label_binarize(y_true, classes=present_classes)
                    # label_binarize returns (n_samples, 1) for 2 classes, convert to (n_samples, 2)
                    if len(present_classes) == 2 and y_true_onehot.shape[1] == 1:
                        y_true_onehot = np.hstack([1 - y_true_onehot, y_true_onehot])
                    metrics['roc_auc'] = float(roc_auc_score(
                        y_true_onehot, y_proba_filtered, multi_class='ovr', average='macro'
                    ))
                    metrics['pr_auc'] = float(average_precision_score(
                        y_true_onehot, y_proba_filtered, average='macro'
                    ))
        except Exception as e:
            console.print(f"  [dim yellow]Warning: Could not compute ROC/PR AUC: {e}[/dim yellow]")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None

    return metrics


def aggregate_metrics(history_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple runs.

    Returns dict with mean, std, and all values for each metric.
    """
    if not history_metrics:
        return {}

    aggregated = {}
    keys = history_metrics[0].keys()

    for key in keys:
        values = [run[key] for run in history_metrics if run.get(key) is not None]
        if values:
            aggregated[f"mean_{key}"] = float(np.mean(values))
            aggregated[f"std_{key}"] = float(np.std(values))
            aggregated[f"all_{key}_values"] = values

    return aggregated


def run_single_experiment(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    k_shots: int = 5,
    max_samples: Optional[int] = 500,
    test_size: float = 0.2,
    prediction_mode: str = 'predict_proba',
    verbose: bool = False,
    preloaded_data: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Parameters
    ----------
    preloaded_data : tuple, optional
        Pre-loaded (X, y, embeddings, info) tuple to avoid reloading dataset
        for each seed. If None, dataset will be loaded from scratch.

    Returns
    -------
    result : dict
        Contains metrics, predictions, and metadata
    """
    start_time = time.time()

    if preloaded_data is not None:
        X, y, embeddings, info = preloaded_data
    else:
        df, target_column = load_dataset(dataset_name)
        info = get_dataset_info(dataset_name)
        embedding_column = info.get('embedding_column')
        embeddings = extract_embeddings(df, embedding_column)
        X, y = prepare_features(df, target_column)

    if embeddings is not None:
        X_train, X_test, y_train, y_test, emb_train, emb_test = split_data_with_embeddings(
            X, y, embeddings, test_size=test_size, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=seed)
        emb_train, emb_test = None, None

    if max_samples and len(X_test) > max_samples:
        if emb_test is not None:
            X_test, y_test, emb_test = sample_stratified_with_embeddings(
                X_test, y_test, emb_test, max_samples, random_state=seed
            )
        else:
            X_test, y_test = sample_stratified(X_test, y_test, max_samples, random_state=seed)

    is_binary = info.get('binary', True)
    positive_class = info.get('positive_class')
    task_description = info.get('description', '')
    classes = list(y.unique())

    embedder = None
    precomputed_train_embeddings = None
    if mode == 'semantic_few_shot':
        if emb_train is not None:
            precomputed_train_embeddings = emb_train
            console.print(f"  [dim]Using pre-computed embeddings ({emb_train.shape[1]} dims)[/dim]")
        else:
            embedding_spec = info.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
            if ':' in embedding_spec:
                backend, model_name = embedding_spec.split(':', 1)
            else:
                backend, model_name = 'sentence_transformers', embedding_spec

            if backend == 'sentence_transformers':
                embedder = SentenceTransformerEmbedder(model=model_name)
            elif backend == 'ollama':
                embedder = OllamaEmbedder(model=model_name)
            else:
                raise ValueError(f"Unknown embedding backend: '{backend}'. Use 'sentence_transformers' or 'ollama'.")

    clf = TabularLLMClassifier(
        llm=model,
        mode=mode,
        k_shots=k_shots,
        embedder=embedder,
        precomputed_embeddings=precomputed_train_embeddings,
        task_description=task_description,
        positive_class=positive_class,
        random_state=seed,
        verbose=verbose,
        show_progress=True
    )

    clf.fit(X_train, y_train)

    if prediction_mode == 'predict':
        prompt_type = 'standard'
    elif prediction_mode == 'predict_proba':
        prompt_type = 'binary_confidence' if is_binary else 'multiclass_confidence'
    else:
        prompt_type = 'standard'

    results_with_metadata = clf.predict_with_metadata(
        X_test, prompt_type=prompt_type,
        precomputed_embeddings=emb_test
    )

    n_classes = len(clf.classes_)
    y_pred = []
    y_proba = None if prediction_mode != 'predict_proba' else []

    for result in results_with_metadata:
        pred = result.get('prediction')
        if pred is None or pred not in [str(c) for c in clf.classes_]:
            pred = str(clf.classes_[0])
        y_pred.append(pred)

        if prediction_mode == 'predict_proba':
            if 'probabilities' in result:
                proba = [result['probabilities'].get(str(c), 1.0 / n_classes) for c in clf.classes_]
            elif 'confidence' in result and is_binary:
                confidence = result['confidence']
                proba = []
                for c in clf.classes_:
                    if str(c) == pred:
                        proba.append(confidence)
                    else:
                        proba.append(1.0 - confidence)
            else:
                proba = [1.0 / n_classes] * n_classes
            y_proba.append(proba)

    y_pred = np.array(y_pred)
    if y_proba is not None:
        y_proba = np.array(y_proba)

    metrics = compute_metrics(
        y_test.values, y_pred, y_proba,
        is_binary=is_binary,
        positive_class=positive_class,
        classes=classes
    )

    elapsed_time = time.time() - start_time

    predictions = []
    for i, (idx, row) in enumerate(X_test.iterrows()):
        result = results_with_metadata[i]
        pred_entry = {
            'id': int(idx) if hasattr(idx, '__int__') else str(idx),
            'ground_truth': str(y_test.iloc[i]),
            'predicted': str(y_pred[i]),
            'prompt': result.get('prompt'),
            'raw_response': result.get('raw_response'),
            'usage': result.get('usage'),
            'duration_s': result.get('duration'),
        }

        if y_proba is not None:
            pred_entry['probabilities'] = {str(c): float(y_proba[i, j]) for j, c in enumerate(clf.classes_)}
            if is_binary:
                pred_entry['positive_class_score'] = float(y_proba[i, 1])

        if 'retrieved_examples' in result:
            examples = result['retrieved_examples']
            pred_entry['retrieved_examples'] = examples
            pred_entry['retrieved_examples_serialized'] = [
                serialize_example(ex) for ex in examples
            ]

        if result.get('reasoning'):
            pred_entry['reasoning'] = result['reasoning']

        predictions.append(pred_entry)

    static_examples = None
    static_examples_serialized = None
    if mode == 'random_few_shot' and hasattr(clf, '_static_indices'):
        static_examples = []
        for idx in clf._static_indices:
            static_examples.append({
                'features': clf._X_train.iloc[idx].to_dict(),
                'label': str(clf._y_train[idx])
            })
        static_examples_serialized = [serialize_example(ex) for ex in static_examples]

    return {
        'seed': seed,
        'metrics': metrics,
        'predictions': predictions,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'classes': [str(c) for c in classes],
        'task_description': task_description,
        'positive_class': positive_class,
        'prediction_mode': prediction_mode,
        'static_few_shot_examples': static_examples,
        'static_few_shot_examples_serialized': static_examples_serialized,
    }


def execute_experiment_group(
    dataset_name: str,
    model: str,
    mode: str,
    seeds: List[int],
    k_shots: int,
    max_samples: Optional[int],
    test_size: float,
    prediction_mode: str,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Execute all seeds for a single (dataset, model, mode) combination.

    Saves individual run results and aggregated metrics.
    Dataset is loaded ONCE and reused across all seeds.
    """
    exp_dir = setup_experiment_directory(
        output_dir, dataset_name, model, mode, k_shots, prediction_mode
    )

    run_results = []
    run_metrics = []

    info = get_dataset_info(dataset_name)

    console.print(f"  [dim]Loading dataset: {dataset_name}...[/dim]")
    df, target_column = load_dataset(dataset_name)
    embedding_column = info.get('embedding_column')
    embeddings = extract_embeddings(df, embedding_column)
    X, y = prepare_features(df, target_column)
    preloaded_data = (X, y, embeddings, info)
    if embeddings is not None:
        console.print(f"  [dim]Dataset loaded: {len(X)} samples with pre-computed embeddings ({embeddings.shape[1]} dims)[/dim]")
    else:
        console.print(f"  [dim]Dataset loaded: {len(X)} samples (no pre-computed embeddings)[/dim]")

    pred_label = "proba" if prediction_mode == "predict_proba" else "standard"

    for run_id, seed in enumerate(seeds, 1):
        desc = f"{dataset_name} / {model.split(':')[-1]} / {mode} / {pred_label} / seed={seed}"
        console.print(f"\n  [bold]{desc}[/bold]")

        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                model=model,
                mode=mode,
                seed=seed,
                k_shots=k_shots,
                max_samples=max_samples,
                test_size=test_size,
                prediction_mode=prediction_mode,
                preloaded_data=preloaded_data
            )

            run_file = exp_dir / f"run_{run_id}_seed_{seed}_results.json"
            save_to_json(result, run_file)

            run_results.append(result)
            run_metrics.append(result['metrics'])

            if prediction_mode == 'predict_proba':
                roc = result['metrics'].get('roc_auc')
                pr = result['metrics'].get('pr_auc')
                roc_str = f"{roc:.3f}" if roc is not None else "N/A"
                pr_str = f"{pr:.3f}" if pr is not None else "N/A"
                console.print(f"  [green]✓ Completed[/green]: roc_auc={roc_str}, pr_auc={pr_str}")
            else:
                f1 = result['metrics']['f1']
                mcc = result['metrics']['mcc']
                console.print(f"  [green]✓ Completed[/green]: f1={f1:.3f}, mcc={mcc:.3f}")

        except Exception as e:
            console.print(f"  [red]✗ ERROR[/red]: {e}")
            run_results.append({
                'seed': seed,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    save_experiment_config(
        output_dir=exp_dir,
        dataset_name=dataset_name,
        model=model,
        mode=mode,
        k_shots=k_shots,
        prediction_mode=prediction_mode,
        task_description=info.get('description', ''),
        positive_class=info.get('positive_class'),
        possible_classes=run_results[0].get('classes', []) if run_results else [],
        max_samples=max_samples,
        test_size=test_size
    )

    if run_metrics:
        aggregated = aggregate_metrics(run_metrics)
        aggregated['dataset_name'] = dataset_name
        aggregated['n_runs'] = len(run_metrics)
        aggregated['seeds'] = seeds
        save_to_json(aggregated, exp_dir / 'metrics_summary.json')

        console.print(f"  [cyan]-> Saved to: {exp_dir}[/cyan]")

    return run_results


def execute_experiment(exp: ExperimentConfig, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Execute a single experiment configuration.

    Returns
    -------
    results : list
        List of result dictionaries for each run
    """
    results = []
    total_runs = exp.get_total_runs()

    pred_label = "predict_proba" if exp.prediction_mode == "predict_proba" else "predict"
    console.print(f"\n[bold]Running: {exp.name or 'Experiment'}[/bold]")
    console.print(f"Prediction mode: [cyan]{pred_label}[/cyan]")
    console.print(f"Total runs: {total_runs}\n")

    for dataset in exp.datasets:
        for model in exp.models:
            for mode in exp.modes:
                group_results = execute_experiment_group(
                    dataset_name=dataset,
                    model=model,
                    mode=mode,
                    seeds=exp.seeds,
                    k_shots=exp.k_shots,
                    max_samples=exp.max_samples,
                    test_size=exp.test_size,
                    prediction_mode=exp.prediction_mode,
                    output_dir=output_dir
                )

                for r in group_results:
                    r['dataset'] = dataset
                    r['model'] = model
                    r['mode'] = mode
                    results.append(r)

    return results


def execute_suite(suite: ExperimentSuite, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Execute all experiments in a suite.

    Parameters
    ----------
    suite : ExperimentSuite
        The experiment suite to execute
    output_dir : Path, optional
        Directory to save results

    Returns
    -------
    all_results : list
        All results from all experiments
    """
    if output_dir is None:
        output_dir = Path('experiment-results')
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Starting Experiment Execution[/bold cyan]")
    console.print(f"Output directory: {output_dir}\n")

    all_results = []
    start_time = time.time()

    for i, exp in enumerate(suite.experiments, 1):
        console.print(f"\n[bold]Experiment {i}/{len(suite.experiments)}[/bold]")
        exp_start = time.time()
        results = execute_experiment(exp, output_dir)
        exp_time = time.time() - exp_start

        exp_title = exp.name or f"Experiment {i}"
        show_results_summary(results, exp_time, title=exp_title)

        all_results.extend(results)

    total_time = time.time() - start_time
    console.print(f"\n[bold cyan]All experiments completed in {total_time:.1f}s[/bold cyan]\n")

    return all_results


def show_results_summary(
    results: List[Dict[str, Any]],
    total_time: float,
    title: Optional[str] = None
):
    """Display a summary of experiment results."""
    header = f"Results: {title}" if title else "Results Summary"
    console.print(f"\n[bold cyan]{header}[/bold cyan]\n")

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    console.print(f"Successful runs: {len(successful)}")
    console.print(f"Failed runs: {len(failed)}")
    console.print(f"Time: {total_time:.1f}s\n")

    if not successful:
        return

    from collections import defaultdict

    grouped = defaultdict(list)
    for r in successful:
        key = (r.get('dataset'), r.get('model'), r.get('mode'), r.get('prediction_mode', 'predict'))
        if r.get('metrics'):
            grouped[key].append(r['metrics'])

    has_standard = any(k[3] == 'predict' for k in grouped)
    has_proba = any(k[3] == 'predict_proba' for k in grouped)

    table_title = f"{title} — Aggregated Metrics (mean +/- std)" if title else "Aggregated Metrics (mean +/- std)"
    table = Table(title=table_title, box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Mode", style="yellow")

    if has_standard and has_proba:
        table.add_column("Type", style="dim")

    if has_standard:
        table.add_column("F1", style="green")
        table.add_column("MCC", style="blue")
    if has_proba:
        table.add_column("ROC AUC", style="white")
        table.add_column("PR AUC", style="white")

    for (dataset, model, mode, pred_mode), metrics_list in sorted(grouped.items()):
        if not metrics_list:
            continue

        row = [
            dataset or "?",
            (model or "?").split(':')[-1],
            mode or "?",
        ]

        if has_standard and has_proba:
            pred_label = "proba" if pred_mode == "predict_proba" else "standard"
            row.append(pred_label)

        if has_standard:
            if pred_mode == 'predict':
                f1_mean = np.mean([m['f1'] for m in metrics_list])
                f1_std = np.std([m['f1'] for m in metrics_list])
                mcc_mean = np.mean([m['mcc'] for m in metrics_list])
                mcc_std = np.std([m['mcc'] for m in metrics_list])
                row.append(f"{f1_mean:.3f} +/- {f1_std:.3f}")
                row.append(f"{mcc_mean:.3f} +/- {mcc_std:.3f}")
            else:
                row.extend(["", ""])

        if has_proba:
            if pred_mode == 'predict_proba':
                roc_values = [m['roc_auc'] for m in metrics_list if m.get('roc_auc') is not None]
                pr_values = [m['pr_auc'] for m in metrics_list if m.get('pr_auc') is not None]
                if roc_values:
                    row.append(f"{np.mean(roc_values):.3f} +/- {np.std(roc_values):.3f}")
                else:
                    row.append("N/A")
                if pr_values:
                    row.append(f"{np.mean(pr_values):.3f} +/- {np.std(pr_values):.3f}")
                else:
                    row.append("N/A")
            else:
                row.extend(["", ""])

        table.add_row(*row)

    console.print(table)
