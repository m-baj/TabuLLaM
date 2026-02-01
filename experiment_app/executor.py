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
from sklearn.preprocessing import LabelBinarizer

from tabullam.classifier import TabularLLMClassifier
from tabullam.embeddings.sentence_transformers import SentenceTransformerEmbedder

from .config import ExperimentConfig, ExperimentSuite, get_dataset_info, is_binary_dataset
from .datasets import load_dataset, prepare_features, split_data, sample_stratified

console = Console()


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

    # Mode directory name
    if mode in ['semantic_few_shot', 'random_few_shot']:
        mode_dir = f"{mode}_{k_shots}-shot"
    else:
        mode_dir = mode

    # Timestamp and prediction mode suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_suffix = "proba" if prediction_mode == "predict_proba" else "standard"
    leaf_dir = f"{timestamp}_{pred_suffix}"

    # Create full path
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

    # ROC AUC and PR AUC
    if y_proba is not None:
        try:
            if is_binary and y_proba.ndim == 2:
                # Binary classification - use probability of positive class
                pos_proba = y_proba[:, 1]
                y_true_bin = (y_true == positive_class).astype(int) if positive_class else y_true
                metrics['roc_auc'] = float(roc_auc_score(y_true_bin, pos_proba))
                metrics['pr_auc'] = float(average_precision_score(y_true_bin, pos_proba))
            elif not is_binary and classes is not None:
                # Multiclass
                lb = LabelBinarizer()
                lb.fit(classes)
                y_true_onehot = lb.transform(y_true)
                metrics['roc_auc'] = float(roc_auc_score(
                    y_true_onehot, y_proba, multi_class='ovr', average='macro'
                ))
                metrics['pr_auc'] = float(average_precision_score(
                    y_true_onehot, y_proba, average='macro'
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
        Pre-loaded (X, y, info) tuple to avoid reloading dataset for each seed.
        If None, dataset will be loaded from scratch.

    Returns
    -------
    result : dict
        Contains metrics, predictions, and metadata
    """
    start_time = time.time()

    # Use preloaded data or load dataset
    if preloaded_data is not None:
        X, y, info = preloaded_data
    else:
        df, target_column = load_dataset(dataset_name)
        X, y = prepare_features(df, target_column)
        info = get_dataset_info(dataset_name)

    # Split with seed-specific random state
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=seed)

    # Apply max_samples to TEST set only (to limit API calls during prediction)
    if max_samples and len(X_test) > max_samples:
        X_test, y_test = sample_stratified(X_test, y_test, max_samples, random_state=seed)

    # Get dataset info
    is_binary = info.get('binary', True)
    positive_class = info.get('positive_class')
    task_description = info.get('description', '')
    classes = list(y.unique())

    # Create embedder for semantic few-shot
    embedder = None
    if mode == 'semantic_few_shot':
        embedder = SentenceTransformerEmbedder(model='all-MiniLM-L6-v2')

    # Create classifier
    clf = TabularLLMClassifier(
        llm=model,
        mode=mode,
        k_shots=k_shots,
        embedder=embedder,
        task_description=task_description,
        positive_class=positive_class,
        random_state=seed,
        verbose=verbose,
        show_progress=True  # Show progress bar for each run
    )

    # Fit and predict
    clf.fit(X_train, y_train)

    # Choose prediction method based on prediction_mode
    if prediction_mode == 'predict_proba':
        # Use predict_proba and derive predictions from probabilities
        y_proba = clf.predict_proba(X_test)
        y_pred = clf.classes_[np.argmax(y_proba, axis=1)]
    else:
        # Standard predict (faster, no probabilities)
        y_pred = clf.predict(X_test)
        y_proba = None

    # Compute metrics
    metrics = compute_metrics(
        y_test.values, y_pred, y_proba,
        is_binary=is_binary,
        positive_class=positive_class,
        classes=classes
    )

    elapsed_time = time.time() - start_time

    # Build detailed results
    predictions = []
    for i, (idx, row) in enumerate(X_test.iterrows()):
        pred_entry = {
            'id': int(idx) if hasattr(idx, '__int__') else str(idx),
            'ground_truth': str(y_test.iloc[i]),
            'predicted': str(y_pred[i]),
        }
        if y_proba is not None:
            pred_entry['probabilities'] = {str(c): float(y_proba[i, j]) for j, c in enumerate(clf.classes_)}
            if is_binary:
                pred_entry['positive_class_score'] = float(y_proba[i, 1])
        predictions.append(pred_entry)

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
        'prediction_mode': prediction_mode
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
    # Setup directory for this experiment group
    exp_dir = setup_experiment_directory(
        output_dir, dataset_name, model, mode, k_shots, prediction_mode
    )

    run_results = []
    run_metrics = []

    # Get dataset info for config
    info = get_dataset_info(dataset_name)

    # Load FULL dataset ONCE for all seeds
    # max_samples will be applied to TEST set after split in run_single_experiment
    console.print(f"  [dim]Loading dataset: {dataset_name}...[/dim]")
    df, target_column = load_dataset(dataset_name)
    X, y = prepare_features(df, target_column)
    preloaded_data = (X, y, info)
    console.print(f"  [dim]Dataset loaded: {len(X)} samples (max_samples={max_samples} applied to test set)[/dim]")

    for run_id, seed in enumerate(seeds, 1):
        desc = f"{dataset_name} / {model.split(':')[-1]} / {mode} / seed={seed}"
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

            # Save individual run result
            run_file = exp_dir / f"run_{run_id}_seed_{seed}_results.json"
            save_to_json(result, run_file)

            run_results.append(result)
            run_metrics.append(result['metrics'])

            # Show brief result
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

    # Save experiment config
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

    # Aggregate metrics and save summary
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

    console.print(f"\n[bold]Running: {exp.name or 'Experiment'}[/bold]")
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
        results = execute_experiment(exp, output_dir)
        all_results.extend(results)

    total_time = time.time() - start_time

    # Show summary
    show_results_summary(all_results, total_time)

    return all_results


def show_results_summary(results: List[Dict[str, Any]], total_time: float):
    """Display a summary of experiment results."""
    console.print("\n[bold cyan]Results Summary[/bold cyan]\n")

    # Filter successful results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    console.print(f"Successful runs: {len(successful)}")
    console.print(f"Failed runs: {len(failed)}")
    console.print(f"Total time: {total_time:.1f}s\n")

    if not successful:
        return

    # Check if any results have ROC AUC (predict_proba mode)
    from collections import defaultdict
    grouped = defaultdict(list)

    for r in successful:
        key = (r.get('dataset'), r.get('model'), r.get('mode'))
        if r.get('metrics'):
            grouped[key].append(r['metrics'])

    # Check if ROC AUC is available
    has_roc_auc = any(
        m.get('roc_auc') is not None
        for metrics_list in grouped.values()
        for m in metrics_list
    )

    # Group by dataset
    table = Table(title="Metrics by Dataset and Model (mean +/- std)", box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Mode", style="yellow")
    table.add_column("F1", style="green")
    table.add_column("MCC", style="blue")
    if has_roc_auc:
        table.add_column("ROC AUC", style="white")

    for (dataset, model, mode), metrics_list in sorted(grouped.items()):
        if not metrics_list:
            continue

        f1_mean = np.mean([m['f1'] for m in metrics_list])
        f1_std = np.std([m['f1'] for m in metrics_list])
        mcc_mean = np.mean([m['mcc'] for m in metrics_list])
        mcc_std = np.std([m['mcc'] for m in metrics_list])

        row = [
            dataset or "?",
            (model or "?").split(':')[-1],
            mode or "?",
            f"{f1_mean:.3f} +/- {f1_std:.3f}",
            f"{mcc_mean:.3f} +/- {mcc_std:.3f}"
        ]

        if has_roc_auc:
            roc_values = [m['roc_auc'] for m in metrics_list if m.get('roc_auc') is not None]
            if roc_values:
                roc_mean = np.mean(roc_values)
                roc_std = np.std(roc_values)
                row.append(f"{roc_mean:.3f} +/- {roc_std:.3f}")
            else:
                row.append("N/A")

        table.add_row(*row)

    console.print(table)
