#!/usr/bin/env python3
"""
Entry point for the experiment app.

Usage:
    python -m experiment_app          # Interactive mode
    python -m experiment_app --help   # Show help
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Tabular Classification Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m experiment_app                     # Interactive mode
    python -m experiment_app -c config.yaml      # Run from config file
    python -m experiment_app --quick             # Quick test run
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('experiment-results'),
        help='Output directory for results (default: experiment-results)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with minimal settings'
    )

    args = parser.parse_args()

    try:
        if args.config:
            run_from_config(args.config, args.output)
        elif args.quick:
            run_quick_test(args.output)
        else:
            run_interactive(args.output)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_interactive(output_dir: Path):
    """Run in interactive mode."""
    from .runner import InteractiveRunner
    from .executor import execute_suite

    runner = InteractiveRunner()
    suite = runner.run()

    if suite and suite.experiments:
        console.print("\n[bold green]Starting Experiment Execution[/bold green]\n")
        execute_suite(suite, output_dir)


def run_from_config(config_path: Path, output_dir: Path):
    """Run experiments from configuration file."""
    from .config import ExperimentSuite
    from .executor import execute_suite

    console.print(f"\n[bold]Loading configuration from: {config_path}[/bold]\n")

    if config_path.suffix in ('.yaml', '.yml'):
        suite = ExperimentSuite.from_yaml(config_path)
    elif config_path.suffix == '.json':
        suite = ExperimentSuite.from_json(config_path)
    else:
        console.print(f"[red]Unsupported format: {config_path.suffix}[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(suite.experiments)} experiment(s)")
    console.print(f"Total runs: {suite.get_total_runs()}\n")

    execute_suite(suite, output_dir)


def run_quick_test(output_dir: Path):
    """Run a quick test with minimal settings."""
    from .config import ExperimentConfig, ExperimentSuite
    from .executor import execute_suite

    console.print("\n[bold]Quick Test Mode[/bold]\n")
    console.print("Running minimal experiment for testing...\n")

    exp = ExperimentConfig(
        name="Quick Test",
        datasets=['breast_cancer'],
        models=['ollama:llama3.1:8b'],
        modes=['zero_shot'],
        seeds=[42],
        k_shots=3,
        max_samples=50,
        test_size=0.2
    )

    suite = ExperimentSuite(experiments=[exp])
    execute_suite(suite, output_dir)


if __name__ == "__main__":
    main()
