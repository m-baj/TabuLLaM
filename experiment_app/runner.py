"""
Interactive experiment runner with terminal UI.
Uses InquirerPy for prompts and Rich for formatting.
"""

from pathlib import Path
from typing import Optional
import sys

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box

from .config import (
    ExperimentConfig,
    ExperimentSuite,
    AVAILABLE_DATASETS,
    AVAILABLE_MODELS,
    AVAILABLE_MODES,
    AVAILABLE_PREDICTION_MODES,
    DEFAULT_SEEDS,
    is_binary_dataset
)

console = Console()


class InteractiveRunner:
    """Interactive CLI for configuring and running experiments."""

    def __init__(self):
        self.suite = ExperimentSuite(experiments=[])

    def run(self) -> Optional[ExperimentSuite]:
        """Main entry point for interactive mode."""
        self.show_welcome()

        while True:
            action = self.ask_main_menu()

            if action == "individual":
                self.configure_individual_experiment()
            elif action == "batch":
                self.configure_batch_experiment()
            elif action == "load":
                self.load_from_file()
            elif action == "review":
                self.review_configuration()
            elif action == "manage":
                self.manage_experiments()
            elif action == "save":
                self.save_configuration()
            elif action == "execute":
                if self.confirm_execution():
                    return self.suite
            elif action == "exit":
                console.print("\n[yellow]👋 Goodbye![/yellow]\n")
                sys.exit(0)

    def show_welcome(self):
        """Display welcome message."""
        console.clear()
        welcome_text = """
[bold cyan]╔══════════════════════════════════════════╗
║  🤖 LLM Tabular Classification Runner 🤖 ║
╚══════════════════════════════════════════╝[/bold cyan]

[dim]Configure and run experiments using the
TabularLLMClassifier library.[/dim]
        """
        console.print(welcome_text)

    def ask_main_menu(self) -> str:
        """Show main menu and get user choice."""
        console.print()
        choices = [
            Choice(value="individual", name="🧪 Define Individual Experiment"),
            Choice(value="batch", name="📦 Configure Batch Experiment"),
            Separator(),
            Choice(value="load", name="📂 Load Configuration from File"),
            Choice(value="save", name="💾 Save Current Configuration"),
            Separator(),
            Choice(value="review", name="📋 Review Current Configuration"),
        ]

        if self.suite.experiments:
            choices.append(Choice(value="manage", name="⚙️  Manage Experiments"))
            choices.append(Separator())
            choices.append(Choice(value="execute", name="🚀 Execute Experiments"))
        else:
            choices.append(Separator())
            choices.append(Choice(
                value="execute",
                name="🚀 Execute Experiments (no experiments defined)",
                enabled=False
            ))

        choices.append(Separator())
        choices.append(Choice(value="exit", name="🚪 Exit"))

        action = inquirer.select(
            message="What would you like to do?",
            choices=choices,
            default="individual"
        ).execute()

        return action

    def configure_individual_experiment(self):
        """Configure a single experiment."""
        console.print("\n[bold cyan]🧪 Individual Experiment Configuration[/bold cyan]\n")

        try:
            exp = self.create_experiment()
            self.show_experiment_summary(exp)

            confirm = inquirer.confirm(
                message="Add this experiment?",
                default=True
            ).execute()

            if confirm:
                self.suite.add_experiment(exp)
                console.print(f"\n[green]✅ Experiment added! Total: {len(self.suite.experiments)}[/green]\n")
            else:
                console.print("\n[yellow]❌ Experiment cancelled[/yellow]\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Cancelled[/yellow]\n")

    def create_experiment(self) -> ExperimentConfig:
        """Create a single experiment configuration through prompts."""
        # Experiment name
        name = inquirer.text(
            message="Experiment name (optional):",
            default="",
        ).execute()

        # Datasets
        datasets = inquirer.checkbox(
            message="Select dataset(s):",
            choices=[
                Choice(
                    value=ds,
                    name=f"{ds} ({'binary' if is_binary_dataset(ds) else 'multiclass'})"
                )
                for ds in AVAILABLE_DATASETS
            ],
            validate=lambda result: len(result) > 0,
            invalid_message="Select at least one dataset"
        ).execute()

        # Models
        models = inquirer.checkbox(
            message="Select model(s):",
            choices=[Choice(value=m, name=m) for m in AVAILABLE_MODELS],
            validate=lambda result: len(result) > 0,
            invalid_message="Select at least one model"
        ).execute()

        # Modes
        modes = inquirer.checkbox(
            message="Select mode(s):",
            choices=[
                Choice(value="zero_shot", name="🎯 Zero-shot (no examples)"),
                Choice(value="random_few_shot", name="🎲 Random Few-shot (random examples)"),
                Choice(value="semantic_few_shot", name="🔍 Semantic Few-shot (similar examples)")
            ],
            validate=lambda result: len(result) > 0,
            invalid_message="Select at least one mode"
        ).execute()

        # Seeds
        seed_choice = inquirer.select(
            message="Select seeds:",
            choices=[
                Choice(value="default", name=f"Default seeds {DEFAULT_SEEDS}"),
                Choice(value="custom", name="Custom seeds")
            ],
            default="default"
        ).execute()

        if seed_choice == "default":
            seeds = DEFAULT_SEEDS.copy()
        else:
            seeds_input = inquirer.text(
                message="Enter seeds (comma-separated):",
                default=",".join(map(str, DEFAULT_SEEDS)),
                validate=lambda text: all(s.strip().isdigit() for s in text.split(",")),
                invalid_message="Enter valid comma-separated integers"
            ).execute()
            seeds = [int(s.strip()) for s in seeds_input.split(",")]

        # Prediction mode
        prediction_mode = inquirer.select(
            message="Prediction mode:",
            choices=[
                Choice(value="predict", name="⚡ Standard (predict) - faster, no probabilities"),
                Choice(value="predict_proba", name="📊 Probabilities (predict_proba) - includes confidence scores")
            ],
            default="predict_proba"
        ).execute()

        # Advanced options
        show_advanced = inquirer.confirm(
            message="Configure advanced options?",
            default=False
        ).execute()

        k_shots = 5
        max_samples = 500
        test_size = 0.2

        if show_advanced:
            k_shots = int(inquirer.text(
                message="K-shots (number of examples for few-shot):",
                default="5",
                validate=lambda text: text.isdigit() and int(text) > 0,
                invalid_message="Enter a positive integer"
            ).execute())

            max_samples_input = inquirer.text(
                message="Max samples (empty for full dataset):",
                default="500",
            ).execute()
            max_samples = int(max_samples_input) if max_samples_input else None

            test_size = float(inquirer.text(
                message="Test set size (0.0-1.0):",
                default="0.2",
                validate=lambda text: 0 < float(text) < 1,
                invalid_message="Enter a value between 0 and 1"
            ).execute())

        return ExperimentConfig(
            name=name or None,
            datasets=datasets,
            models=models,
            modes=modes,
            seeds=seeds,
            k_shots=k_shots,
            max_samples=max_samples,
            test_size=test_size,
            prediction_mode=prediction_mode
        )

    def configure_batch_experiment(self):
        """Configure a batch experiment with multiple combinations."""
        console.print("\n[bold cyan]📦 Batch Experiment Configuration[/bold cyan]\n")
        console.print("[dim]Run experiments across multiple datasets/models/modes[/dim]\n")

        # Select scope
        scope = inquirer.select(
            message="What scope?",
            choices=[
                Choice(value="all", name="ALL combinations (all datasets, models, modes)"),
                Choice(value="filtered", name="FILTERED combinations (select specific)")
            ],
            default="filtered"
        ).execute()

        if scope == "all":
            datasets = AVAILABLE_DATASETS.copy()
            models = AVAILABLE_MODELS.copy()
            modes = AVAILABLE_MODES.copy()
        else:
            datasets = inquirer.checkbox(
                message="Select dataset(s) (empty for all):",
                choices=[
                    Choice(
                        value=ds,
                        name=f"{ds} ({'binary' if is_binary_dataset(ds) else 'multiclass'})"
                    )
                    for ds in AVAILABLE_DATASETS
                ],
            ).execute() or AVAILABLE_DATASETS.copy()

            models = inquirer.checkbox(
                message="Select model(s) (empty for all):",
                choices=[Choice(value=m, name=m) for m in AVAILABLE_MODELS],
            ).execute() or AVAILABLE_MODELS.copy()

            modes = inquirer.checkbox(
                message="Select mode(s) (empty for all):",
                choices=[
                    Choice(value="zero_shot", name="Zero-shot"),
                    Choice(value="random_few_shot", name="Random Few-shot"),
                    Choice(value="semantic_few_shot", name="Semantic Few-shot")
                ],
            ).execute() or AVAILABLE_MODES.copy()

        # Seeds
        seed_choice = inquirer.select(
            message="Select seeds:",
            choices=[
                Choice(value="default", name=f"Default seeds {DEFAULT_SEEDS}"),
                Choice(value="custom", name="Custom seeds")
            ],
            default="default"
        ).execute()

        if seed_choice == "default":
            seeds = DEFAULT_SEEDS.copy()
        else:
            seeds_input = inquirer.text(
                message="Enter seeds (comma-separated):",
                default=",".join(map(str, DEFAULT_SEEDS)),
                validate=lambda text: all(s.strip().isdigit() for s in text.split(",")),
                invalid_message="Enter valid comma-separated integers"
            ).execute()
            seeds = [int(s.strip()) for s in seeds_input.split(",")]

        # Max samples
        max_samples_input = inquirer.text(
            message="Max samples (empty for full dataset):",
            default="500",
        ).execute()
        max_samples = int(max_samples_input) if max_samples_input else None

        # Prediction mode
        prediction_mode = inquirer.select(
            message="Prediction mode:",
            choices=[
                Choice(value="predict", name="⚡ Standard (predict) - faster, no probabilities"),
                Choice(value="predict_proba", name="📊 Probabilities (predict_proba) - includes confidence scores")
            ],
            default="predict_proba"
        ).execute()

        exp = ExperimentConfig(
            name="Batch Experiment",
            datasets=datasets,
            models=models,
            modes=modes,
            seeds=seeds,
            max_samples=max_samples,
            prediction_mode=prediction_mode
        )

        self.suite.add_experiment(exp)
        self.show_experiment_summary(exp)
        console.print("\n[green]✅ Batch experiment configured[/green]\n")

    def show_experiment_summary(self, exp: ExperimentConfig):
        """Display a summary of an experiment."""
        table = Table(title=f"Experiment: {exp.name or 'Unnamed'}", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Datasets", ", ".join(exp.datasets))
        table.add_row("Models", ", ".join(exp.models))
        table.add_row("Modes", ", ".join(exp.modes))
        table.add_row("Seeds", str(exp.seeds))
        table.add_row("K-shots", str(exp.k_shots))
        table.add_row("Max Samples", str(exp.max_samples) if exp.max_samples else "Full dataset")
        table.add_row("Test Size", str(exp.test_size))
        table.add_row("Prediction Mode", exp.prediction_mode)
        table.add_row("Total Runs", str(exp.get_total_runs()))

        console.print(table)

    def review_configuration(self):
        """Review all configured experiments."""
        if not self.suite.experiments:
            console.print("\n[yellow]⚠️  No experiments configured yet[/yellow]\n")
            return

        console.print("\n[bold cyan]📋 Configuration Review[/bold cyan]\n")

        for i, exp in enumerate(self.suite.experiments, 1):
            console.print(f"\n[bold]Experiment {i}:[/bold]")
            self.show_experiment_summary(exp)

        total_runs = self.suite.get_total_runs()
        console.print(f"\n[bold green]Total estimated runs: {total_runs}[/bold green]\n")

    def manage_experiments(self):
        """Manage configured experiments."""
        if not self.suite.experiments:
            console.print("\n[yellow]⚠️  No experiments configured yet[/yellow]\n")
            return

        console.print("\n[bold cyan]⚙️  Manage Experiments[/bold cyan]\n")

        for i, exp in enumerate(self.suite.experiments, 1):
            name = exp.name or f"Experiment {i}"
            console.print(f"{i}. [cyan]{name}[/cyan] - {len(exp.datasets)} dataset(s)")

        console.print()

        choices = []
        for i, exp in enumerate(self.suite.experiments, 1):
            name = exp.name or f"Experiment {i}"
            choices.append(Choice(value=i-1, name=f"Remove: {name}"))

        choices.append(Separator())
        choices.append(Choice(value="clear_all", name="Remove All"))
        choices.append(Choice(value="cancel", name="Back"))

        action = inquirer.select(
            message="Action:",
            choices=choices,
            default="cancel"
        ).execute()

        if action == "cancel":
            return
        elif action == "clear_all":
            confirm = inquirer.confirm(
                message="Remove ALL experiments?",
                default=False
            ).execute()
            if confirm:
                self.suite.experiments.clear()
                console.print("\n[green]All experiments removed[/green]\n")
        else:
            exp_name = self.suite.experiments[action].name or f"Experiment {action + 1}"
            confirm = inquirer.confirm(
                message=f"Remove '{exp_name}'?",
                default=True
            ).execute()
            if confirm:
                self.suite.experiments.pop(action)
                console.print(f"\n[green]Removed '{exp_name}'[/green]\n")

    def save_configuration(self):
        """Save configuration to file."""
        if not self.suite.experiments:
            console.print("\n[yellow]⚠️  No experiments to save[/yellow]\n")
            return

        format_choice = inquirer.select(
            message="Format:",
            choices=[
                Choice(value="yaml", name="YAML"),
                Choice(value="json", name="JSON")
            ],
            default="yaml"
        ).execute()

        filename = inquirer.text(
            message="Filename:",
            default=f"experiment_config.{format_choice}",
        ).execute()

        filepath = Path(filename)

        try:
            if format_choice == "yaml":
                self.suite.to_yaml(filepath)
            else:
                self.suite.to_json(filepath)

            console.print(f"\n[green]💾 Saved to: {filepath}[/green]\n")

            with open(filepath, 'r') as f:
                content = f.read()
            syntax = Syntax(content, format_choice, theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=str(filepath), border_style="green"))

        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]\n")

    def load_from_file(self):
        """Load configuration from file."""
        filename = inquirer.text(
            message="Config file path:",
            default="experiment_config.yaml",
        ).execute()

        filepath = Path(filename)

        if not filepath.exists():
            console.print(f"\n[red]❌ File not found: {filepath}[/red]\n")
            return

        try:
            if filepath.suffix in ('.yaml', '.yml'):
                self.suite = ExperimentSuite.from_yaml(filepath)
            elif filepath.suffix == '.json':
                self.suite = ExperimentSuite.from_json(filepath)
            else:
                console.print(f"\n[red]❌ Unsupported format: {filepath.suffix}[/red]\n")
                return

            console.print(f"\n[green]📂 Loaded from: {filepath}[/green]\n")
            self.review_configuration()

        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]\n")

    def confirm_execution(self) -> bool:
        """Confirm before executing."""
        console.print("\n[bold yellow]🚀 Ready to Execute[/bold yellow]\n")
        self.review_configuration()

        confirm = inquirer.confirm(
            message="Execute these experiments?",
            default=True
        ).execute()

        return confirm
