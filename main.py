"""Command-line interface and entry point for continual-learning-benchmark."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.optim as optim

from cl_benchmark.datasets import TaskStream, get_benchmark
from cl_benchmark.evaluator import Evaluator
from cl_benchmark.models import MLP, ContinualModel, SimpleConvNet
from cl_benchmark.strategies.base import BaseStrategy
from cl_benchmark.strategies.ewc import EWC
from cl_benchmark.strategies.naive import Naive
from cl_benchmark.strategies.rehearsal import Rehearsal
from cl_benchmark.utils import seed_everything
from cl_benchmark.viz import plot_accuracy_trajectories


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for benchmark execution.

    Args:
        args: Optional custom argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Continual Learning Benchmark CLI runner in PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="split-mnist",
        choices=["split-mnist", "split-cifar10", "permuted-mnist"],
        help="Continual learning dataset protocol.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="naive",
        choices=["naive", "rehearsal", "ewc"],
        help="Continual learning training strategy algorithm.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "convnet"],
        help="Backbone neural network architecture.",
    )
    parser.add_argument(
        "--multi-head",
        action="store_true",
        help="Enable task-specific classification heads (multi-head setting). "
        "If omitted, a shared expanding head (single-head) is used.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Total number of tasks in the benchmark stream.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs per task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for training and testing.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=500,
        help="Replay memory buffer capacity for Rehearsal strategy.",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=400.0,
        help="Regularization penalty hyperparameter for EWC strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to output a performance plot (e.g. 'results.png').",
    )

    return parser.parse_args(args)


def create_backbone(model_name: str, dataset_name: str) -> torch.nn.Module:
    """Instantiate the backbone feature extractor based on model and dataset.

    Args:
        model_name: Name of the architecture ('mlp' or 'convnet').
        dataset_name: Name of the dataset stream.

    Returns:
        PyTorch module feature extractor.
    """
    model_name = model_name.lower().strip()
    dataset_name = dataset_name.lower().replace("-", "_").strip()

    if model_name == "mlp":
        if dataset_name == "split_cifar10":
            return MLP(input_dim=3 * 32 * 32, hidden_dim=256)
        else:
            return MLP(input_dim=28 * 28, hidden_dim=256)

    elif model_name == "convnet":
        if dataset_name != "split_cifar10":
            raise ValueError(
                f"SimpleConvNet expects 3-channel RGB images (Split-CIFAR10), "
                f"got '{dataset_name}'. Please use '--model mlp' for MNIST."
            )
        return SimpleConvNet(hidden_dim=512)

    raise ValueError(f"Unsupported model: '{model_name}'.")


def create_strategy(
    strategy_name: str,
    model: ContinualModel,
    optimizer: optim.Optimizer,
    memory_size: int,
    ewc_lambda: float,
    batch_size: int,
    device: torch.device,
) -> BaseStrategy:
    """Instantiate the continual learning strategy.

    Args:
        strategy_name: Name of the strategy ('naive', 'rehearsal', 'ewc').
        model: ContinualModel wrapper instance.
        optimizer: PyTorch optimizer instance.
        memory_size: Capacity of the exemplar buffer for rehearsal.
        ewc_lambda: Regularization coefficient for EWC.
        batch_size: DataLoader mini-batch size.
        device: Compute device.

    Returns:
        Configured BaseStrategy subclass instance.
    """
    strategy_name = strategy_name.lower().strip()

    if strategy_name == "naive":
        return Naive(model=model, optimizer=optimizer, device=device)
    elif strategy_name == "rehearsal":
        replay_batch_size = max(1, batch_size // 2)
        return Rehearsal(
            model=model,
            optimizer=optimizer,
            buffer_size=memory_size,
            replay_batch_size=replay_batch_size,
            device=device,
        )
    elif strategy_name == "ewc":
        return EWC(
            model=model,
            optimizer=optimizer,
            ewc_lambda=ewc_lambda,
            device=device,
        )

    raise ValueError(f"Unsupported strategy: '{strategy_name}'.")


def plot_results(
    evaluator: Evaluator,
    stream: TaskStream,
    dataset_name: str,
    strategy_name: str,
    multi_head: bool,
    save_path: str,
) -> None:
    """Plot and save task accuracy progression across continual learning stream.

    Args:
        evaluator: Evaluator holding the R evaluation matrix.
        stream: Stream containing task metadata.
        dataset_name: Dataset name identifier.
        strategy_name: Strategy name identifier.
        multi_head: Head configuration flag.
        save_path: Output file path for the plot.
    """
    head_type = "Multi-Head" if multi_head else "Single-Head"
    title = f"{strategy_name.capitalize()} on {dataset_name.upper()} ({head_type})"
    plot_accuracy_trajectories(
        task_accuracies=evaluator.R,
        save_path=save_path,
        title=title,
    )


def run_benchmark(args: argparse.Namespace) -> Evaluator:
    """Run continual learning benchmark workflow according to parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Evaluator containing completed experiment metrics and accuracy matrix.
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_key = args.dataset.replace("-", "_").strip()
    stream = get_benchmark(
        name=dataset_key,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
    )

    backbone = create_backbone(args.model, dataset_key)
    model = ContinualModel(backbone=backbone, multi_head=args.multi_head)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    strategy = create_strategy(
        strategy_name=args.strategy,
        model=model,
        optimizer=optimizer,
        memory_size=args.memory_size,
        ewc_lambda=args.ewc_lambda,
        batch_size=args.batch_size,
        device=device,
    )

    evaluator = Evaluator(num_tasks=len(stream))

    head_label = "Multi-Head" if args.multi_head else "Single-Head"
    print("=" * 60)
    print(" STARTING CONTINUAL LEARNING BENCHMARK")
    print("=" * 60)
    print(f" Dataset:        {args.dataset}")
    print(f" Strategy:       {args.strategy.capitalize()}")
    print(f" Model Backbone: {args.model.upper()} ({head_label})")
    print(f" Tasks:          {len(stream)}")
    print(f" Epochs/Task:    {args.epochs}")
    print(f" Batch Size:     {args.batch_size}")
    print(f" Learning Rate:  {args.lr}")
    print(f" Device:         {device}")
    print("=" * 60)

    # Continual Task Execution Loop
    for task_idx, task in enumerate(stream):
        print(
            f"\n>>> Task {task.task_id + 1}/{len(stream)} "
            f"(ID: {task.task_id}) | Classes: {task.classes}"
        )

        # 1. Register task head
        model.add_task_head(
            task_id=task.task_id,
            classes=task.classes,
            multi_head=args.multi_head,
        )

        # 2. Update optimizer parameter groups for dynamically added head
        existing_params = {
            p for group in optimizer.param_groups for p in group["params"]
        }
        new_params = [p for p in model.parameters() if p not in existing_params]
        if new_params:
            optimizer.add_param_group({"params": new_params})

        # 3. Before training task hook
        strategy.before_training_task(task)

        # 4. Train epochs
        for epoch in range(1, args.epochs + 1):
            strategy.before_training_epoch(epoch, task)
            epoch_loss = strategy.train_epoch(task)
            strategy.after_training_epoch(epoch, task)
            print(f"    Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f}")

        # 5. After training task hook
        strategy.after_training_task(task)

        # 6. Evaluate on all observed tasks up to current task ID (0 .. task_idx)
        print(f"    Evaluating across seen tasks (0..{task.task_id})...")
        for eval_idx in range(task_idx + 1):
            eval_task = stream[eval_idx]
            acc = strategy.evaluate(eval_task)
            evaluator.update(
                training_task_id=task.task_id,
                test_task_id=eval_task.task_id,
                accuracy=acc,
            )
            print(
                f"      -> Test Acc on Task {eval_task.task_id} "
                f"(Classes {eval_task.classes}): {acc * 100:.2f}%"
            )

    # Compute Summary Metrics
    avg_acc = evaluator.average_accuracy()
    bwt = evaluator.backward_transfer()
    fwt = evaluator.forward_transfer()
    fm = evaluator.forgetting_measure()

    print("\n" + "=" * 60)
    print(" FINAL BENCHMARK METRICS SUMMARY")
    print("=" * 60)
    print(f" Final Average Accuracy (A_T):  {avg_acc * 100:.2f}%")
    print(f" Backward Transfer (BWT):       {bwt * 100:+.2f}%")
    print(f" Forward Transfer (FWT):        {fwt * 100:+.2f}%")
    print(f" Forgetting Measure (FM):       {fm * 100:.2f}%")
    print("-" * 60)
    print(" Evaluation Matrix R (Accuracy after Task i on Task j):")
    print(evaluator.R)
    print("=" * 60)

    if args.save_plot:
        plot_results(
            evaluator=evaluator,
            stream=stream,
            dataset_name=args.dataset,
            strategy_name=args.strategy,
            multi_head=args.multi_head,
            save_path=args.save_plot,
        )

    return evaluator


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
