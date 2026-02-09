"""
Training Metrics Visualization

Generates graphs from JSON metrics logged during training.
Supports both step-level and epoch-level metrics visualization.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys


def load_jsonl_metrics(json_file: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load JSONL metrics file
    
    Args:
        json_file: Path to JSONL metrics file
        
    Returns:
        Tuple of (step_metrics, epoch_metrics)
    """
    step_metrics = []
    epoch_metrics = []
    
    if not json_file.exists():
        print(f"Warning: Metrics file not found: {json_file}")
        return step_metrics, epoch_metrics
    
    with open(json_file, 'r') as f:
        for line in f:
            if line.strip():
                metric = json.loads(line)
                # Classify as step or epoch metric based on presence of 'batch' field
                if 'batch' in metric:
                    step_metrics.append(metric)
                else:
                    epoch_metrics.append(metric)
    
    return step_metrics, epoch_metrics


def load_summary_metrics(summary_file: Path) -> Dict:
    """
    Load summary metrics file
    
    Args:
        summary_file: Path to summary JSON file
        
    Returns:
        Metrics dictionary
    """
    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return {}
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_loss_curves(step_metrics: List[Dict], epoch_metrics: List[Dict], 
                      output_file: Optional[Path] = None):
    """
    Plot training and validation loss curves
    
    Args:
        step_metrics: Step-level metrics
        epoch_metrics: Epoch-level metrics
        output_file: Optional output file path (PNG)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Step-level training loss
    if step_metrics:
        steps = [m['global_step'] for m in step_metrics]
        train_losses = [m['train_loss'] for m in step_metrics]
        
        axes[0].plot(steps, train_losses, label='Training Loss (per step)', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Global Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Over Steps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Epoch-level training and validation loss
    if epoch_metrics:
        epochs = [m['epoch'] for m in epoch_metrics]
        train_losses = [m['train_loss'] for m in epoch_metrics]
        val_losses = [m['val_loss'] for m in epoch_metrics]
        best_losses = [m['best_val_loss'] for m in epoch_metrics]
        
        axes[1].plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
        axes[1].plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)
        axes[1].plot(epochs, best_losses, label='Best Validation Loss', 
                    linestyle='--', linewidth=2, alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training vs Validation Loss Over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Loss curves saved to: {output_file}")
    else:
        plt.show()


def plot_epoch_metrics(epoch_metrics: List[Dict], output_file: Optional[Path] = None):
    """
    Plot epoch-level metrics in detail
    
    Args:
        epoch_metrics: Epoch-level metrics
        output_file: Optional output file path (PNG)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return
    
    if not epoch_metrics:
        print("No epoch metrics to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [m['epoch'] for m in epoch_metrics]
    epoch_times = [m['epoch_time'] for m in epoch_metrics]
    
    # Plot epoch times
    axes[0].bar(epochs, epoch_times, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Training Time per Epoch')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot loss comparison
    train_losses = [m['train_loss'] for m in epoch_metrics]
    val_losses = [m['val_loss'] for m in epoch_metrics]
    
    x_pos = list(range(len(epochs)))
    width = 0.35
    
    axes[1].bar([p - width/2 for p in x_pos], train_losses, width, label='Train Loss', alpha=0.8)
    axes[1].bar([p + width/2 for p in x_pos], val_losses, width, label='Val Loss', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Comparison per Epoch')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(epochs)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Epoch metrics saved to: {output_file}")
    else:
        plt.show()


def print_metrics_summary(summary: Dict):
    """
    Print a summary of metrics
    
    Args:
        summary: Metrics summary dictionary
    """
    if not summary:
        print("No metrics summary available")
        return
    
    print("\n" + "="*70)
    print("TRAINING METRICS SUMMARY")
    print("="*70)
    
    print(f"Model Type: {summary.get('model_type', 'N/A')}")
    print(f"Training Start: {summary.get('timestamp', 'N/A')}")
    
    epochs = summary.get('epochs', [])
    if epochs:
        print(f"\nTotal Epochs: {len(epochs)}")
        
        last_epoch = epochs[-1]
        print(f"Final Train Loss: {last_epoch.get('train_loss', 'N/A'):.4f}")
        print(f"Final Val Loss: {last_epoch.get('val_loss', 'N/A'):.4f}")
        print(f"Best Val Loss: {last_epoch.get('best_val_loss', 'N/A'):.4f}")
        
        total_time = sum(e.get('epoch_time', 0) for e in epochs)
        print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        avg_epoch_time = total_time / len(epochs) if epochs else 0
        print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training metrics from JSON logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from default metrics file
  python plot_metrics.py
  
  # Specify custom metrics file
  python plot_metrics.py gptmed/framework/logs/conversation_training_metrics.jsonl
  
  # Save plot to file instead of displaying
  python plot_metrics.py --output training_curves.png
  
  # Plot summary only
  python plot_metrics.py --summary-only
        """
    )
    
    parser.add_argument(
        'metrics_file',
        nargs='?',
        help='Path to JSONL metrics file (default: framework/logs/conversation_training_metrics.jsonl)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path for plots (PNG)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print metrics summary, do not generate plots'
    )
    parser.add_argument(
        '--epoch-only',
        action='store_true',
        help='Only show epoch-level metrics'
    )
    
    args = parser.parse_args()
    
    # Determine metrics file path
    if args.metrics_file:
        metrics_file = Path(args.metrics_file)
    else:
        # Try to find default location
        framework_dir = Path(__file__).parent.parent.parent
        metrics_file = framework_dir / 'logs' / 'conversation_training_metrics.jsonl'
    
    summary_file = metrics_file.parent / 'conversation_training_metrics_summary.json'
    
    # Load metrics
    step_metrics, epoch_metrics = load_jsonl_metrics(metrics_file)
    summary = load_summary_metrics(summary_file)
    
    if not step_metrics and not epoch_metrics:
        print(f"No metrics found in {metrics_file}")
        sys.exit(1)
    
    # Print summary
    print_metrics_summary(summary)
    
    # Generate plots
    if not args.summary_only:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = metrics_file.parent / "training_curves.png"
        
        if not args.epoch_only:
            plot_loss_curves(step_metrics, epoch_metrics, output_path)
        
        epoch_output = metrics_file.parent / "epoch_metrics.png"
        plot_epoch_metrics(epoch_metrics, epoch_output)


if __name__ == '__main__':
    main()
