import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import glob
import json
from sklearn.metrics import precision_recall_fscore_support

from globals import *
from network import VisionTransformer, reconstruct_pruned_model

from data import create_tiny_imagenet_datasets
from utils import set_seed, count_effective_parameters, measure_inference_time


class ClassificationMetrics:
    """
    Comprehensive metrics for classification model evaluation
    """
    
    @staticmethod
    def top_k_accuracy(outputs, labels, k=5):
        """Compute top-k accuracy"""
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item() * 100
    
    @staticmethod
    def per_class_accuracy(predictions, labels, num_classes):
        """Compute per-class accuracy"""
        accuracies = []
        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                correct = (predictions[mask] == cls).sum()
                acc = correct.float() / mask.sum()
                accuracies.append(acc.item())
        return np.mean(accuracies) * 100
    
    @staticmethod
    def confidence_metrics(probs, predictions, labels):
        """Compute confidence-related metrics"""
        # Get max probability (confidence)
        confidence, _ = probs.max(dim=1)
        
        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask
        
        metrics = {
            'mean_confidence': confidence.mean().item(),
            'correct_confidence': confidence[correct_mask].mean().item() if correct_mask.any() else 0,
            'incorrect_confidence': confidence[incorrect_mask].mean().item() if incorrect_mask.any() else 0,
            'confidence_gap': 0  # Will be calculated below
        }
        
        # Confidence gap: difference between correct and incorrect confidence
        if correct_mask.any() and incorrect_mask.any():
            metrics['confidence_gap'] = metrics['correct_confidence'] - metrics['incorrect_confidence']
        
        return metrics
    
    @staticmethod
    def entropy(probs):
        """Calculate prediction entropy (uncertainty measure)"""
        # Avoid log(0)
        probs = probs + 1e-10
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy.mean().item()


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_history = []
    
    def evaluate_classification_model(self, model, dataloader):
        """Evaluate classification model with comprehensive metrics"""
        model.eval()
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                
                all_outputs.append(outputs)
                all_labels.append(labels)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Convert to probabilities
        probs = torch.softmax(all_outputs, dim=1)
        
        # Get predictions
        _, predictions = all_outputs.max(1)
        
        # Basic accuracy
        accuracy = 100. * correct / total
        
        # Top-5 accuracy
        top5_acc = ClassificationMetrics.top_k_accuracy(all_outputs, all_labels, k=5)
        
        # Per-class accuracy
        per_class_acc = ClassificationMetrics.per_class_accuracy(predictions, all_labels, NUM_CLASSES)
        
        # Confidence metrics
        confidence_metrics = ClassificationMetrics.confidence_metrics(probs, predictions, all_labels)
        
        # Entropy
        avg_entropy = ClassificationMetrics.entropy(probs)
        
        # Precision, Recall, F1 (macro average)
        predictions_cpu = predictions.cpu().numpy()
        labels_cpu = all_labels.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_cpu, predictions_cpu, average='macro', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'per_class_accuracy': per_class_acc,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'avg_entropy': avg_entropy,
            **confidence_metrics,
            'outputs': all_outputs,
            'labels': all_labels,
            'predictions': predictions
        }
    
    def evaluate_model_comprehensive(self, model, dataloader, model_name="Model"):
        """Comprehensive evaluation including all metrics and inference time"""
        print(f"\nEvaluating {model_name}...")
        
        # Get classification results
        results = self.evaluate_classification_model(model, dataloader)
        
        # Measure inference time
        inference_time = measure_inference_time(
            model, dataloader, self.device, 
            num_batches=EVAL_NUM_BATCHES, 
            warmup=WARMUP_ITERATIONS
        )
        
        # Count parameters
        param_count = count_effective_parameters(model)
        
        # Calculate FLOPs estimate (simplified)
        # For ViT: roughly proportional to sequence length * embedding dim * layers
        flops_estimate = (NUM_PATCHES + 1) * EMBED_DIM * NUM_LAYERS * param_count / 1e9  # GFLOPs
        
        # Compile all metrics
        full_metrics = {
            'model_name': model_name,
            'accuracy': results['accuracy'],
            'top5_accuracy': results['top5_accuracy'],
            'per_class_accuracy': results['per_class_accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'mean_confidence': results['mean_confidence'],
            'correct_confidence': results['correct_confidence'],
            'incorrect_confidence': results['incorrect_confidence'],
            'confidence_gap': results['confidence_gap'],
            'avg_entropy': results['avg_entropy'],
            'inference_time': inference_time,
            'parameters': param_count,
            'gflops': flops_estimate
        }
        
        self.metrics_history.append(full_metrics)
        
        return full_metrics
    
    def compute_pareto_frontier(self, metrics_list, x_metric='inference_time', y_metric='accuracy'):
        """
        Compute Pareto frontier for the trade-off between two metrics
        For inference_time (minimize) vs accuracy (maximize)
        """
        points = [(m[x_metric], m[y_metric], m['model_name']) for m in metrics_list]
        points.sort(key=lambda p: p[0])  # Sort by x_metric
        
        pareto_points = []
        current_max_y = float('-inf')
        
        for point in points:
            x, y, name = point
            # For Pareto optimality: minimize x (inference time), maximize y (accuracy)
            if y > current_max_y:
                pareto_points.append(point)
                current_max_y = y
        
        return pareto_points
    
    def plot_results(self, metrics_list, save_path='evaluation_results.png'):
        """Plot comprehensive evaluation results including Pareto frontier"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Extract data
        model_names = [m['model_name'] for m in metrics_list]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        # Accuracy metrics comparison
        ax = axes[0, 0]
        metrics_to_plot = ['accuracy', 'top5_accuracy', 'f1_score']
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [m[metric] for m in metrics_list]
            label = metric.replace('_', ' ').title()
            if metric == 'accuracy':
                label = 'Top-1 Acc'
            elif metric == 'top5_accuracy':
                label = 'Top-5 Acc'
            elif metric == 'f1_score':
                label = 'F1 Score'
            ax.bar(x + i*width, values, width, label=label)
        
        ax.set_ylabel('Score (%)')
        ax.set_title('Classification Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Inference time and efficiency
        ax = axes[0, 1]
        inference_times = [m['inference_time']*1000 for m in metrics_list]
        throughput = [1000/t for t in inference_times]  # Images per second
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, inference_times, width, label='Inference Time', color='skyblue')
        bars2 = ax2.bar(x + width/2, throughput, width, label='Throughput', color='orange')
        
        ax.set_ylabel('Inference Time (ms/batch)', color='skyblue')
        ax2.set_ylabel('Throughput (batches/sec)', color='orange')
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_title('Speed Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits with 10% padding to show differences better
        inf_min, inf_max = min(inference_times), max(inference_times)
        inf_range = inf_max - inf_min
        ax.set_ylim(inf_min - 0.1*inf_range, inf_max + 0.1*inf_range)
        
        thr_min, thr_max = min(throughput), max(throughput)
        thr_range = thr_max - thr_min
        ax2.set_ylim(thr_min - 0.1*thr_range, thr_max + 0.1*thr_range)
        
        # Model complexity
        ax = axes[0, 2]
        params = [m['parameters']/1e6 for m in metrics_list]
        gflops = [m['gflops'] for m in metrics_list]
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, params, width, label='Parameters', color='green')
        bars2 = ax2.bar(x + width/2, gflops, width, label='GFLOPs', color='red')
        
        ax.set_ylabel('Parameters (M)', color='green')
        ax2.set_ylabel('GFLOPs', color='red')
        ax.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title('Model Complexity')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits with padding
        param_min, param_max = min(params), max(params)
        param_range = param_max - param_min
        ax.set_ylim(param_min - 0.1*param_range, param_max + 0.1*param_range)
        
        gflop_min, gflop_max = min(gflops), max(gflops)
        gflop_range = gflop_max - gflop_min
        ax2.set_ylim(gflop_min - 0.1*gflop_range, gflop_max + 0.1*gflop_range)
        
        # Pareto frontier: Error vs Inference Time (both to minimize)
        ax = axes[1, 0]
        
        # Convert accuracy to error rate for proper Pareto orientation
        error_rates = [100 - m['accuracy'] for m in metrics_list]
        
        # Compute Pareto frontier for error (minimize) vs inference time (minimize)
        points = [(m['inference_time']*1000, 100-m['accuracy'], m['model_name']) 
                for m in metrics_list]
        
        # Find Pareto optimal points (both objectives to minimize)
        pareto_points = []
        sorted_points = sorted(points, key=lambda p: p[0])  # Sort by inference time
        
        current_min_error = float('inf')
        for point in sorted_points:
            inf_time, error, name = point
            if error < current_min_error:
                pareto_points.append(point)
                current_min_error = error
        
        # Plot all points with error on x-axis, inference time on y-axis
        pareto_names = {name for x, y, name in pareto_points}
        for i, m in enumerate(metrics_list):
            name = m['model_name']
            is_opt = name in pareto_names
            error = 100 - m['accuracy']
            
            ax.scatter(
                error,  # Error on x-axis
                m['inference_time'] * 1000,  # Inference time on y-axis
                s=200 if is_opt else 100,
                c=[colors[i]],
                alpha=.9,
                marker='*' if is_opt else 'o',
                edgecolors='black',
                linewidth=2 if is_opt else 1,
                label=name,
                zorder=5 if is_opt else 3
            )
        
        # Plot Pareto frontier
        if pareto_points:
            pareto_x = [p[1] for p in pareto_points]  # errors
            pareto_y = [p[0] for p in pareto_points]  # inference times
            ax.plot(pareto_x, pareto_y,
                    linestyle='--',
                    color='red',
                    linewidth=2,
                    label='Pareto Frontier')
        
        ax.set_xlabel('Error Rate (%)')
        ax.set_ylabel('Inference Time (ms/batch)')
        ax.set_title('Pareto Frontier: Error vs Speed Trade-off')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Confidence analysis
        ax = axes[1, 1]
        conf_metrics = ['mean_confidence', 'correct_confidence', 'incorrect_confidence']
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(conf_metrics):
            values = [m[metric] for m in metrics_list]
            label = metric.replace('_', ' ').title()
            ax.bar(x + i*width, values, width, label=label)
        
        ax.set_ylabel('Confidence Score')
        ax.set_title('Model Confidence Analysis')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Efficiency score
        ax = axes[1, 2]
        # Calculate efficiency scores (accuracy per ms)
        efficiency_scores = [m['accuracy'] / (m['inference_time'] * 1000) for m in metrics_list]
        # Normalized efficiency (relative to best)
        max_efficiency = max(efficiency_scores)
        normalized_efficiency = [e/max_efficiency * 100 for e in efficiency_scores]
        
        bars = ax.bar(range(len(model_names)), normalized_efficiency, color=colors)
        ax.set_ylabel('Normalized Efficiency Score (%)')
        ax.set_title('Model Efficiency (Accuracy per ms)')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits with padding to show differences
        eff_min, eff_max = min(normalized_efficiency), max(normalized_efficiency)
        eff_range = eff_max - eff_min
        ax.set_ylim(eff_min - 0.1*eff_range, eff_max + 0.1*eff_range)
        
        # Add value labels
        for bar, eff in zip(bars, normalized_efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return pareto_points
    
    def generate_report(self, metrics_list, pareto_points):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Model comparison table
        print("\nModel Performance Summary:")
        print("-"*100)
        print(f"{'Model':<15} {'Top-1':<8} {'Top-5':<8} {'F1':<8} {'Inference':<12} {'Params':<10} {'Efficiency':<12}")
        print(f"{'Name':<15} {'Acc(%)':<8} {'Acc(%)':<8} {'(%)':<8} {'(ms/batch)':<12} {'(M)':<10} {'(Acc/ms)':<12}")
        print("-"*100)
        
        for m in metrics_list:
            efficiency = m['accuracy'] / (m['inference_time'] * 1000)
            print(f"{m['model_name']:<15} "
                  f"{m['accuracy']:<8.2f} "
                  f"{m['top5_accuracy']:<8.2f} "
                  f"{m['f1_score']:<8.2f} "
                  f"{m['inference_time']*1000:<12.2f} "
                  f"{m['parameters']/1e6:<10.2f} "
                  f"{efficiency:<12.4f}")
        
        # Confidence analysis
        print("\n" + "-"*100)
        print("CONFIDENCE ANALYSIS:")
        print("-"*100)
        print(f"{'Model':<15} {'Mean Conf':<12} {'Correct Conf':<14} {'Incorrect Conf':<16} {'Conf Gap':<10}")
        print("-"*100)
        
        for m in metrics_list:
            print(f"{m['model_name']:<15} "
                  f"{m['mean_confidence']:<12.4f} "
                  f"{m['correct_confidence']:<14.4f} "
                  f"{m['incorrect_confidence']:<16.4f} "
                  f"{m['confidence_gap']:<10.4f}")
        
        # Pareto optimal models
        print("\n" + "-"*100)
        print("PARETO OPTIMAL MODELS:")
        print("-"*100)
        print("Models on the Pareto frontier (optimal trade-off between accuracy and speed):")
        
        for x, y, name in pareto_points:
            efficiency = y / (x * 1000)
            print(f"  - {name}: {y:.2f}% accuracy, {x*1000:.2f}ms inference, {efficiency:.4f} efficiency")
        
        # Deployment recommendations
        print("\n" + "-"*100)
        print("DEPLOYMENT RECOMMENDATIONS:")
        print("-"*100)
        
        # Find best models for different scenarios
        best_accuracy = max(metrics_list, key=lambda m: m['accuracy'])
        fastest = min(metrics_list, key=lambda m: m['inference_time'])
        most_efficient = max(metrics_list, key=lambda m: m['accuracy']/(m['inference_time']*1000))
        smallest = min(metrics_list, key=lambda m: m['parameters'])
        best_f1 = max(metrics_list, key=lambda m: m['f1_score'])
        
        print(f"\n1. Maximum Accuracy (Cloud/Server deployment):")
        print(f"   → {best_accuracy['model_name']}: {best_accuracy['accuracy']:.2f}% top-1, {best_accuracy['top5_accuracy']:.2f}% top-5")
        
        print(f"\n2. Fastest Inference (Real-time applications):")
        print(f"   → {fastest['model_name']}: {fastest['inference_time']*1000:.2f}ms per batch ({1000/fastest['inference_time']/1000:.1f} batches/sec)")
        
        print(f"\n3. Best Efficiency (Balanced deployment):")
        print(f"   → {most_efficient['model_name']}: {most_efficient['accuracy']:.2f}% accuracy at {most_efficient['inference_time']*1000:.2f}ms")
        
        print(f"\n4. Smallest Model (Edge devices/Mobile):")
        print(f"   → {smallest['model_name']}: {smallest['parameters']/1e6:.2f}M parameters, {smallest['gflops']:.2f} GFLOPs")
        
        print(f"\n5. Most Robust (Best F1 Score):")
        print(f"   → {best_f1['model_name']}: {best_f1['f1_score']:.2f}% F1, {best_f1['precision']:.2f}% precision, {best_f1['recall']:.2f}% recall")
        
        # Summary statistics
        print("\n" + "-"*100)
        print("SUMMARY STATISTICS:")
        print("-"*100)
        
        accuracies = [m['accuracy'] for m in metrics_list]
        times = [m['inference_time']*1000 for m in metrics_list]
        
        print(f"Accuracy range: {min(accuracies):.2f}% - {max(accuracies):.2f}% (Δ = {max(accuracies)-min(accuracies):.2f}%)")
        print(f"Inference time range: {min(times):.2f}ms - {max(times):.2f}ms (Speedup = {max(times)/min(times):.2f}x)")
        print(f"Average confidence gap: {np.mean([m['confidence_gap'] for m in metrics_list]):.4f}")
        print(f"Models evaluated: {len(metrics_list)}")


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint, handling both baseline and optimized models"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']

    # Build the base ViT on the device
    base_model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_dim=config['mlp_dim'],
        dropout=config.get('dropout', 0.1)
    ).to(DEVICE)

    # If this checkpoint is a pruned model, first reconstruct the pruned net
    if 'removed_components' in checkpoint:
        print("Loading optimized model – reconstructing pruned architecture…")
        model = reconstruct_pruned_model(
            base_model,
            checkpoint['removed_components'],
            config
        ).to(DEVICE)
    else:
        model = base_model

    # Load weights with strict=False so pruned parameters are ignored
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=False
    )
    if missing_keys:
        print(f"Ignoring {len(missing_keys)} missing keys (pruned):")
        for k in missing_keys:
            print("  ", k)
    if unexpected_keys:
        print(f"Ignoring {len(unexpected_keys)} unexpected keys:")
        for k in unexpected_keys:
            print("  ", k)

    return model, checkpoint

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare Vision Transformer models')
    parser.add_argument('--model-paths', nargs='+', type=str, 
                        help='Paths to model checkpoints to evaluate')
    parser.add_argument('--baseline', type=str, default=BASELINE_MODEL_PATH,
                        help='Path to baseline model')
    parser.add_argument('--output-dir', type=str, default=EVALUATION_RESULTS_DIR,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--auto-find', action='store_true',
                        help='Automatically find all models in the optimized models directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Load dataset
    print("Loading validation dataset...")
    _, val_dataset = create_tiny_imagenet_datasets(
        DATA_PATH, NORMALIZE_MEAN, NORMALIZE_STD
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=DEVICE)
    
    # Collect all model paths
    all_model_paths = []
    
    # Add baseline
    if os.path.exists(args.baseline):
        all_model_paths.append(('Baseline', args.baseline))
    
    # Add specified models
    if args.model_paths:
        for path in args.model_paths:
            if os.path.exists(path):
                # Parse model name from path
                basename = os.path.basename(path).replace('.pth', '')
                
                # Determine model type and name
                if 'threshold' in basename:
                    # ACDC model
                    threshold = basename.split('threshold_')[1]
                    name = f'ACDC_t{threshold}'
                elif 'keep' in basename:
                    # EAP model
                    keep_ratio = basename.split('keep_')[1]
                    name = f'EAP_k{keep_ratio}'
                else:
                    name = basename
                    
                all_model_paths.append((name, path))
    
    # Auto-find models if requested or if no models specified
    if args.auto_find or (not args.model_paths and not all_model_paths):
        print("Auto-finding optimized models...")
        
        # Find all models in optimized directory
        optimized_models = glob.glob(os.path.join(OPTIMIZED_MODEL_DIR, "*.pth"))
        
        for path in optimized_models:
            basename = os.path.basename(path).replace('.pth', '')
            
            # Parse ACDC models
            if ACDC_MODEL_BASE_NAME in basename and 'threshold' in basename:
                threshold = basename.split('threshold_')[1]
                name = f'ACDC_t{threshold}'
                all_model_paths.append((name, path))
            
            # Parse EAP models
            elif EAP_MODEL_BASE_NAME in basename and 'keep' in basename:
                keep_ratio = basename.split('keep_')[1]
                name = f'EAP_k{keep_ratio}'
                all_model_paths.append((name, path))
    
    if not all_model_paths:
        print("No models found to evaluate!")
        return 1
    
    print(f"\nFound {len(all_model_paths)} models to evaluate:")
    for name, path in all_model_paths:
        print(f"  - {name}: {path}")
    
    # Evaluate all models
    all_metrics = []
    for model_name, model_path in all_model_paths:
        print(f"\nLoading {model_name} from {model_path}...")
        
        try:
            model, checkpoint = load_model_from_checkpoint(model_path)
            
            # Evaluate
            metrics = evaluator.evaluate_model_comprehensive(model, val_loader, model_name)
            all_metrics.append(metrics)
            
            # Print immediate results
            print(f"Results for {model_name}:")
            print(f"  - Top-1 Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  - Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
            print(f"  - F1 Score: {metrics['f1_score']:.2f}%")
            print(f"  - Inference time: {metrics['inference_time']*1000:.2f}ms")
            print(f"  - Parameters: {metrics['parameters']/1e6:.2f}M")
            print(f"  - Efficiency: {metrics['accuracy']/(metrics['inference_time']*1000):.4f} acc/ms")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    if not all_metrics:
        print("No models were successfully evaluated!")
        return 1
    
    # Plot results and compute Pareto frontier
    plot_path = os.path.join(args.output_dir, 'evaluation_results.png')
    pareto_points = evaluator.plot_results(all_metrics, save_path=plot_path)
    
    # Generate report
    evaluator.generate_report(all_metrics, pareto_points)
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Visualization: {plot_path}")
    print(f"  - Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()