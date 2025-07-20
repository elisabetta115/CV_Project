import argparse
import os
import subprocess
import sys

from globals import *


def run_command(cmd):
    """Run a command and print output"""
    print(f"\nRunning: {cmd}")
    print("-" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Optimization Pipeline')
    parser.add_argument('--mode', type=str, 
                        choices=['full', 'train', 'optimize', 'evaluate', 'acdc', 'attribution', 'both'], 
                        default='full', help='Pipeline mode')
    parser.add_argument('--baseline-epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs for baseline training')
    
    # ACDC specific arguments
    parser.add_argument('--thresholds', nargs='+', type=float, 
                        default=[0.01, 0.05, 0.1, 0.15, 0.2],
                        help='ACDC thresholds to test')
    
    # Attribution Patching specific arguments
    parser.add_argument('--keep-ratios', nargs='+', type=float,
                        default=[0.9, 0.8, 0.7, 0.6, 0.5],
                        help='Attribution patching keep ratios to test')
    parser.add_argument('--analysis-batches', type=int, default=ACDC_ANALYSIS_BATCHES,
                        help='Number of batches for attribution computation')
    parser.add_argument('--use-task-loss', action='store_true',
                        help='Use task-specific loss for attribution patching')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip baseline training if model exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VISION TRANSFORMER OPTIMIZATION PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Device: {DEVICE}")
    print(f"Data path: {DATA_PATH}")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Tiny-ImageNet dataset not found at {DATA_PATH}")
        print("Please download the dataset first.")
        return 1
    
    # Train baseline model
    if args.mode in ['full', 'train', 'both']:
        if args.skip_training and os.path.exists(BASELINE_MODEL_PATH):
            print(f"\nSkipping baseline training - model exists at {BASELINE_MODEL_PATH}")
        else:
            print("\n" + "="*60)
            print("Training Baseline Model")
            print("="*60)
            
            cmd = f"python train_baseline.py --epochs {args.baseline_epochs}"
            if run_command(cmd) != 0:
                print("Error in baseline training!")
                return 1
    
    # Apply ACDC optimization with different thresholds
    if args.mode in ['full', 'optimize', 'acdc', 'both']:
        print("\n" + "="*60)
        print("ACDC Optimization")
        print("="*60)
        
        if not os.path.exists(BASELINE_MODEL_PATH):
            print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
            print("Please train the baseline model first.")
            return 1
        
        for threshold in args.thresholds:
            print(f"\nOptimizing with ACDC threshold={threshold}")
            cmd = f"python acdc_optimizer.py --threshold {threshold}"
            if run_command(cmd) != 0:
                print(f"Error in ACDC optimization with threshold {threshold}!")
                # Continue with other thresholds
    
    # Apply Attribution Patching optimization
    if args.mode in ['full', 'optimize', 'attribution', 'both']:
        print("\n" + "="*60)
        print("Edge Attribution Patching Optimization")
        print("="*60)
        
        if not os.path.exists(BASELINE_MODEL_PATH):
            print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
            print("Please train the baseline model first.")
            return 1
        
        for keep_ratio in args.keep_ratios:
            print(f"\nOptimizing with Attribution Patching keep_ratio={keep_ratio}")
            cmd = f"python attribution_patcher.py --keep-ratio {keep_ratio} --analysis-batches {args.analysis_batches}"
            if args.use_task_loss:
                cmd += " --use-task-loss"
            
            if run_command(cmd) != 0:
                print(f"Error in Attribution Patching with keep_ratio {keep_ratio}!")
                # Continue with other keep ratios
    
    # Evaluate all models
    if args.mode in ['full', 'evaluate', 'both']:
        print("\n" + "="*60)
        print("Comprehensive Evaluation")
        print("="*60)
        
        # Find all models
        all_models = []
        
        # ACDC optimized models
        for threshold in args.thresholds:
            
            model_path = f"{OPTIMIZED_MODEL_PREFIX}optimized_vit_threshold_{threshold}.pth"
            if os.path.exists(model_path):
                all_models.append(model_path)
        
        # Attribution patched models
        for keep_ratio in args.keep_ratios:
            model_path = f"{OPTIMIZED_MODEL_PREFIX}attribution_patched_model_keep_{keep_ratio}.pth"
            if os.path.exists(model_path):
                all_models.append(model_path)
        
        if not all_models and not os.path.exists(BASELINE_MODEL_PATH):
            print("No models found to evaluate!")
            return 1
        
        cmd = "python evaluator.py"
        if all_models:
            cmd += f" --model-paths {' '.join(all_models)}"
        
        if run_command(cmd) != 0:
            print("Error in evaluation!")
            return 1
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    if args.mode in ['full', 'both']:
        print("\nSummary:")
        print(f"- Baseline model: {BASELINE_MODEL_PATH}")
        
        if args.mode == 'full' or args.mode == 'both':
            print(f"- ACDC optimized models: {len(args.thresholds)} variants")
        
        if args.mode == 'both':
            print(f"- Attribution patched models: {len(args.keep_ratios)} variants")
        
        print(f"- Evaluation results: ./evaluation_results/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())