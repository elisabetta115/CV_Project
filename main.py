import argparse
import os
import subprocess
import sys

from globals import *


def run_command(cmd):
    """Run a command and print output"""
    print(f"\nRunning: {cmd}")
    print("-" * 60)
    result = subprocess.run(cmd, shell=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer ACDC Optimization Pipeline')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'optimize', 'evaluate'], 
                        default='full', help='Pipeline mode')
    parser.add_argument('--baseline-epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs for baseline training')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                        default=[0.01, 0.05, 0.1, 0.15, 0.2],
                        help='ACDC thresholds to test')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip baseline training if model exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VISION TRANSFORMER ACDC OPTIMIZATION PIPELINE")
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
    if args.mode in ['full', 'train']:
        if args.skip_training and os.path.exists(BASELINE_MODEL_PATH):
            print(f"\nSkipping baseline training - model exists at {BASELINE_MODEL_PATH}")
        else:
            print("\n" + "="*60)
            print("STEP 1: Training Baseline Model")
            print("="*60)
            
            cmd = f"python train_baseline.py --epochs {args.baseline_epochs}"
            if run_command(cmd) != 0:
                print("Error in baseline training!")
                return 1
    
    # Apply ACDC optimization with different thresholds
    if args.mode in ['full', 'optimize']:
        print("\n" + "="*60)
        print("STEP 2: ACDC Optimization")
        print("="*60)
        
        if not os.path.exists(BASELINE_MODEL_PATH):
            print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
            print("Please train the baseline model first.")
            return 1
        
        for threshold in args.thresholds:
            print(f"\nOptimizing with threshold={threshold}")
            cmd = f"python acdc_optimizer.py --threshold {threshold}"
            if run_command(cmd) != 0:
                print(f"Error in ACDC optimization with threshold {threshold}!")
                # Continue with other thresholds
    
    # Evaluate all models
    if args.mode in ['full', 'evaluate']:
        print("\n" + "="*60)
        print("STEP 3: Comprehensive Evaluation")
        print("="*60)
        
        # Find all models
        optimized_models = []
        for threshold in args.thresholds:
            model_path = f"{OPTIMIZED_MODEL_PREFIX}threshold_{threshold}.pth"
            if os.path.exists(model_path):
                optimized_models.append(model_path)
        
        if not optimized_models and not os.path.exists(BASELINE_MODEL_PATH):
            print("No models found to evaluate!")
            return 1
        
        cmd = "python evaluator.py"
        if optimized_models:
            cmd += f" --model-paths {' '.join(optimized_models)}"
        
        if run_command(cmd) != 0:
            print("Error in evaluation!")
            return 1
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    if args.mode == 'full':
        print("\nSummary:")
        print(f"- Baseline model: {BASELINE_MODEL_PATH}")
        print(f"- Optimized models: {len(args.thresholds)} variants")
        print(f"- Evaluation results: ./evaluation_results/")
        print("\nNext steps:")
        print("1. Check evaluation_results/evaluation_results.png for visual comparison")
        print("2. Review Pareto frontier to select best model for your use case")
        print("3. Deploy selected model based on your requirements")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())