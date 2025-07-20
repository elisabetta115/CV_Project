 































# 


# 


# 


# 


# 










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
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'optimize', 'evaluate'], 
                        default='full', help='Pipeline mode')
    parser.add_argument('--method', type=str, choices=['acdc', 'eap', 'both'], 
                        default='both', help='Optimization method to use')
    parser.add_argument('--baseline-epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs for baseline training')
    
    # Model naming arguments
    parser.add_argument('--baseline-name', type=str, default=BASELINE_MODEL_NAME,
                        help='Name for baseline model file')
    parser.add_argument('--acdc-base-name', type=str, default=ACDC_MODEL_BASE_NAME,
                        help='Base name for ACDC optimized models')
    parser.add_argument('--eap-base-name', type=str, default=EAP_MODEL_BASE_NAME,
                        help='Base name for EAP optimized models')
    
    # ACDC specific arguments
    parser.add_argument('--acdc-thresholds', nargs='+', type=float, 
                        default=[0.01, 0.05, 0.1, 0.15, 0.2],
                        help='ACDC thresholds to test')
    
    # EAP specific arguments
    parser.add_argument('--eap-keep-ratios', nargs='+', type=float, 
                        default=[0.9, 0.8, 0.7, 0.6, 0.5],
                        help='EAP keep ratios to test')
    
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip baseline training if model exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VISION TRANSFORMER OPTIMIZATION PIPELINE")
    print("VISION TRANSFORMER OPTIMIZATION PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Method: {args.method}")
    print(f"Device: {DEVICE}")
    print(f"Data path: {DATA_PATH}")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Tiny-ImageNet dataset not found at {DATA_PATH}")
        print("Please download the dataset first.")
        return 1
    
    # Build baseline model path
    baseline_model_path = os.path.join('./models/', args.baseline_name)
    
    # Train baseline model
    if args.mode in ['full', 'train']:
        if args.skip_training and os.path.exists(baseline_model_path):
            print(f"\nSkipping baseline training - model exists at {baseline_model_path}")
        else:
            print("\n" + "="*60)
            print("Training Baseline Model")
            print("="*60)
            
            cmd = f"python train_baseline.py --epochs {args.baseline_epochs} --save-path {baseline_model_path}"
            if run_command(cmd) != 0:
                print("Error in baseline training!")
                return 1
    
    # Apply optimization
    if args.mode in ['full', 'optimize']:
        print("\n" + "="*60)
        print("STEP 2: Model Optimization")
        print("="*60)
        
        if not os.path.exists(baseline_model_path):
            print(f"Error: Baseline model not found at {baseline_model_path}")
            print("Please train the baseline model first.")
            return 1
        
        # ACDC optimization
        if args.method in ['acdc', 'both']:
            print("\n--- ACDC Optimization ---")
            for threshold in args.acdc_thresholds:
                print(f"\nOptimizing with ACDC threshold={threshold}")
                cmd = (f"python acdc_optimizer.py --threshold {threshold} "
                       f"--model-path {baseline_model_path} "
                       f"--output-base-name {args.acdc_base_name}")
                if run_command(cmd) != 0:
                    print(f"Error in ACDC optimization with threshold {threshold}!")
                    # Continue with other thresholds
        
        # EAP optimization
        if args.method in ['eap', 'both']:
            print("\n--- Attribution Patching Optimization ---")
            for keep_ratio in args.eap_keep_ratios:
                print(f"\nOptimizing with EAP keep_ratio={keep_ratio}")
                cmd = (f"python attribution_patcher.py --keep-ratio {keep_ratio} "
                       f"--model-path {baseline_model_path} "
                       f"--output-base-name {args.eap_base_name}")
                if run_command(cmd) != 0:
                    print(f"Error in EAP optimization with keep_ratio {keep_ratio}!")
                    # Continue with other ratios
    
    # Evaluate all models
    if args.mode in ['full', 'evaluate', 'both']:
        print("\n" + "="*60)
        print("Comprehensive Evaluation")
        print("="*60)
        
        # Find all models to evaluate
        model_paths = []
        
        # Add baseline
        if os.path.exists(baseline_model_path):
            model_paths.append(baseline_model_path)
        
        # Find ACDC models
        if args.method in ['acdc', 'both']:
            for threshold in args.acdc_thresholds:
                model_path = os.path.join(OPTIMIZED_MODEL_DIR, f"{args.acdc_base_name}_threshold_{threshold}.pth")
                if os.path.exists(model_path):
                    model_paths.append(model_path)
        
        # Find EAP models
        if args.method in ['eap', 'both']:
            for keep_ratio in args.eap_keep_ratios:
                model_path = os.path.join(OPTIMIZED_MODEL_DIR, f"{args.eap_base_name}_keep_{keep_ratio}.pth")
                if os.path.exists(model_path):
                    model_paths.append(model_path)
        
        if not model_paths:
            print("No models found to evaluate!")
            return 1
        
        cmd = f"python evaluator.py --baseline {baseline_model_path}"
        if len(model_paths) > 1:  # More than just baseline
            cmd += f" --model-paths {' '.join(model_paths[1:])}"
        
        if run_command(cmd) != 0:
            print("Error in evaluation!")
            return 1
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    if args.mode in ['full', 'both']:
        print("\nSummary:")
        print(f"- Baseline model: {baseline_model_path}")
        
        num_optimized = 0
        if args.method in ['acdc', 'both']:
            num_optimized += len(args.acdc_thresholds)
        if args.method in ['eap', 'both']:
            num_optimized += len(args.eap_keep_ratios)
            
        print(f"- Optimized models: {num_optimized} variants")
        print(f"- Evaluation results: ./evaluation_results/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())