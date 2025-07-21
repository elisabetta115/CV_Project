# Vision Transformer Optimization Framework

This repository implements a modular framework for training Vision Transformers (ViT) and optimizing them using Automated Circuit DisCovery (ACDC) and Edge Attribution Patching (EAP) techniques. The framework includes comprehensive evaluation tools to analyze the trade-off between model performance and inference speed.

## Project Structure

```
.
├── globals.py             # All configuration parameters and constants
├── network.py             # Vision Transformer related classes and functions
├── data.py                # Dataset creation and augmentation
├── utils.py               # Utility functions and helper classes
├── train_baseline.py      # Script to train the baseline ViT model
├── acdc_optimizer.py      # ACDC optimization implementation
├── eap_optimizer.py       # EAP optimization implementation
├── evaluator.py           # Comprehensive evaluation and Pareto analysis
├── main.py                # Main pipeline script
└── README.md              # This file
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

## Dataset

The framework expects Tiny-ImageNet dataset. Download and extract it to `./tiny-imagenet-200/`:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
1. Train a baseline Vision Transformer
2. Apply both ACDC and EAP optimizations with multiple configurations
3. Evaluate all models and generate comparison reports

### Pipeline Modes

#### Run Both Optimization Methods (Default)
```bash
python main.py --method both
```

#### Run Only ACDC Optimization
```bash
python main.py --method acdc --acdc-thresholds 0.01 0.05 0.1
```

#### Run Only EAP Optimization
```bash
python main.py --method eap --eap-keep-ratios 0.9 0.8 0.7
```

### Individual Steps

#### 1. Train Baseline Model

```bash
python train_baseline.py --epochs 50
```

Options:
- `--epochs`: Number of training epochs (default: 50)
- `--save-path`: Path to save the trained model (default: ./models/baseline_vit.pth)
- `--resume`: Path to checkpoint to resume training

#### 2. ACDC Optimization

Apply ACDC optimization to a trained model:

```bash
python acdc_optimizer.py --threshold 0.01 --model-path ./models/baseline_vit.pth
```

Options:
- `--model-path`: Path to baseline model (default: ./models/baseline_vit.pth)
- `--threshold`: KL divergence threshold for component removal (default: 0.01)
- `--analysis-batches`: Number of batches for KL divergence computation (default: 30)
- `--output-base-name`: Base name for output model file (default: acdc_optimized_vit)

#### 3. Edge Attribution Patching (EAP)

Apply EAP optimization to a trained model:

```bash
python eap_optimizer.py --keep-ratio 0.9 --model-path ./models/baseline_vit.pth
```

Options:
- `--model-path`: Path to baseline model (default: ./models/baseline_vit.pth)
- `--keep-ratio`: Fraction of components to keep by attribution score (default: 0.9)
- `--analysis-batches`: Number of batches for attribution computation (default: 20)
- `--use-task-loss`: Use task-specific loss instead of KL divergence
- `--output-base-name`: Base name for output model file (default: eap_optimized_vit)

#### 4. Evaluate Models

Evaluate and compare multiple models:

```bash
# Automatically find and evaluate all models
python evaluator.py --auto-find

# Or specify models manually
python evaluator.py --model-paths ./optimized_models/acdc_optimized_vit_threshold_0.01.pth ./optimized_models/eap_optimized_vit_keep_0.9.pth
```

Options:
- `--model-paths`: List of model paths to evaluate
- `--baseline`: Path to baseline model (default: ./models/baseline_vit.pth)
- `--output-dir`: Directory for results (default: ./evaluation_results)
- `--batch-size`: Batch size for evaluation
- `--auto-find`: Automatically find all models in the optimized models directory

### Advanced Usage

#### Custom Model Names

```bash
python main.py --baseline-name my_baseline.pth --acdc-base-name my_acdc --eap-base-name my_eap
```

#### Skip Baseline Training

If you already have a trained baseline:

```bash
python main.py --skip-training
```

#### Run Specific Pipeline Stage

```bash
# Only training
python main.py --mode train

# Only optimization
python main.py --mode optimize --method both

# Only evaluation
python main.py --mode evaluate
```

## File Organization

The framework uses a consistent file naming convention:

- **Baseline models**: `./models/{baseline_name}.pth`
- **ACDC optimized models**: `./optimized_models/{acdc_base_name}_threshold_{value}.pth`
- **EAP optimized models**: `./optimized_models/{eap_base_name}_keep_{value}.pth`

Default names are defined in `globals.py` but can be overridden via command-line arguments.

## Configuration

All important parameters are centralized in `globals.py`:

### Model Architecture
- `EMBED_DIM`: Embedding dimension (576)
- `NUM_HEADS`: Number of attention heads (6)
- `NUM_LAYERS`: Number of transformer layers (6)
- `MLP_DIM`: MLP hidden dimension (1536)
- `PATCH_SIZE`: Patch size for image tokenization (8)

### Training Parameters
- `BATCH_SIZE`: Training batch size (64)
- `LEARNING_RATE`: Initial learning rate (1e-4)
- `NUM_EPOCHS`: Training epochs (50)
- `WEIGHT_DECAY`: Adam weight decay (0.01)
- `LABEL_SMOOTHING`: Label smoothing factor (0.1)

### Optimization Parameters
- **ACDC**: 
  - `ACDC_THRESHOLD`: Default KL divergence threshold (0.01)
  - `ACDC_ANALYSIS_BATCHES`: Batches for importance analysis (30)
- **EAP**:
  - `KEEP_RATIO`: Default fraction of components to keep (0.9)
  - `EAP_ANALYSIS_BATCHES`: Batches for attribution analysis (20)

### File Paths
- `BASELINE_MODEL_NAME`: Default baseline model filename
- `ACDC_MODEL_BASE_NAME`: Base name for ACDC models
- `EAP_MODEL_BASE_NAME`: Base name for EAP models
- `OPTIMIZED_MODEL_DIR`: Directory for optimized models

## Evaluation Metrics

The evaluator provides comprehensive metrics:

1. **Classification Metrics**:
   - Top-1 and Top-5 accuracy
   - Precision, Recall, F1 Score (macro average)
   - Per-class accuracy

2. **Confidence Analysis**:
   - Mean confidence scores
   - Correct vs incorrect prediction confidence
   - Confidence gap analysis
   - Prediction entropy (uncertainty)

3. **Efficiency Metrics**:
   - Inference time (ms/batch)
   - Throughput (batches/second)
   - Parameters count
   - GFLOPs estimation
   - Efficiency score (accuracy per ms)

4. **Pareto Analysis**:
   - Identifies models with optimal accuracy-speed trade-offs
   - Highlights Pareto-optimal models in visualizations

## Output Structure

```
evaluation_results/
├── evaluation_results.png    # Comprehensive visualization with 6 subplots
├── evaluation_metrics.json   # Detailed metrics for all models
└── [console output]          # Detailed report with deployment recommendations
```

## Visualization Outputs

The evaluation generates a comprehensive plot with:
1. **Classification Performance**: Top-1, Top-5 accuracy, and F1 scores
2. **Speed Metrics**: Inference time and throughput comparison
3. **Model Complexity**: Parameters and GFLOPs comparison
4. **Pareto Frontier**: Accuracy vs inference time trade-off
5. **Confidence Analysis**: Model confidence behavior
6. **Efficiency Score**: Normalized efficiency comparison

## Optimization Methods

### ACDC (Automated Circuit DisCovery)
- Uses KL divergence to measure impact of removing components
- Tests removal of entire blocks, attention heads, and MLP neurons
- Preserves components critical for maintaining output distribution

### EAP (Edge Attribution Patching)
- Uses gradient-based attribution scores
- Removes components with lowest importance scores
- Can use either task loss or KL divergence for scoring

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `globals.py`
- Reduce model size parameters (`EMBED_DIM`, `NUM_LAYERS`)

### Dataset Not Found
- Ensure Tiny-ImageNet is downloaded and extracted to `DATA_PATH`
- Check path configuration in `globals.py`

### Models Not Found During Evaluation
- Use `--auto-find` flag to automatically discover models
- Ensure models are saved in the correct directory structure
- Check that model names follow the expected naming convention