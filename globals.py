import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset parameters
DATA_PATH = './tiny-imagenet-200'
NUM_CLASSES = 200  # Tiny-ImageNet has 200 classes
IMAGE_SIZE = 64    # Tiny-ImageNet images are 64x64

# Model architecture parameters
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
MLP_DIM = 1536
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4

# ACDC optimization parameters
EDGE_THRESHOLD = 0.01  # Threshold for edge importance in ACDC
ACDC_ANALYSIS_BATCHES = 30  # Number of batches for edge importance analysis
ACDC_FINETUNE_EPOCHS = 20   # Epochs for fine-tuning after ACDC optimization

# Evaluation parameters
EVAL_NUM_BATCHES = 50  # Number of batches for inference time measurement
WARMUP_ITERATIONS = 5  # Skip first few iterations for warm-up

# File paths
BASELINE_MODEL_PATH = 'baseline_vit.pth'
OPTIMIZED_MODEL_PREFIX = 'optimized_vit_'  # Will append threshold value

# Visualization
PLOT_SAVE_PATH = 'vit_results.png'

# Random seed for reproducibility
RANDOM_SEED = 25

# Data normalization parameters (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]