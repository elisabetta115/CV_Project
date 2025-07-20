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
HEAD_DIM = 96
NUM_HEADS = 6
EMBED_DIM = HEAD_DIM * NUM_HEADS
NUM_LAYERS = 6
MLP_DIM = 1536
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1   # Label smoothing on loss function
GRAD_CLIP = 1.0

# ACDC optimization parameters
ACDC_THRESHOLD = 0.01  # Threshold for edge importance in ACDC
ACDC_ANALYSIS_BATCHES = 30  # Number of batches for edge importance analysis

# Attribution Patching optimization parameter
KEEP_RATIO = 0.9 # Fraction of components to keep (by attribution score)
EAP_ANALYSIS_BATCHES = 20 # Number of batches for edge importance analysis

# Evaluation parameters
EVAL_NUM_BATCHES = 50  # Number of batches for inference time measurement
WARMUP_ITERATIONS = 5  # Skip first few iterations for warm-up

# File paths
BASELINE_MODEL_NAME = 'baseline_vit.pth'
BASELINE_MODEL_PATH = f'./models/{BASELINE_MODEL_NAME}'
ACDC_MODEL_BASE_NAME = 'acdc_optimized_vit'
EAP_MODEL_BASE_NAME = 'eap_optimized_vit'
OPTIMIZED_MODEL_DIR = './optimized_models/'
EVALUATION_RESULTS_DIR = './evaluation_results'

# Visualization
PLOT_SAVE_PATH = 'vit_results.png'

# Random seed for reproducibility
RANDOM_SEED = 25

# Data normalization parameters (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]