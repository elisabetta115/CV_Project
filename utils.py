import torch
import numpy as np
import os
from torchvision import transforms, datasets
from collections import defaultdict
import time

def set_seed(seed = 25):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_tiny_imagenet_datasets(data_path, normalize_mean, normalize_std):
    """Create Tiny-ImageNet train and validation datasets"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    try:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'train'), 
            transform=transform_train
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'val'), 
            transform=transform_val
        )
    except:
        raise FileNotFoundError(f"Tiny-ImageNet dataset not found at {data_path}")
    
    return train_dataset, val_dataset


def measure_inference_time(model, dataloader, device, num_batches=50, warmup=5):
    """Measure average inference time per batch"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
    
    # Skip warmup iterations
    return np.mean(times[warmup:]) if len(times) > warmup else np.mean(times)


class EdgeImportanceTracker:
    """Track importance of edges in the computational graph"""
    def __init__(self):
        self.edge_scores = defaultdict(list)
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks to track activations"""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Calculate importance as mean absolute activation
                    importance = output.abs().mean().item()
                    self.edge_scores[name].append(importance)
            return hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_edge_importance(self):
        """Calculate average importance for each edge"""
        importance_dict = {}
        for name, scores in self.edge_scores.items():
            importance_dict[name] = np.mean(scores)
        return importance_dict