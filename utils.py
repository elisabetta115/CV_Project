import torch
import numpy as np
import time

def set_seed(seed = 25):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_effective_parameters(model):
    """Count effective parameters in model after structural pruning"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
    return total

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