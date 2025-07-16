import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy
import argparse

from globals import *
from models import VisionTransformer, MultiHeadAttention, MLP
from utils import set_seed, count_parameters, create_tiny_imagenet_datasets, EdgeImportanceTracker
from train_baseline import train_epoch, evaluate


class ACDCOptimizer:
    """ACDC-based model optimization"""
    def __init__(self, model, dataloader, threshold=0.01, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.threshold = threshold
        self.device = device
        self.tracker = EdgeImportanceTracker()
    
    def analyze_edges(self, num_batches=20):
        """Analyze edge importance using forward passes"""
        print("Analyzing edge importance...")
        self.model.eval()
        self.tracker.register_hooks(self.model)
        
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(self.dataloader)):
                if i >= num_batches:
                    break
                images = images.to(self.device)
                _ = self.model(images)
        
        self.tracker.remove_hooks()
        importance_scores = self.tracker.get_edge_importance()
        
        return importance_scores
    
    def create_edge_masks(self, importance_scores):
        """Create masks for pruning unimportant edges"""
        masks = {}
        
        # Get all scores and calculate threshold
        all_scores = list(importance_scores.values())
        if not all_scores:
            print("Warning: No importance scores found")
            return masks
        
        # Use percentile-based threshold
        threshold_value = np.percentile(all_scores, self.threshold * 100)
        
        print(f"Edge importance statistics:")
        print(f"  - Min: {np.min(all_scores):.6f}")
        print(f"  - Max: {np.max(all_scores):.6f}")
        print(f"  - Mean: {np.mean(all_scores):.6f}")
        print(f"  - Threshold (percentile {self.threshold*100}%): {threshold_value:.6f}")
        
        # Create binary masks
        pruned_count = 0
        for name, score in importance_scores.items():
            if score < threshold_value:
                masks[name] = 0.0
                pruned_count += 1
            else:
                masks[name] = 1.0
        
        print(f"Pruning {pruned_count}/{len(importance_scores)} edges ({pruned_count/len(importance_scores)*100:.1f}%)")
        
        return masks
    
    def apply_masks(self, masks):
        """Apply edge masks to model"""
        optimized_model = copy.deepcopy(self.model)
        optimized_model = optimized_model.to(self.device)
        
        # Apply masks to attention and MLP layers
        for name, module in optimized_model.named_modules():
            if isinstance(module, MultiHeadAttention):
                # Create attention mask
                mask_tensor = torch.ones(1, module.num_heads, 1, 1).to(self.device)
                
                # Check if this attention module should be pruned
                for mask_name, importance in masks.items():
                    if name in mask_name and 'attn' in mask_name and importance == 0.0:
                        # Prune specific attention heads based on importance
                        head_idx = hash(mask_name) % module.num_heads
                        mask_tensor[0, head_idx, 0, 0] = 0.0
                
                module.set_edge_mask(mask_tensor)
            
            elif isinstance(module, MLP):
                # Create MLP masks based on importance
                mask1 = torch.ones(1, 1, module.fc1.out_features).to(self.device)
                mask2 = torch.ones(1, 1, module.fc2.out_features).to(self.device)
                
                # Apply pruning based on importance scores
                for mask_name, importance in masks.items():
                    if name in mask_name and importance == 0.0:
                        if 'fc1' in mask_name:
                            # Prune random subset of fc1 neurons
                            prune_indices = np.random.choice(
                                module.fc1.out_features, 
                                size=int(module.fc1.out_features * 0.3), 
                                replace=False
                            )
                            mask1[0, 0, prune_indices] = 0.0
                        elif 'fc2' in mask_name:
                            # Prune random subset of fc2 neurons
                            prune_indices = np.random.choice(
                                module.fc2.out_features, 
                                size=int(module.fc2.out_features * 0.3), 
                                replace=False
                            )
                            mask2[0, 0, prune_indices] = 0.0
                
                module.set_edge_masks(mask1, mask2)
        
        return optimized_model


def finetune_model(model, train_loader, val_loader, epochs, device):
    """Fine-tune the optimized model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_acc = 0
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        history['loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_acc = evaluate(model, val_loader, device)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return model, history, best_val_acc


def optimize_model(args):
    """Main ACDC optimization function"""
    set_seed(RANDOM_SEED)
    
    # Load baseline model
    print(f"Loading baseline model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    
    # Recreate model with saved config
    config = checkpoint['config']
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_tiny_imagenet_datasets(
        DATA_PATH, NORMALIZE_MEAN, NORMALIZE_STD
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # Apply ACDC optimization
    print(f"\nApplying ACDC optimization with threshold={args.threshold}...")
    acdc = ACDCOptimizer(model, train_loader, threshold=args.threshold, device=DEVICE)
    
    # Analyze edge importance
    importance_scores = acdc.analyze_edges(num_batches=args.analysis_batches)
    
    # Create and apply masks
    edge_masks = acdc.create_edge_masks(importance_scores)
    optimized_model = acdc.apply_masks(edge_masks)
    
    print(f"\nOriginal parameters: {count_parameters(model):,}")
    print(f"Optimized parameters: {count_parameters(optimized_model):,}")
    
    # Fine-tune optimized model
    if args.finetune_epochs > 0:
        print(f"\nFine-tuning optimized model for {args.finetune_epochs} epochs...")
        optimized_model, history, best_acc = finetune_model(
            optimized_model, train_loader, val_loader, args.finetune_epochs, DEVICE
        )
        print(f"Fine-tuning completed. Best validation accuracy: {best_acc:.2f}%")
    
    # Save optimized model
    save_path = f"{OPTIMIZED_MODEL_PREFIX}threshold_{args.threshold}.pth"
    torch.save({
        'model_state_dict': optimized_model.state_dict(),
        'threshold': args.threshold,
        'edge_masks': edge_masks,
        'importance_scores': importance_scores,
        'config': config,
        'baseline_acc': checkpoint['val_acc'],
        'optimized_acc': best_acc if args.finetune_epochs > 0 else None
    }, save_path)
    
    print(f"\nOptimized model saved to: {save_path}")
    
    return optimized_model, importance_scores


def main():
    parser = argparse.ArgumentParser(description='ACDC optimization for Vision Transformer')
    parser.add_argument('--model-path', type=str, default=BASELINE_MODEL_PATH,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--threshold', type=float, default=EDGE_THRESHOLD,
                        help='Edge importance threshold for pruning')
    parser.add_argument('--analysis-batches', type=int, default=ACDC_ANALYSIS_BATCHES,
                        help='Number of batches for edge importance analysis')
    parser.add_argument('--finetune-epochs', type=int, default=ACDC_FINETUNE_EPOCHS,
                        help='Number of epochs for fine-tuning (0 to skip)')
    
    args = parser.parse_args()
    
    # Optimize model
    optimized_model, importance_scores = optimize_model(args)
    
    print("\nACDC optimization completed!")


if __name__ == "__main__":
    main()