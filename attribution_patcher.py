import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import argparse
import numpy as np
import os

from globals import *
from models import VisionTransformer
from data import create_tiny_imagenet_datasets, collate_fn
from utils import set_seed, count_effective_parameters


class AttributionPatcher:
    """Attribution Patching for efficient model pruning using gradient-based importance scores"""
    
    def __init__(self, model, dataloader, device='cuda', use_clean_loss=True):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.use_clean_loss = use_clean_loss  # If True, use task loss; if False, use KL divergence
        
    def compute_attribution_scores(self, num_batches=20):
        """
        Compute attribution scores for all model components using gradients.
        """
        self.model.eval()
        
        # Dictionary to store attribution scores for each component
        attribution_scores = {}
        
        # Process batches
        all_losses = []
        for batch_idx, (images, labels) in enumerate(tqdm(self.dataloader, desc="Computing attributions")):
            if batch_idx >= num_batches:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Enable gradients for this forward pass
            self.model.zero_grad()
            
            # Forward pass with gradient tracking
            images.requires_grad = True
            outputs = self.model(images)
            
            if self.use_clean_loss:
                # Use task-specific loss (classification)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                # Use KL divergence from reference outputs
                with torch.no_grad():
                    reference_outputs = outputs.detach()
                outputs_2 = self.model(images)
                loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(outputs_2, dim=-1),
                    torch.softmax(reference_outputs, dim=-1)
                )
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate gradient magnitudes for each component
            with torch.no_grad():
                # Attention head scores
                for block_idx, block in enumerate(self.model.blocks):
                    if isinstance(block, nn.Identity):
                        continue
                        
                    # Score attention heads by gradient magnitude of QKV projections
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        if hasattr(attn.qkv, 'weight') and attn.qkv.weight.grad is not None:
                            grad = attn.qkv.weight.grad
                            num_heads = attn.num_heads
                            head_dim = attn.embed_dim // num_heads
                            
                            # Split gradients by head
                            for head_idx in range(num_heads):
                                q_start = head_idx * head_dim
                                q_end = (head_idx + 1) * head_dim
                                k_start = attn.embed_dim + head_idx * head_dim
                                k_end = attn.embed_dim + (head_idx + 1) * head_dim
                                v_start = 2 * attn.embed_dim + head_idx * head_dim
                                v_end = 2 * attn.embed_dim + (head_idx + 1) * head_dim
                                
                                # Compute attribution score as L2 norm of gradients
                                head_grad = torch.cat([
                                    grad[q_start:q_end].flatten(),
                                    grad[k_start:k_end].flatten(),
                                    grad[v_start:v_end].flatten()
                                ])
                                
                                key = f'block_{block_idx}_head_{head_idx}'
                                if key not in attribution_scores:
                                    attribution_scores[key] = 0
                                attribution_scores[key] += head_grad.norm().item()
                    
                    # Score MLP neurons
                    if hasattr(block, 'mlp'):
                        mlp = block.mlp
                        
                        # FC1 neuron scores
                        if hasattr(mlp.fc1, 'weight') and mlp.fc1.weight.grad is not None:
                            fc1_grad = mlp.fc1.weight.grad
                            # Group into chunks for efficiency
                            chunk_size = max(32, fc1_grad.shape[0] // 20)
                            for chunk_start in range(0, fc1_grad.shape[0], chunk_size):
                                chunk_end = min(chunk_start + chunk_size, fc1_grad.shape[0])
                                chunk_grad = fc1_grad[chunk_start:chunk_end]
                                
                                key = f'block_{block_idx}_fc1_chunk_{chunk_start}_{chunk_end}'
                                if key not in attribution_scores:
                                    attribution_scores[key] = 0
                                attribution_scores[key] += chunk_grad.norm().item()
                        
                        # FC2 neuron scores
                        if hasattr(mlp.fc2, 'weight') and mlp.fc2.weight.grad is not None:
                            fc2_grad = mlp.fc2.weight.grad
                            # Group into chunks
                            chunk_size = max(32, fc2_grad.shape[0] // 10)
                            for chunk_start in range(0, fc2_grad.shape[0], chunk_size):
                                chunk_end = min(chunk_start + chunk_size, fc2_grad.shape[0])
                                # For FC2, we look at gradients of weights going TO these output neurons
                                chunk_grad = fc2_grad[chunk_start:chunk_end, :]
                                
                                key = f'block_{block_idx}_fc2_chunk_{chunk_start}_{chunk_end}'
                                if key not in attribution_scores:
                                    attribution_scores[key] = 0
                                attribution_scores[key] += chunk_grad.norm().item()
                    
                    # Score entire blocks
                    block_score = 0
                    for name, param in block.named_parameters():
                        if param.grad is not None:
                            block_score += param.grad.norm().item()
                    
                    key = f'block_{block_idx}'
                    if key not in attribution_scores:
                        attribution_scores[key] = 0
                    attribution_scores[key] += block_score
                
                # Score cls_token and pos_embed
                if self.model.cls_token.grad is not None:
                    if 'cls_token' not in attribution_scores:
                        attribution_scores['cls_token'] = 0
                    attribution_scores['cls_token'] += self.model.cls_token.grad.norm().item()
                
                if self.model.pos_embed.grad is not None:
                    if 'pos_embed' not in attribution_scores:
                        attribution_scores['pos_embed'] = 0
                    attribution_scores['pos_embed'] += self.model.pos_embed.grad.norm().item()
            
            all_losses.append(loss.item())
            
        # Average scores across batches
        num_batches_processed = min(num_batches, batch_idx + 1)
        for key in attribution_scores:
            attribution_scores[key] /= num_batches_processed
            
        print(f"Average loss: {np.mean(all_losses):.4f}")
        
        return attribution_scores
    
    def select_components_to_remove(self, attribution_scores, keep_ratio=0.9):
        """
        Select components to remove based on attribution scores.
        Keeps the top keep_ratio fraction of components by attribution score.
        """
        # Convert to list of (component_info, score) tuples
        components_with_scores = []
        
        for key, score in attribution_scores.items():
            parts = key.split('_')
            
            if key.startswith('block_') and 'head' not in key and 'fc' not in key:
                # Entire block
                comp_info = {
                    'name': key,
                    'type': 'block',
                    'block_idx': int(parts[1])
                }
            elif 'head' in key:
                # Attention head
                comp_info = {
                    'name': key,
                    'type': 'attention_head',
                    'block_idx': int(parts[1]),
                    'head_idx': int(parts[3])
                }
            elif 'fc1_chunk' in key:
                # FC1 chunk
                comp_info = {
                    'name': key,
                    'type': 'mlp_fc1_chunk',
                    'block_idx': int(parts[1]),
                    'start': int(parts[4]),
                    'end': int(parts[5])
                }
            elif 'fc2_chunk' in key:
                # FC2 chunk
                comp_info = {
                    'name': key,
                    'type': 'mlp_fc2_chunk',
                    'block_idx': int(parts[1]),
                    'start': int(parts[4]),
                    'end': int(parts[5])
                }
            elif key == 'cls_token':
                comp_info = {
                    'name': key,
                    'type': 'cls_token'
                }
            elif key == 'pos_embed':
                comp_info = {
                    'name': key,
                    'type': 'pos_embed'
                }
            else:
                continue
                
            components_with_scores.append((comp_info, score))
        
        # Sort by score (ascending, so lowest scores are first)
        components_with_scores.sort(key=lambda x: x[1])
        
        # Calculate how many components to remove
        num_components = len(components_with_scores)
        num_to_keep = int(num_components * keep_ratio)
        num_to_remove = num_components - num_to_keep
        
        # Select components to remove (those with lowest scores)
        components_to_remove = [comp_info for comp_info, score in components_with_scores[:num_to_remove]]
        
        print(f"\nAttribution score statistics:")
        scores = [score for _, score in components_with_scores]
        print(f"  Min score: {min(scores):.6f}")
        print(f"  Max score: {max(scores):.6f}")
        print(f"  Mean score: {np.mean(scores):.6f}")
        print(f"  Median score: {np.median(scores):.6f}")
        print(f"\nRemoving {num_to_remove} components with lowest scores (keeping top {keep_ratio*100:.0f}%)")
        
        return components_to_remove
    
    def create_pruned_model(self, components_to_remove):
        """
        Create the final pruned model by removing low-attribution components.
        """
        from models import reconstruct_pruned_model
        
        # Get model config
        config = {
            'num_heads': self.model.blocks[0].attn.num_heads if hasattr(self.model.blocks[0], 'attn') else NUM_HEADS,
            'mlp_dim': self.model.blocks[0].mlp.fc1.out_features if hasattr(self.model.blocks[0], 'mlp') else MLP_DIM,
            'embed_dim': self.model.blocks[0].mlp.fc1.in_features if hasattr(self.model.blocks[0], 'mlp') else EMBED_DIM,
        }
        
        # Use the reconstruct_pruned_model function from models.py
        pruned_model = reconstruct_pruned_model(
            copy.deepcopy(self.model), 
            components_to_remove, 
            config
        )
        
        return pruned_model


def optimize_model_with_attribution(args):
    """Main attribution patching optimization function"""
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
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Apply Attribution Patching
    print(f"\nApplying Attribution Patching with keep_ratio={args.keep_ratio}...")
    patcher = AttributionPatcher(model, train_loader, device=DEVICE, use_clean_loss=args.use_task_loss)
    
    # Compute attribution scores
    attribution_scores = patcher.compute_attribution_scores(num_batches=args.analysis_batches)
    
    # Select components to remove
    components_to_remove = patcher.select_components_to_remove(
        attribution_scores, 
        keep_ratio=args.keep_ratio
    )
    
    # Create pruned model
    optimized_model = patcher.create_pruned_model(components_to_remove)
    
    print(f"\nOriginal parameters: {count_effective_parameters(model):,}")
    print(f"Optimized parameters: {count_effective_parameters(optimized_model):,}")
    print(f"Parameter reduction: {(1 - count_effective_parameters(optimized_model)/count_effective_parameters(model))*100:.1f}%")
    
     # Save optimized model
    save_filename = f"{args.output_base_name}_keep_{args.keep_ratio}.pth"
    save_path = os.path.join(OPTIMIZED_MODEL_DIR, save_filename)
    
    torch.save({
        'model_state_dict': optimized_model.state_dict(),
        'keep_ratio': args.keep_ratio,
        'removed_components': components_to_remove,
        'attribution_scores': attribution_scores,
        'config': config,
        'baseline_acc': checkpoint['val_acc'],
        'optimization_method': 'eap',
    }, save_path)
    
    print(f"\nOptimized model saved to: {save_path}")
    
    return optimize

def main():
    parser = argparse.ArgumentParser(description='Attribution Patching for Vision Transformer')
    parser.add_argument('--model-path', type=str, default=BASELINE_MODEL_PATH,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--keep-ratio', type=float, default=KEEP_RATIO,
                        help='Fraction of components to keep (by attribution score)')
    parser.add_argument('--analysis-batches', type=int, default=EAP_ANALYSIS_BATCHES,
                        help='Number of batches for attribution computation')
    parser.add_argument('--use-task-loss', action='store_true',
                        help='Use task-specific loss instead of KL divergence')
    parser.add_argument('--output-base-name', type=str, default=EAP_MODEL_BASE_NAME,
                        help='Base name for output model file')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(OPTIMIZED_MODEL_DIR, exist_ok=True)
    
    # Optimize model
    optimized_model, removed_components = optimize_model_with_attribution(args)
    
    print("\nAttribution Patching completed!")


if __name__ == "__main__":
    main()