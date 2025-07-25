import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import sqrt
import copy
import argparse
import numpy as np
import os

from globals import *
from network import VisionTransformer
from data import create_tiny_imagenet_datasets, collate_fn
from utils import set_seed, count_effective_parameters, get_finetuning_args, finetune_pruned_model


class AttributionPatcher:
    """Edge Attribution Patching for model pruning"""
    
    def __init__(self, model, dataloader, device='cuda', use_clean_loss=True):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.use_clean_loss = use_clean_loss  # If True, use task loss; if False, use KL divergence
        
    def compute_attribution_scores(self, num_batches=20):
        """
        Compute EAP scores by patching attention edges and measuring impact.
        """
        self.model.eval()
        attribution_scores = {}
        
        # Get clean outputs for reference
        clean_outputs_list = []
        inputs_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.dataloader, desc="Getting clean outputs")):
                if batch_idx >= num_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                clean_outputs_list.append(outputs)
                inputs_list.append(images)
                labels_list.append(labels)
        
        # Evaluate each attention head using EAP
        for block_idx, block in enumerate(self.model.blocks):
            if isinstance(block, nn.Identity):
                continue
                
            if hasattr(block, 'attn'):
                attn = block.attn
                num_heads = attn.num_heads
                
                for head_idx in range(num_heads):
                    # Patch this specific head
                    head_score = self._evaluate_attention_head_eap(
                        block_idx, head_idx, inputs_list, clean_outputs_list, labels_list
                    )
                    
                    key = f'block_{block_idx}_head_{head_idx}'
                    attribution_scores[key] = head_score
            
            # Evaluate MLP neurons using node patching
            if hasattr(block, 'mlp'):
                mlp = block.mlp
                
                # Evaluate FC1 neurons in chunks
                fc1_dim = mlp.fc1.out_features
                chunk_size = max(32, fc1_dim // 20)
                
                for chunk_start in range(0, fc1_dim, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, fc1_dim)
                    
                    chunk_score = self._evaluate_mlp_neurons_eap(
                        block_idx, 'fc1', chunk_start, chunk_end, 
                        inputs_list, clean_outputs_list, labels_list
                    )
                    
                    key = f'block_{block_idx}_fc1_chunk_{chunk_start}_{chunk_end}'
                    attribution_scores[key] = chunk_score
                
                # Evaluate FC2 output dimensions in chunks
                fc2_dim = mlp.fc2.out_features
                chunk_size = max(32, fc2_dim // 10)
                
                for chunk_start in range(0, fc2_dim, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, fc2_dim)
                    
                    chunk_score = self._evaluate_mlp_neurons_eap(
                        block_idx, 'fc2', chunk_start, chunk_end,
                        inputs_list, clean_outputs_list, labels_list
                    )
                    
                    key = f'block_{block_idx}_fc2_chunk_{chunk_start}_{chunk_end}'
                    attribution_scores[key] = chunk_score
            
            # Evaluate entire block
            block_score = self._evaluate_block_eap(
                block_idx, inputs_list, clean_outputs_list, labels_list
            )
            attribution_scores[f'block_{block_idx}'] = block_score
        
        return attribution_scores
    
    def _evaluate_attention_head_eap(self, block_idx, head_idx, inputs_list, clean_outputs_list, labels_list):
        """Evaluate a single attention head using edge patching."""
        total_impact = 0
        
        # Store the original attention module
        attn_module = self.model.blocks[block_idx].attn
        original_forward = attn_module.forward
        
        def patched_forward(x):
            B, N, C = x.shape
            
            # Run through QKV projection
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(q.size(-1)))
            attn = attn.softmax(dim=-1)
            
            # Corrupt the specific head's attention by setting to uniform
            attn[:, head_idx, :, :] = 1.0 / attn.shape[-1]
            
            # Continue with the rest of the forward pass
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            
            # Apply proj_drop if it exists
            if hasattr(attn_module, 'proj_drop'):
                x = attn_module.proj_drop(x)
            
            return x
        
        # Apply the patch
        attn_module.forward = patched_forward
        
        try:
            with torch.no_grad():
                for images, clean_outputs, labels in zip(inputs_list, clean_outputs_list, labels_list):
                    # Get corrupted outputs
                    corrupted_outputs = self.model(images)
                    
                    if self.use_clean_loss:
                        # Measure impact as increase in classification loss
                        clean_loss = nn.CrossEntropyLoss()(clean_outputs, labels).item()
                        corrupted_loss = nn.CrossEntropyLoss()(corrupted_outputs, labels).item()
                        impact = corrupted_loss - clean_loss
                    else:
                        # Measure impact as KL divergence
                        impact = nn.KLDivLoss(reduction='batchmean')(
                            torch.log_softmax(corrupted_outputs, dim=-1),
                            torch.softmax(clean_outputs, dim=-1)
                        ).item()
                    
                    total_impact += impact
        finally:
            # Restore original forward method
            attn_module.forward = original_forward
        
        return total_impact / len(inputs_list)

    def _evaluate_mlp_neurons_eap(self, block_idx, layer_name, start_idx, end_idx, 
                                inputs_list, clean_outputs_list, labels_list):
        """Evaluate MLP neurons by zeroing their activations."""
        total_impact = 0
        
        def corrupt_neurons_hook(module, input, output):
            # Zero out specified neurons
            corrupted_output = output.clone()
            if len(corrupted_output.shape) == 3:  # (B, N, C)
                corrupted_output[:, :, start_idx:end_idx] = 0
            elif len(corrupted_output.shape) == 2:  # (B, C)
                corrupted_output[:, start_idx:end_idx] = 0
            return corrupted_output
        
        # Register hook on appropriate layer
        if layer_name == 'fc1':
            # Hook after activation function (GELU)
            if hasattr(self.model.blocks[block_idx].mlp, 'activation'):
                target_module = self.model.blocks[block_idx].mlp.activation
            elif hasattr(self.model.blocks[block_idx].mlp, 'act'):
                target_module = self.model.blocks[block_idx].mlp.act
            else:
                # Hook on fc1 output directly
                target_module = self.model.blocks[block_idx].mlp.fc1
        else:  # fc2
            target_module = self.model.blocks[block_idx].mlp.fc2
            
        handle = target_module.register_forward_hook(corrupt_neurons_hook)
        
        try:
            with torch.no_grad():
                for images, clean_outputs, labels in zip(inputs_list, clean_outputs_list, labels_list):
                    corrupted_outputs = self.model(images)
                    
                    if self.use_clean_loss:
                        # Measure impact as increase in classification loss
                        clean_loss = nn.CrossEntropyLoss()(clean_outputs, labels).item()
                        corrupted_loss = nn.CrossEntropyLoss()(corrupted_outputs, labels).item()
                        impact = corrupted_loss - clean_loss
                    else:
                        # Measure impact as KL divergence
                        impact = nn.KLDivLoss(reduction='batchmean')(
                            torch.log_softmax(corrupted_outputs, dim=-1),
                            torch.softmax(clean_outputs, dim=-1)
                        ).item()
                    
                    total_impact += impact
        finally:
            handle.remove()
        
        return total_impact / len(inputs_list)

    def _evaluate_block_eap(self, block_idx, inputs_list, clean_outputs_list, labels_list):
        """Evaluate entire block by replacing with identity."""
        total_impact = 0
        
        # Temporarily replace block with identity
        original_block = self.model.blocks[block_idx]
        self.model.blocks[block_idx] = nn.Identity()
        
        try:
            with torch.no_grad():
                for images, clean_outputs, labels in zip(inputs_list, clean_outputs_list, labels_list):
                    corrupted_outputs = self.model(images)
                    
                    if self.use_clean_loss:
                        # Measure impact as increase in classification loss
                        clean_loss = nn.CrossEntropyLoss()(clean_outputs, labels).item()
                        corrupted_loss = nn.CrossEntropyLoss()(corrupted_outputs, labels).item()
                        impact = corrupted_loss - clean_loss
                    else:
                        # Measure impact as KL divergence
                        impact = nn.KLDivLoss(reduction='batchmean')(
                            torch.log_softmax(corrupted_outputs, dim=-1),
                            torch.softmax(clean_outputs, dim=-1)
                        ).item()
                    
                    total_impact += impact
        finally:
            # Restore original block
            self.model.blocks[block_idx] = original_block
        
        return total_impact / len(inputs_list)
        
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
        from network import reconstruct_pruned_model
        
        # Get model config
        config = {
            'num_heads': self.model.blocks[0].attn.num_heads if hasattr(self.model.blocks[0], 'attn') else NUM_HEADS,
            'mlp_dim': self.model.blocks[0].mlp.fc1.out_features if hasattr(self.model.blocks[0], 'mlp') else MLP_DIM,
            'embed_dim': self.model.blocks[0].mlp.fc1.in_features if hasattr(self.model.blocks[0], 'mlp') else EMBED_DIM,
        }
        
        # Use the reconstruct_pruned_model function from network.py
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
    
    return optimized_model

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
    parser = get_finetuning_args(parser)
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(OPTIMIZED_MODEL_DIR, exist_ok=True)
    
    # Optimize model
    optimized_model = optimize_model_with_attribution(args)

    # Fine-tune if requested
    if args.finetune_epochs > 0:
        print("\n" + "="*60)
        print("Starting fine-tuning phase")
        print("="*60)
        
        # Load datasets
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
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            drop_last=True
        )
        
        # Fine-tune
        optimized_model, finetune_history, best_val_acc = finetune_pruned_model(
            optimized_model, train_loader, val_loader, args, device=DEVICE
        )
        
        # Load original checkpoint for comparison
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        
        # Load the pruned model info
        pruned_path = os.path.join(OPTIMIZED_MODEL_DIR, 
                                   f"{args.output_base_name}_keep_{args.keep_ratio}.pth")
        pruned_checkpoint = torch.load(pruned_path, map_location=DEVICE)
        
        # Save fine-tuned model
        save_filename = f"{args.output_base_name}_keep_{args.keep_ratio}.pth"
        save_path = os.path.join(OPTIMIZED_MODEL_DIR, save_filename)
        
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'keep_ratio': args.keep_ratio,
            'removed_components': pruned_checkpoint['removed_components'],
            'attribution_scores': pruned_checkpoint['attribution_scores'],
            'config': checkpoint['config'],
            'baseline_acc': checkpoint['val_acc'],
            'pruned_acc': best_val_acc,
            'finetune_history': finetune_history,
            'optimization_method': 'eap_finetuned',
            'finetune_epochs': args.finetune_epochs,
            'finetune_lr': args.finetune_lr,
        }, save_path)
        
        print(f"\nFine-tuned model saved to: {save_path}")
        print(f"Baseline accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Fine-tuned accuracy: {best_val_acc:.2f}%")
        print(f"Accuracy change: {best_val_acc - checkpoint['val_acc']:+.2f}%")
    
    print("\nAttribution Patching completed!")


if __name__ == "__main__":
    main()