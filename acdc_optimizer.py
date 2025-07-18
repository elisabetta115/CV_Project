import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import argparse

from globals import *
from models import VisionTransformer, PrunedMultiHeadAttention, PrunedMLP
from data import create_tiny_imagenet_datasets, collate_fn

from utils import set_seed, count_effective_parameters


class ACDCOptimizer:
    """ACDC-based model optimization with practical speedups"""
    def __init__(self, model, dataloader, threshold=0.01, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.threshold = threshold
        self.device = device
        self.reference_outputs = None
        
    def compute_reference_outputs(self, num_batches=20):
        """Compute outputs from the full model for KL divergence calculation"""
        self.model.eval()
        outputs = []
        inputs = []
        with torch.no_grad():
            for i, (images, _) in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                images = images.to(self.device)
                out = self.model(images)
                outputs.append(out)
                inputs.append(images)
        self.reference_outputs = torch.cat(outputs, dim=0)
        self.reference_inputs = torch.cat(inputs, dim=0)
        return self.reference_outputs
    
    def compute_kl_divergence_batch(self, model):
        """Compute KL divergence using cached inputs"""
        model.eval()
        with torch.no_grad():
            current_outputs = model(self.reference_inputs)
        
        # Compute KL divergence
        log_probs_ref = torch.log_softmax(self.reference_outputs, dim=-1)
        log_probs_current = torch.log_softmax(current_outputs, dim=-1)
        kl_div = torch.nn.functional.kl_div(log_probs_current, log_probs_ref.exp(), 
                                            reduction='batchmean', log_target=False)
        return kl_div.item()
    
    def get_component_edges(self):
        """Get edges grouped by components for efficient processing"""
        components = []
        
        # Entire transformer blocks (most coarse-grained)
        for block_idx in range(NUM_LAYERS):
            components.append({
                'name': f'block_{block_idx}',
                'type': 'block',
                'block_idx': block_idx
            })
        
        # Attention heads (medium-grained)
        for block_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                components.append({
                    'name': f'block_{block_idx}_head_{head_idx}',
                    'type': 'attention_head',
                    'block_idx': block_idx,
                    'head_idx': head_idx
                })
        
        # MLP neurons in groups (fine-grained but not individual connections)
        for block_idx in range(NUM_LAYERS):
            # Group FC1 neurons into chunks
            fc1_size = MLP_DIM
            chunk_size = max(32, fc1_size // 20)  # At most 20 chunks per layer
            for chunk_start in range(0, fc1_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, fc1_size)
                components.append({
                    'name': f'block_{block_idx}_fc1_chunk_{chunk_start}_{chunk_end}',
                    'type': 'mlp_fc1_chunk',
                    'block_idx': block_idx,
                    'start': chunk_start,
                    'end': chunk_end
                })
            
            # Group FC2 output neurons
            fc2_size = EMBED_DIM
            chunk_size = max(32, fc2_size // 10)
            for chunk_start in range(0, fc2_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, fc2_size)
                components.append({
                    'name': f'block_{block_idx}_fc2_chunk_{chunk_start}_{chunk_end}',
                    'type': 'mlp_fc2_chunk',
                    'block_idx': block_idx,
                    'start': chunk_start,
                    'end': chunk_end
                })
        
        # Other components
        components.extend([
            {'name': 'cls_token', 'type': 'cls_token'},
            {'name': 'pos_embed', 'type': 'pos_embed'},
        ])
        
        # Reverse order for reverse topological processing
        return components[::-1]
    
    def create_masked_model(self, removed_components):
        """Create a model with components removed (without deep copy for each test)"""
        # Create a registry of masks
        masks = {}
        
        for comp in removed_components:
            if comp['type'] == 'block':
                # Mask entire block
                masks[f"block_{comp['block_idx']}_mask"] = 0.0
            
            elif comp['type'] == 'attention_head':
                # Mask specific attention head
                key = f"block_{comp['block_idx']}_attn_head_{comp['head_idx']}"
                masks[key] = 0.0
            
            elif comp['type'] == 'mlp_fc1_chunk':
                # Mask FC1 neurons
                key = f"block_{comp['block_idx']}_fc1_{comp['start']}_{comp['end']}"
                masks[key] = 0.0
            
            elif comp['type'] == 'mlp_fc2_chunk':
                # Mask FC2 neurons
                key = f"block_{comp['block_idx']}_fc2_{comp['start']}_{comp['end']}"
                masks[key] = 0.0
            
            elif comp['type'] == 'cls_token':
                masks['cls_token'] = 0.0
            
            elif comp['type'] == 'pos_embed':
                masks['pos_embed'] = 0.0
        
        return masks
    
    def apply_masks_to_model(self, model, masks):
        """Apply masks efficiently using hooks instead of modifying weights"""
        hooks = []
        
        def create_attention_mask_hook(head_indices_to_mask):
            def hook(module, input, output):
                B, N, C = input[0].shape
                # Reshape output to separate heads
                output_reshaped = output.view(B, N, NUM_HEADS, C // NUM_HEADS)
                # Zero out masked heads
                for head_idx in head_indices_to_mask:
                    output_reshaped[:, :, head_idx, :] = 0
                return output_reshaped.view(B, N, C)
            return hook
        
        def create_mlp_mask_hook(mask_indices, layer='fc1'):
            def hook(module, input, output):
                output_clone = output.clone()
                output_clone[:, :, mask_indices] = 0
                return output_clone
            return hook
        
        def create_block_mask_hook():
            def hook(module, input, output):
                return input[0]  # Return input unchanged (skip block)
            return hook
        
        # Apply hooks based on masks
        for block_idx in range(NUM_LAYERS):
            block = model.blocks[block_idx]
            
            # Check if entire block is masked
            if f"block_{block_idx}_mask" in masks and masks[f"block_{block_idx}_mask"] == 0.0:
                hook = block.register_forward_hook(create_block_mask_hook())
                hooks.append(hook)
                continue
            
            # Check attention heads
            heads_to_mask = []
            for head_idx in range(NUM_HEADS):
                key = f"block_{block_idx}_attn_head_{head_idx}"
                if key in masks and masks[key] == 0.0:
                    heads_to_mask.append(head_idx)
            
            if heads_to_mask:
                hook = block.attn.proj.register_forward_hook(
                    create_attention_mask_hook(heads_to_mask)
                )
                hooks.append(hook)
            
            # Check MLP chunks
            fc1_indices_to_mask = []
            fc2_indices_to_mask = []
            
            for key, value in masks.items():
                if f"block_{block_idx}_fc1_" in key and value == 0.0:
                    parts = key.split('_')
                    start, end = int(parts[-2]), int(parts[-1])
                    fc1_indices_to_mask.extend(range(start, end))
                
                elif f"block_{block_idx}_fc2_" in key and value == 0.0:
                    parts = key.split('_')
                    start, end = int(parts[-2]), int(parts[-1])
                    fc2_indices_to_mask.extend(range(start, end))
            
            if fc1_indices_to_mask:
                hook = block.mlp.fc1.register_forward_hook(
                    create_mlp_mask_hook(fc1_indices_to_mask, 'fc1')
                )
                hooks.append(hook)
            
            if fc2_indices_to_mask:
                hook = block.mlp.fc2.register_forward_hook(
                    create_mlp_mask_hook(fc2_indices_to_mask, 'fc2')
                )
                hooks.append(hook)
        
        # Handle cls_token and pos_embed with hooks
        if 'cls_token' in masks and masks['cls_token'] == 0.0:
            # Zero out cls token contribution
            original_forward = model.forward
            def forward_no_cls(x):
                # Temporarily set cls_token to zero
                original_cls = model.cls_token.data.clone()
                model.cls_token.data.zero_()
                output = original_forward(x)
                model.cls_token.data = original_cls
                return output
            model.forward = forward_no_cls
        
        return hooks
    
    def apply_acdc_optimization(self, num_batches=20):
        """Apply ACDC algorithm with component-level granularity"""
        # Compute reference outputs
        print("Computing reference outputs...")
        self.compute_reference_outputs(num_batches)
        
        # Get components to test
        components = self.get_component_edges()
        print(f"Testing {len(components)} components...")
        
        # Iterate through components
        removed_components = []
        current_masks = {}
        
        # Use a single model instance with hooks
        test_model = copy.deepcopy(self.model)
        test_model.eval()
        
        for comp in tqdm(components, desc="Processing components"):
            # Test removing this component
            test_masks = current_masks.copy()
            test_masks.update(self.create_masked_model([comp]))
            
            # Apply masks using hooks
            hooks = self.apply_masks_to_model(test_model, test_masks)
            
            # Compute KL divergence
            kl_div = self.compute_kl_divergence_batch(test_model)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Check if KL divergence is below threshold
            if kl_div < self.threshold:
                # Permanently remove component
                current_masks = test_masks
                removed_components.append(comp)
                print(f"Removed {comp['name']}, KL divergence: {kl_div:.6f}")
        
        print(f"Removed {len(removed_components)} components out of {len(components)}")
        
        # Create final model with permanent modifications
        final_model = self.create_final_model(removed_components)
        
        return final_model, removed_components
    
   
    def create_final_model(self, removed_components):
        """Create final model with architectural changes for speed"""
        final_model = copy.deepcopy(self.model)
        
        # Group removed components by block and type
        blocks_to_remove = set()
        heads_by_block = {}
        mlp_fc1_by_block = {}
        mlp_fc2_by_block = {}
        remove_cls_token = False
        remove_pos_embed = False
        
        for comp in removed_components:
            if comp['type'] == 'block':
                blocks_to_remove.add(comp['block_idx'])
            elif comp['type'] == 'attention_head':
                block_idx = comp['block_idx']
                if block_idx not in heads_by_block:
                    heads_by_block[block_idx] = []
                heads_by_block[block_idx].append(comp['head_idx'])
            elif comp['type'] == 'mlp_fc1_chunk':
                block_idx = comp['block_idx']
                if block_idx not in mlp_fc1_by_block:
                    mlp_fc1_by_block[block_idx] = []
                mlp_fc1_by_block[block_idx].extend(range(comp['start'], comp['end']))
            elif comp['type'] == 'mlp_fc2_chunk':
                block_idx = comp['block_idx']
                if block_idx not in mlp_fc2_by_block:
                    mlp_fc2_by_block[block_idx] = []
                mlp_fc2_by_block[block_idx].extend(range(comp['start'], comp['end']))
            elif comp['type'] == 'cls_token':
                remove_cls_token = True
            elif comp['type'] == 'pos_embed':
                remove_pos_embed = True
        
        # Replace entire blocks with Identity
        for block_idx in blocks_to_remove:
            final_model.blocks[block_idx] = nn.Identity()
        
        # Process remaining blocks
        for block_idx in range(len(final_model.blocks)):
            if block_idx in blocks_to_remove:
                continue  # Already handled
            
            block = final_model.blocks[block_idx]
            
            # Replace attention modules with pruned versions
            if block_idx in heads_by_block:
                removed_heads = heads_by_block[block_idx]
                active_heads = [h for h in range(self.model.blocks[0].attn.num_heads) 
                            if h not in removed_heads]
                if active_heads:  # Only if some heads remain
                    old_attn = block.attn
                    block.attn = PrunedMultiHeadAttention(old_attn, active_heads)
                else:
                    # All heads removed - replace with zero output
                    class ZeroAttention(nn.Module):
                        def forward(self, x):
                            return torch.zeros_like(x)
                    block.attn = ZeroAttention()
            
            # Replace MLP modules with pruned versions
            mlp_modified = False
            fc1_removed = mlp_fc1_by_block.get(block_idx, [])
            fc2_removed = mlp_fc2_by_block.get(block_idx, [])
            
            if fc1_removed or fc2_removed:
                old_mlp = block.mlp
                mlp_dim = old_mlp.fc1.out_features
                embed_dim = old_mlp.fc1.in_features
                
                # Determine active neurons
                fc1_active = [i for i in range(mlp_dim) if i not in fc1_removed]
                fc2_active = [i for i in range(embed_dim) if i not in fc2_removed]
                
                if fc1_active and fc2_active:  # Some neurons remain
                    block.mlp = PrunedMLP(old_mlp, fc1_active, fc2_active)
                else:
                    # No active neurons - replace with zero output
                    class ZeroMLP(nn.Module):
                        def forward(self, x):
                            return torch.zeros_like(x)
                    block.mlp = ZeroMLP()
        
        # Handle cls_token removal
        if remove_cls_token:
            # Replace forward to skip cls_token
            original_forward = final_model.forward
            def forward_no_cls(x):
                B = x.shape[0]
                x = final_model.patch_embed(x)
                # Skip cls_token concatenation
                x = x + final_model.pos_embed[:, 1:]  # Only use position embeddings for patches
                x = final_model.dropout(x)
                
                for block in final_model.blocks:
                    x = block(x)
                
                x = final_model.norm(x)
                # Use mean of all patches instead of cls token
                x = x.mean(dim=1)
                x = final_model.head(x)
                return x
            
            final_model.forward = forward_no_cls.__get__(final_model, type(final_model))
        
        # Handle pos_embed removal (rare but possible)
        if remove_pos_embed:
            with torch.no_grad():
                final_model.pos_embed.zero_()
        
        return final_model



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
    
    # Apply ACDC optimization
    print(f"\nApplying ACDC optimization with threshold={args.threshold}...")
    acdc = ACDCOptimizer(model, train_loader, threshold=args.threshold, device=DEVICE)
    
    # Run ACDC algorithm (replaces analyze_edges, create_edge_masks, and apply_masks)
    optimized_model, removed_components = acdc.apply_acdc_optimization(
        num_batches=args.analysis_batches
    )
    
    print(f"\nOriginal parameters: {count_effective_parameters(model):,}")
    print(f"Optimized parameters: {count_effective_parameters(optimized_model):,}")
    
    # Save optimized model
    save_path = f"{OPTIMIZED_MODEL_PREFIX}threshold_{args.threshold}.pth"
    torch.save({
        'model_state_dict': optimized_model.state_dict(),
        'threshold': args.threshold,
        'removed_components': removed_components,
        'config': config,
        'baseline_acc': checkpoint['val_acc'],
    }, save_path)
    
    print(f"\nOptimized model saved to: {save_path}")
    
    return optimized_model, removed_components


def main():
    parser = argparse.ArgumentParser(description='ACDC optimization for Vision Transformer')
    parser.add_argument('--model-path', type=str, default=BASELINE_MODEL_PATH,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--threshold', type=float, default=ACDC_THRESHOLD,
                        help='KL divergence threshold for component removal')
    parser.add_argument('--analysis-batches', type=int, default=ACDC_ANALYSIS_BATCHES,
                        help='Number of batches for KL divergence computation')
    args = parser.parse_args()
    
    # Optimize model
    optimized_model, removed_components = optimize_model(args)
    
    print("\nACDC optimization completed!")


if __name__ == "__main__":
    main()