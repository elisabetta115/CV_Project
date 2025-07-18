import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class PrunedMultiHeadAttention(nn.Module):
    """Attention module that skips computation for pruned heads"""
    def __init__(self, original_attn, active_heads):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.active_heads = active_heads  # List of head indices that are active
        self.num_active_heads = len(active_heads)
        
        if self.num_active_heads == 0:
            # If no active heads, this becomes identity
            self.is_identity = True
            return
        
        self.is_identity = False
        self.embed_dim = original_attn.embed_dim
        self.head_dim = original_attn.head_dim
        
        # Create new smaller QKV projection
        old_qkv = original_attn.qkv
        new_qkv_dim = self.num_active_heads * self.head_dim * 3
        self.qkv = nn.Linear(self.embed_dim, new_qkv_dim)
        
        # Copy only weights for active heads
        with torch.no_grad():
            for new_idx, old_idx in enumerate(active_heads):
                # Copy Q weights
                start_old = old_idx * self.head_dim
                end_old = (old_idx + 1) * self.head_dim
                start_new = new_idx * self.head_dim
                end_new = (new_idx + 1) * self.head_dim
                
                # Q weights
                self.qkv.weight.data[start_new:end_new] = old_qkv.weight.data[start_old:end_old]
                # K weights  
                self.qkv.weight.data[new_qkv_dim//3 + start_new:new_qkv_dim//3 + end_new] = \
                    old_qkv.weight.data[self.embed_dim + start_old:self.embed_dim + end_old]
                # V weights
                self.qkv.weight.data[2*new_qkv_dim//3 + start_new:2*new_qkv_dim//3 + end_new] = \
                    old_qkv.weight.data[2*self.embed_dim + start_old:2*self.embed_dim + end_old]
        
        # Output projection - only need size for active heads
        self.proj = nn.Linear(self.num_active_heads * self.head_dim, self.embed_dim)
        # Copy weights accordingly
        with torch.no_grad():
            for new_idx, old_idx in enumerate(active_heads):
                start_old = old_idx * self.head_dim
                end_old = (old_idx + 1) * self.head_dim
                start_new = new_idx * self.head_dim
                end_new = (new_idx + 1) * self.head_dim
                self.proj.weight.data[:, start_new:end_new] = \
                    original_attn.proj.weight.data[:, start_old:end_old]
        
        self.dropout = original_attn.dropout
    
    def forward(self, x):
        if self.is_identity:
            return torch.zeros_like(x)
        
        B, N, C = x.shape
        # Smaller QKV computation - only for active heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_active_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_active_heads * self.head_dim)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    """MLP block"""
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        x = self.fc1(x)      
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class PrunedMLP(nn.Module):
    """MLP module with actual neuron pruning - fixed to not add parameters"""
    def __init__(self, original_mlp, fc1_active_neurons=None, fc2_active_neurons=None):
        super().__init__()
        
        # Get dimensions from original
        embed_dim = original_mlp.fc1.in_features
        mlp_dim = original_mlp.fc1.out_features
        
        # Default to all neurons if not specified
        if fc1_active_neurons is None:
            fc1_active_neurons = list(range(mlp_dim))
        if fc2_active_neurons is None:
            fc2_active_neurons = list(range(embed_dim))
        
        self.fc1_active = fc1_active_neurons
        self.fc2_active = fc2_active_neurons
        self.embed_dim = embed_dim
        
        # Prune INPUT neurons (from fc1), 
        self.fc1 = nn.Linear(embed_dim, len(fc1_active_neurons))
        self.fc2 = nn.Linear(len(fc1_active_neurons), embed_dim)
        self.act = original_mlp.act
        self.dropout = original_mlp.dropout
        
        # Copy only active weights
        with torch.no_grad():
            # FC1: Select active output neurons
            for new_idx, old_idx in enumerate(fc1_active_neurons):
                self.fc1.weight.data[new_idx] = original_mlp.fc1.weight.data[old_idx]
                if self.fc1.bias is not None:
                    self.fc1.bias.data[new_idx] = original_mlp.fc1.bias.data[old_idx]
            
            # FC2: Select active input neurons, but keep all output neurons
            # Zero out the entire FC2
            self.fc2.weight.data.zero_()
            if self.fc2.bias is not None:
                self.fc2.bias.data.zero_()
            
            # Cpy weights for active connections only
            for new_in_idx, old_in_idx in enumerate(fc1_active_neurons):
                # Copy weights for active FC2 output neurons
                for out_idx in fc2_active_neurons:
                    self.fc2.weight.data[out_idx, new_in_idx] = \
                        original_mlp.fc2.weight.data[out_idx, old_in_idx]
            
            # Copy bias only for active output neurons
            if self.fc2.bias is not None:
                for out_idx in fc2_active_neurons:
                    self.fc2.bias.data[out_idx] = original_mlp.fc2.bias.data[out_idx]
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with ACDC optimization support"""
    def __init__(self, image_size=64, patch_size=8, num_classes=200, embed_dim=384, 
                 num_heads=6, num_layers=6, mlp_dim=1536, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        
        return x
    
def reconstruct_pruned_model(base_model, removed_components, config):
    """Reconstruct a pruned model from base model and pruning info"""
    model = base_model  # Work with the provided model
    
    # Group removed components by block and type
    blocks_to_remove = set()
    heads_by_block = {}
    mlp_fc1_by_block = {}
    mlp_fc2_by_block = {}
    
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
    
    # Replace blocks and components
    for block_idx in range(len(model.blocks)):
        if block_idx in blocks_to_remove:
            model.blocks[block_idx] = nn.Identity()
            continue
        
        block = model.blocks[block_idx]
        
        # Handle attention pruning
        if block_idx in heads_by_block:
                removed_heads = heads_by_block[block_idx]
                active_heads = [h for h in range(config['num_heads']) if h not in removed_heads]
                if active_heads:
                    old_attn = block.attn
                    new_attn = PrunedMultiHeadAttention(old_attn, active_heads)
                    block.attn = new_attn

                
        # Handle MLP pruning
        fc1_removed = mlp_fc1_by_block.get(block_idx, [])
        fc2_removed = mlp_fc2_by_block.get(block_idx, [])
        
        if fc1_removed or fc2_removed:
            mlp_dim = config['mlp_dim']
            embed_dim = config['embed_dim']
            
            fc1_active = [i for i in range(mlp_dim) if i not in fc1_removed]
            fc2_active = [i for i in range(embed_dim) if i not in fc2_removed]
            
            if fc1_active:  # Some neurons remain
                old_mlp = block.mlp
                # Invoke the proper initializer, which handles weight copying
                new_mlp = PrunedMLP(old_mlp, fc1_active_neurons=fc1_active, fc2_active_neurons=fc2_active)
                block.mlp = new_mlp

    
    return model
