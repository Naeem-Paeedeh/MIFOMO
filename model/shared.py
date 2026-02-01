import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_uniform_
from timm.layers import drop_path, to_2tuple  # , trunc_normal_
from torch.nn.init import trunc_normal_
import numpy as np
from torch import Tensor as T
from new_types import InitializationType


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)
    
    
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# class PatchEmbed_pixels(nn.Module):
#     def __init__(self, patch_size: int):
#         super().__init__()
#         self.patch_size = patch_size
        
#     def forward(self, x: T):
#         pass
    

# class HybridEmbed(nn.Module):
#     """ CNN Feature Map Embedding
#     Extract feature map from CNN, flatten, project to embedding dim.
#     """
#     def __init__(self, backbone, image_size_model_tuple=(224, 224), feature_size=None, in_chans=3, embed_dim=768):
#         super().__init__()
#         assert isinstance(backbone, nn.Module)
#         image_size_model_tuple = to_2tuple(image_size_model_tuple)
#         self.img_size = image_size_model_tuple
#         self.backbone = backbone
        
#         if feature_size is None:
#             with torch.no_grad():
#                 # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
#                 # map for all networks, the feature metadata has reliable channel and stride info, but using
#                 # stride to calc feature dim requires info about padding of each stage that isn't captured.
#                 training = backbone.training
#                 if training:
#                     backbone.eval()
#                 o = self.backbone(torch.zeros(1, in_chans, image_size_model_tuple[0], image_size_model_tuple[1]))[-1]
#                 feature_size = o.shape[-2:]
#                 feature_dim = o.shape[1]
#                 backbone.train(training)
#         else:
#             feature_size = to_2tuple(feature_size)
#             feature_dim = self.backbone.feature_info.channels()[-1]
#         self.num_patches = feature_size[0] * feature_size[1]
#         self.proj = nn.Linear(feature_dim, embed_dim)

#     def forward(self, x):
#         x = self.backbone(x)[-1]
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        # self.window_size = window_size
        # q_size = window_size[0]
        # kv_size = q_size
        # rel_sp_dim = 2 * q_size - 1
        # self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'
        # self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, coalescent_projection: T = None):
        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv.forward(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3，B，H，N，C
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B，H，N，C

        q = q * self.scale
        
        # q.shape = [batch_size, num_heads, seq_length, head_dim]
        
        if coalescent_projection is not None:
            q = q @ coalescent_projection
        
        attn = (q @ k.transpose(-2, -1))  # B,H,N,N
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
             
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: T, coalescent_projection: T = None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn.forward(self.norm1(x), coalescent_projection=coalescent_projection))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn.forward(self.norm1(x), coalescent_projection=coalescent_projection))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def initialize_a_tensor(dimensions: list | tuple, initialization: InitializationType, device, dtype, std: float = 0.02):
    var = torch.zeros(*dimensions, dtype=dtype, device=device)
            
    if initialization == InitializationType.Normal:
        var = trunc_normal_(var, std=std)
    elif initialization == InitializationType.Kaiming:
        var = nn.init.kaiming_normal_(var)
    elif initialization == InitializationType.Uniform:
        var = nn.init.uniform_(var, -0.08, 0.08)
    elif initialization in [InitializationType.AlmostDiagonal, InitializationType.Diagonal]:
        if initialization == InitializationType.AlmostDiagonal:
            var = trunc_normal_(var, std=std)
        
        if len(dimensions) == 2:
            var.fill_diagonal_(1.0)
        elif len(dimensions) == 3:
            for head_id in range(dimensions[0]):
                var[head_id].fill_diagonal_(1)
    else:
        raise NotImplementedError
        
    var = nn.Parameter(var)
        
    return var

    
def obtain_prompts_with_layer_id(prompts_dict: dict, layer_id: int):
    key_coalescent_projection = 'c,' + str(layer_id)
    
    coalescent_projections_current_layer = None
    
    if prompts_dict is not None:
        if key_coalescent_projection in prompts_dict.keys():
            coalescent_projections_current_layer = prompts_dict[key_coalescent_projection]     # [num_heads, dim_head, dim_head]
            
    return coalescent_projections_current_layer
    

class CoalescentProjection(nn.Module):
    def __init__(self,
                 dim_embed: int,
                 num_heads: int,
                 num_layers: int,
                 device=None,
                 dtype=torch.float32,
                 shared_across_heads: bool = False,
                 std: float = 0.02,
                 initialization: InitializationType = InitializationType.AlmostDiagonal,
                 ):
        super().__init__()
        
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.shared_across_heads = shared_across_heads
        self.dtype = dtype
        self.device = device
        self.std = std
        self.initialization = initialization
        
        self.CP_dict = nn.ParameterDict()
        
        for layer_number in range(num_layers):
            dim_head = dim_embed // num_heads
            # dim_embed = num_heads * dim_head
            
            if not shared_across_heads:     # Default
                self.CP_dict['c,' + str(layer_number)] = initialize_a_tensor([num_heads, dim_head, dim_head], std=std, initialization=initialization, dtype=dtype, device=device)
            else:
                self.CP_dict['c,' + str(layer_number)] = initialize_a_tensor([dim_head, dim_head], std=std, initialization=initialization, dtype=dtype, device=device)
            
        assert len(self.CP_dict) > 0
