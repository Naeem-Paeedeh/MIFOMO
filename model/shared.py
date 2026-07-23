import math
from typing import Any
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
import einops as eo


# PEFT { --------------------------------------------------------------

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

    
class CoalescentProjectionForHyperSIGMA(nn.Module):
    def __init__(
        self,
        dim_embed_spat: int,
        dim_embed_spec: int,
        num_heads_spat: int,
        num_heads_spec: int,
        num_layers_spat: int,
        num_layers_spec: int,
        shared_across_heads: bool = False,
        DCP: bool = False,
        initialization: InitializationType = InitializationType.AlmostDiagonal,
        std: float = 0.02,
        dtype=torch.float32,
        device='cuda:0'
    ) -> None:
        super().__init__()
        
        self.dim_embed_spat = dim_embed_spat
        self.dim_embed_spec = dim_embed_spec
        self.num_heads_spat = num_heads_spat
        self.num_heads_spec = num_heads_spec
        self.num_layers_spat = num_layers_spat
        self.num_layers_spec = num_layers_spec
        self.shared_across_heads = shared_across_heads
        self.DCP = DCP
        
        self.initialization: InitializationType = initialization
        self.std = std
        self.dtype = dtype
        self.device = device
        
        self.CP_dict = nn.ParameterDict()
        
        self.dim_head_spat = dim_embed_spat // num_heads_spat
        self.dim_head_spec = dim_embed_spec // num_heads_spec
        
        # Spatial
        for layer_number in range(num_layers_spat):
            
            self.CP_dict[f"CPs,QK,spat,{layer_number}"] = self._init_a_CP(
                num_heads=num_heads_spat,
                dim_head=self.dim_head_spat,
            )
            
            if self.DCP:
                self.CP_dict[f"CPs,SV,spat,{layer_number}"] = self._init_a_CP(
                    num_heads=num_heads_spat,
                    dim_head=self.dim_head_spat,
                )
            
        # Spectral
        for layer_number in range(num_layers_spec):
            self.CP_dict[f"CPs,QK,spec,{layer_number}"] = self._init_a_CP(
                num_heads=num_heads_spec,
                dim_head=self.dim_head_spec,
            )
            
            if self.DCP:
                self.CP_dict[f"CPs,SV,spec,{layer_number}"] = self._init_a_CP(
                    num_heads=num_heads_spec,
                    dim_head=self.dim_head_spec,
                )
            
    def _init_a_CP(
        self,
        num_heads: int,
        dim_head: int,
    ) -> nn.Parameter:
        if self.shared_across_heads:
            dimensions=[dim_head, dim_head]
        else:
            dimensions=[num_heads, dim_head, dim_head]
        
        CP: nn.Parameter = initialize_a_tensor(
            dimensions=dimensions,
            std=self.std,
            initialization=self.initialization,
            dtype=self.dtype,
            device=self.device
        )
            
        return CP
    
    def obtain_CP_spat_for_a_layer(
        self,
        layer_number,
    ) -> tuple[nn.Parameter, Any] | tuple[nn.Parameter, None]:
        
        if self.DCP:
            return self.CP_dict[f"CPs,QK,spat,{layer_number}"], self.CP_dict[f"CPs,SV,spat,{layer_number}"]
        else:
            return self.CP_dict[f"CPs,QK,spat,{layer_number}"], None
            
    
    def obtain_CP_spec_for_a_layer(
        self,
        layer_number,
    ) -> tuple[nn.Parameter, Any] | tuple[nn.Parameter, None]:
        if self.DCP:
            return self.CP_dict[f"CPs,QK,spec,{layer_number}"], self.CP_dict[f"CPs,SV,spec,{layer_number}"]
        else:
            return self.CP_dict[f"CPs,QK,spec,{layer_number}"], None
        

class LoRA(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        downsize_dimension: int,
        device
    ):
        super().__init__()
        
        self.dim_embed = dim_embed
        self.down_size = downsize_dimension
        self.device = device

        self.down_proj = nn.Linear(self.dim_embed, self.down_size, bias=False, device=device)      # B
        self.up_proj = nn.Linear(self.down_size, self.dim_embed, bias=False, device=device)        # A

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: T):
        inter_x = self.down_proj.forward(x)
        out = self.up_proj.forward(inter_x)
        return out


class LoRA_QKV(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        downsize_dimension: int,
        qkv_mask: list[bool],
        device
    ):
        super().__init__()
        self.dim_embed = dim_embed
        self.down_size = downsize_dimension
        self.qkv_mask = qkv_mask
        self.device = device
        
        assert len(qkv_mask) == 3
        
        self.adapters_list = nn.ModuleList()
        
        for mask in qkv_mask:
            if mask:
                self.adapters_list.append(LoRA(dim_embed=self.dim_embed, downsize_dimension=self.down_size, device=device))
            else:
                self.adapters_list.append(nn.Identity())
    
        
class LoRAForHyperSIGMA(nn.Module):
    def __init__(
        self,
        dim_embed_spat: int,
        dim_embed_spec: int,
        downsize_dim_spat: int,
        downsize_dim_spec: int,
        qkv_mask: list[bool],
        num_layers_spat: int,
        num_layers_spec: int,
        device
    ) -> None:
        super().__init__()
        
        self.dim_embed_spat = dim_embed_spat
        self.dim_embed_spec = dim_embed_spec
        self.downsize_dim_spat = downsize_dim_spat
        self.downsize_dim_spec = downsize_dim_spec
        self.qkv_mask = qkv_mask
        self.num_layers_spat = num_layers_spat
        self.num_layers_spec = num_layers_spec
        self.device = device
        
        self.lora_dict = nn.ModuleDict()
        
        for layer_number in range(num_layers_spat):
            self.lora_dict[f'lora,spat,{layer_number}'] = LoRA_QKV(
                dim_embed=dim_embed_spat,
                downsize_dimension=downsize_dim_spat,
                qkv_mask=qkv_mask,
                device=device
            )
            
        for layer_number in range(num_layers_spec):
            self.lora_dict[f'lora,spec,{layer_number}'] = LoRA_QKV(
                dim_embed=dim_embed_spec,
                downsize_dimension=downsize_dim_spec,
                qkv_mask=qkv_mask,
                device=device
            )
            
    def obtain_LoRAs_spat_for_a_layer(
        self,
        layer_number,
    ) -> LoRA_QKV:
        return self.lora_dict[f'lora,spat,{layer_number}']
    
    def obtain_LoRAs_spec_for_a_layer(
        self,
        layer_number,
    ) -> LoRA_QKV:
        return self.lora_dict[f'lora,spec,{layer_number}']
            

class Prompt(nn.Module):
    def __init__(
        self,
        num_prompts: int,
        dim_embed: int,
        device
    ) -> None:
        super().__init__()
        
        self.scale = dim_embed ** -0.5
        self.num_prompts = num_prompts
        self.dim_embed = dim_embed
        self.device = device
        
        self.prompt = nn.Parameter(self.scale * torch.randn(num_prompts, dim_embed, device=device))


class DeepPrompts(nn.Module):
    def __init__(
        self,
        num_prompts_spat: int,
        num_prompts_spec: int,
        dim_embed_spat: int,
        dim_embed_spec: int,
        num_layers_spat: int,
        num_layers_spec: int,
        device='cuda:0',
    ) -> None:
        super().__init__()
        
        self.num_prompts_spat = num_prompts_spat
        self.num_prompts_spec = num_prompts_spec
        self.dim_embed_spat = dim_embed_spat
        self.dim_embed_spec = dim_embed_spec
        
        self.num_layers_spat = num_layers_spat
        self.num_layers_spec = num_layers_spec
        self.device = device
        
        self.prompt_dict = nn.ModuleDict()
        
        for layer_number in range(num_layers_spat):
            self.prompt_dict[f'prompt,spat,{layer_number}'] = Prompt(
                num_prompts=num_prompts_spat,
                dim_embed=dim_embed_spat,
                device=device
            )
            
        for layer_number in range(num_layers_spec):
            self.prompt_dict[f'prompt,spec,{layer_number}'] = Prompt(
                num_prompts=num_prompts_spec,
                dim_embed=dim_embed_spec,
                device=device
            )
            
    def obtain_prompts_spat_for_a_layer(
        self,
        layer_number,
    ) -> Prompt:
        return self.prompt_dict[f'prompt,spat,{layer_number}']
    
    def obtain_prompts_spec_for_a_layer(
        self,
        layer_number,
    ) -> Prompt:
        return self.prompt_dict[f'prompt,spec,{layer_number}']
        
# PEFT } --------------------------------------------------------------


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
        self.head_dim = dim // num_heads
        if attn_head_dim is not None:
            self.head_dim = attn_head_dim
        self.all_head_dim = self.head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=qkv_bias)
        # self.window_size = window_size
        # q_size = window_size[0]
        # kv_size = q_size
        # rel_sp_dim = 2 * q_size - 1
        # self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'
        # self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
        CP_QK: T = None,
        CP_SV: T = None,
        lora_qkv: LoRA_QKV = None
    ):
        B, N, D = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv.forward(x)
        
        # qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3，B，H，N，C
        
        if lora_qkv is not None:
            # Before: qkv.shape: [95, 81, 2304]
            # After: qkv.shape:  [3, 95, 12, 81, 64]
            qkv = eo.rearrange(qkv, "b n (c h d) -> c b n (h d)", c=3, h=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # b n (h d)
            
            if lora_qkv.qkv_mask[0]:
                q = q + lora_qkv.adapters_list[0].forward(q)
            if lora_qkv.qkv_mask[1]:
                k = k + lora_qkv.adapters_list[1].forward(k)
            if lora_qkv.qkv_mask[2]:
                v = v + lora_qkv.adapters_list[2].forward(v)
                
            q = eo.rearrange(q, 'b n (h d) ->  b h n d', h=self.num_heads)
            k = eo.rearrange(k, 'b n (h d) ->  b h n d', h=self.num_heads)
            v = eo.rearrange(v, 'b n (h d) ->  b h n d', h=self.num_heads)
        else:       # Without LoRA
            qkv = eo.rearrange(qkv, "b n (c h d) -> c b h n d", c=3, h=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B，H，N，D

        q = q * self.scale
        
        # q.shape = [batch_size, num_heads, seq_length, head_dim]
        
        if CP_QK is not None:
            q = q @ CP_QK
        
        attn = (q @ k.transpose(-2, -1))  # B,H,N,N
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if CP_SV is not None:
            v = v @ CP_SV

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

    def forward(
        self,
        x: T,
        CP_QK: T = None,
        CP_SV: T = None,
        lora_qkv: LoRA_QKV = None
        ):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn.forward(self.norm1(x), CP_QK=CP_QK, CP_SV=CP_SV, lora_qkv=lora_qkv))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn.forward(self.norm1(x), CP_QK=CP_QK, CP_SV=CP_SV, lora_qkv=lora_qkv))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    
# References:
# https://github.com/Naeem-Paeedeh/CVLC