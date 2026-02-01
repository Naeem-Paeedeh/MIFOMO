# https://github.com/WHU-Sigma/HyperSIGMA   # ss_fusion_seg

# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import warnings
import math
import torch
from typing import Optional, Union, List
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from model.shared import CoalescentProjection
import einops as eo
from timm.layers import drop_path, to_2tuple, trunc_normal_
from mmengine.dist import get_dist_info
from torch.nn.init import constant_, xavier_uniform_
from model.shared import Block, obtain_prompts_with_layer_id
from torch import Tensor as T


# The Spatial branch maps all bands in each pixel and use it as one of the tokens.
# The Spectral branch maps all pixels in each band and use it a one of the tokens.


class PatchEmbed_Pixels(nn.Module):
    def __init__(self, num_bands: int, dim_embed: int):     # patch_size: int, num_bands: int, dim_embed: int
        super().__init__()
        # self.patch_size = patch_size
        self.num_bands = num_bands
        self.dim_embed = dim_embed
        
        self.proj = nn.Linear(num_bands, dim_embed)
        
    def forward(self, x: T):
        assert x.dim() == 4 and x.shape[-1] == x.shape[-2]      # Its shape must be (batch_size, num_bands, patch_size, patch_size)
        # We consider each vector of all elements of all bands for each pixel as a token.
        x = eo.rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj.forward(x)    # b, h * w, num_bands -> [b, h * w, dim_embed]
        
        return x
    
    
class PatchEmbed_Spectral(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = [1, num_bands, p, p]
        x = eo.rearrange(x, 'b c h w -> b (h w) c')
        return x    # x.shape = [1, p*p, num_bands]


# @BACKBONES.register_module()
class SpatialViT(nn.Module):        # Class SpatViT
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size: tuple | list, num_bands=3, dim_embed=768, num_layers=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, use_checkpoint=False,
                 out_indices=[11], pretrained=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.num_features = self.dim_embed = dim_embed  # num_features for consistency with other models
        self.num_bands = num_bands
        self.patch_size = patch_size
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.out_channels = (3, dim_embed, dim_embed, dim_embed, dim_embed)
        
        dim_embed = self.dim_embed
        
        # self.patch_embed = PatchEmbed_Spatial(image_size_model_tuple=patch_size, patch_size=patch_size, in_channels=in_channels, dim_embed=dim_embed)
        
        self.patch_embed = PatchEmbed_Pixels(num_bands=num_bands, dim_embed=dim_embed)

        num_tokens = self.patch_size ** 2

        self.out_indices = out_indices

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim_embed))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            block = Block(
                dim=dim_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            
            self.blocks.append(block)
         
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm = norm_layer(dim_embed)
        
        self.fpn1 = nn.Identity()

        self.fpn2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        # self.fpn2 = nn.Identity()

        self.fpn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # if patch_size == 1:
        #     self.fpn1 = nn.Sequential(
        #         nn.ConvTranspose2d(dim_embed, dim_embed, kernel_size=2, stride=2),
        #     )

        #     self.fpn2 = nn.Identity()

        #     self.fpn3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #     )

        #     self.fpn4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=4, stride=4),
        #     )
        # elif patch_size == 2:
        #     self.fpn1 = nn.Sequential(
        #         nn.ConvTranspose2d(dim_embed, dim_embed, kernel_size=2, stride=2),
        #     )

        #     self.fpn2 = nn.Identity()

        #     self.fpn3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #     )

        #     self.fpn4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=4, stride=4),
        #     )
        # elif patch_size == 4:
        #     self.fpn1 = nn.Identity()

        #     self.fpn2 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=1, stride=1),
        #     )

        #     self.fpn3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #     )

        #     self.fpn4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=4, stride=4),
        #     )
        # elif patch_size == 8:
        #     self.fpn1 = nn.Identity()

        #     self.fpn2 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=1, stride=1),
        #     )

        #     self.fpn3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #     )

        #     self.fpn4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=4, stride=4),
        #     )
        # elif patch_size == 16:
        #     self.fpn1 = nn.Identity()

        #     self.fpn2 = nn.Identity()

        #     self.fpn3 = nn.Identity()

        #     self.fpn4 = nn.Identity()
        # else:
        #     raise NotImplementedError
        
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            # remove patch embed when inchan != 3

            if self.num_bands != 3:
                for k in list(state_dict.keys()):
                    if 'patch_embed.proj' in k:
                        del state_dict[k]

            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = self.patch_embed.patch_shape
                num_patches = self.patch_embed.num_patches
                num_extra_tokens = 0
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    if rank == 0:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

            msg = self.load_state_dict(state_dict, False)
            print(msg)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x: T, coalescent_projection_dict: dict = None):
        (a, c, h, w) = x.shape
        
        # x.shape: [1, 20, 145, 145]
        upsampling = nn.UpsamplingBilinear2d(size=(self.patch_size, self.patch_size))     # size=x.shape[2:4] # Height and width
        
        # x.shape: [1, num_bands, p, p]
        # img = [x]

        # B, C, H, W = x.shape
        
        x = self.patch_embed.forward(x)
        
        batch_size, seq_len, _ = x.size()   # x.shape: [batch_size, 81, 768]

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)        # x.shape: [1, 256, 768]
        features_chosen_layers_list = []
        for i, blk in enumerate(self.blocks):
            
            if i > max(self.out_indices):
                break
            
            coalescent_projections_current_layer = None
            
            if coalescent_projection_dict is not None:
                coalescent_projections_current_layer = obtain_prompts_with_layer_id(coalescent_projection_dict, i)
            
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, coalescent_projections_current_layer, use_reentrant=False)
            else:
                x = blk.forward(x, coalescent_projection=coalescent_projections_current_layer)
                
            # self.out_indices == [3, 5, 7, 11]
            if i in self.out_indices:
                features_chosen_layers_list.append(x)
        
        # The shapes become: [batch_size, dim_embed, patch_size, patch_size]

        # img[0].shape = [batch_size, 20, 9, 9]
        # features[0].shape = [1, 768, 9, 9]  <- x
        # features[1].shape = [1, 768, 9, 9]  <- block 3
        # features[2].shape = [1, 768, 4, 4]  <- block 5
        # features[3].shape = [1, 768, 4, 4]  <- block 7
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        
        for i in range(len(ops)):
            x = features_chosen_layers_list[i]      # The shape of all features were [batch_size, 81, 768]
            x = eo.rearrange(x, 'b (h w) d -> b d h w', h=self.patch_size, w=self.patch_size)
            x = ops[i].forward(x)
            features_chosen_layers_list[i] = upsampling(x)
        
        # The shape four outputs: [batch_size, 768, 9, 9]
        return features_chosen_layers_list


# @BACKBONES.register_module()
class SpectralViT(nn.Module):       # class SpectralVisionTransformer
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size: int, num_tokens_spectral_branch=None, num_bands=3, dim_embed=768, num_layers=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, use_checkpoint=False, out_indices=[11], pretrained=None
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_tokens = num_tokens_spectral_branch        # We adapt the number of tokens for the spectral embedding in spectral branch to this num_tokens with a nn.AdaptiveAvgPool1d.
        self.in_channels = num_bands
        self.use_checkpoint = use_checkpoint
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.patch_embed = PatchEmbed_Spectral()

        self.spec_embed = nn.AdaptiveAvgPool1d(num_tokens_spectral_branch)
        self.spat_map = nn.Linear(int(patch_size ** 2), dim_embed)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_indices = out_indices

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens_spectral_branch, dim_embed))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule

        # MHSA after interval layers
        # WMHSA in other layers
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            block = Block(
                dim=dim_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            
            self.blocks.append(block)
         
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(dim_embed)

        self.l1 = nn.Linear(dim_embed, 128, bias=False)
        # self.l2 = nn.Conv2d(NUM_TOKENS, 128, kernel_size=1, bias=False)
        # self.l3 = nn.Conv2d(NUM_TOKENS, 128, kernel_size=1, bias=False)
        # self.l4 = nn.Conv2d(NUM_TOKENS, 128, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained

        # self.freeze_attn()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            # remove patch embed when inchan != 3

            if self.in_channels != 3:
                for k in list(state_dict.keys()):
                    if 'patch_embed.proj' in k:
                        del state_dict[k]

            if self.patch_size[0] != 64 or self.patch_size[1] != 64:
                for k in list(state_dict.keys()):
                    if 'spat_map' in k:
                        del state_dict[k]

            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                # H, W = self.patch_embed.patch_shape
                # num_patches = self.patch_embed.num_patches
                num_extra_tokens = 1
                # height (== width) for the checkpoint position embedding
                orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
                # height (== width) for the new position embedding
                new_size = int(self.num_tokens)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    # if rank == 0:
                    #     print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, 1, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(self.num_tokens, 1), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

            msg = self.load_state_dict(state_dict, False)
            print(msg)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x: T, coalescent_projection_dict: dict = None):

        # x.shape: [batch_size, num_bands, patch_size, patch_size]
        # B, C, H, W = x.shape
        x = self.patch_embed.forward(x)  # B, N, C

        # x.shape: [batch_size, patch_size * patch_size, num_bands]
        # B, N, C = x.shape

        x = self.spec_embed.forward(x)

        # x.shape: [batch_size, patch_size * patch_size, num_tokens]    # num_tokens_spectral_branch=100
        # _, _, num_tokens = x.shape

        x = x.transpose(1, 2)  # B, N1, Hp*Wp  # x.shape: [batch_size, 81, p*p]
        
        # x_in = x.reshape(B, num_tokens, H, W)
        
        x = self.spat_map.forward(x)  # B, N1, C1  -> [batch_size, num_tokens, dim_embed]     # dim_embed = 768

        # x.shape: [1, num_tokens, dim_embed]  # num_tokens=100, dim_embed=768
        batch_size, _, embed_dim = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)    # x.shape: [1, 100, dim_embed]

        features = []
        
        for i, blk in enumerate(self.blocks):
            
            if i > max(self.out_indices):
                break
            
            coalescent_projections_current_layer = None
            
            if coalescent_projection_dict is not None:
                coalescent_projections_current_layer = obtain_prompts_with_layer_id(coalescent_projection_dict, i)
            
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, coalescent_projections_current_layer, use_reentrant=False)
            else:
                x = blk.forward(x, coalescent_projection=coalescent_projections_current_layer)
                
            if i in self.out_indices:  # self.out_indices = [3]
                features.append(x)  # b, channels, embed_dim

        # features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, H, W), features))

        # features[0].shape: [batch_size, num_tokens, dim_embed]
        ops = [self.l1]
        for i in range(len(ops)):
            features[i] = ops[i](features[i])

        # features[0].shape: [1, 100, 128]
        return features


class HyperSIGMA(torch.nn.Module):       # class SSFusionFramework
    
    def __init__(self,
                 patch_size: tuple | list = None,       # Each sample is a small patch (For instance, 9x9).
                 num_bands=None,
                 num_classes: int = 1,
                 model_size=None,
                 num_tokens_spectral_branch=100,   # encoder
                 use_checkpoint: bool = False,
                 use_classifier_head: bool = True       # To keep the default functionality of the HyperSIGMA.
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        self.center_location = patch_size // 2      # The location of the center pixel in a sample.
        
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.model_size = model_size
        self.num_tokens_spectral_branch = num_tokens_spectral_branch
        self.use_checkpoint = use_checkpoint
        self.use_classifier_head = use_classifier_head
        
        self.chosen_layers_spatial = []
        self.chosen_layers_spectral = []

        if model_size == 'base':
            dim_embed = 768
            self.chosen_layers_spatial = [3, 5, 7, 11]
            self.chosen_layers_spectral = [3]      # Default: [3]
            num_layers = 12
            num_heads = 12
        elif model_size == 'large':
            dim_embed = 1024
            self.chosen_layers_spatial = [7, 11, 15, 23]
            self.chosen_layers_spectral = [7]
            num_layers = 24
            num_heads = 16
        elif model_size == 'huge':
            dim_embed = 1280
            self.chosen_layers_spatial = [10, 15, 20, 31]
            self.chosen_layers_spectral = [10]
            num_layers = 32
            num_heads = 16
        else:
            raise NotImplementedError
        
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        shared_args = dict(
            patch_size=patch_size,
            num_bands=num_bands,
            dim_embed=dim_embed,
            num_heads=num_heads,
            num_layers=num_layers,
            drop_path_rate=0.1,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=use_checkpoint
        )
        
        self.spat_encoder = SpatialViT(
            out_indices=self.chosen_layers_spatial,
            **shared_args,
        )
        
        self.spec_encoder = SpectralViT(
            num_tokens_spectral_branch=num_tokens_spectral_branch,
            out_indices=self.chosen_layers_spectral,
            **shared_args,
        )
        
        # decoder

        # self.spat_encoder.init_weights(r"spat-fina.pth")

        # self.spec_encoder.init_weights(r"spec-base.pth")
        # print('################# Initing pretrained weights for Finetuning! ###################')

        # self.conv_features = nn.Conv2d(self.dim_embed, 128, kernel_size=1, bias=False)
        
        # self.DR1 = nn.Conv2d(self.dim_embed, 128, kernel_size=1, bias=False)
        # self.DR2 = nn.Conv2d(self.dim_embed, 128, kernel_size=1, bias=False)
        # self.DR3 = nn.Conv2d(self.dim_embed, 128, kernel_size=1, bias=False)
        # self.DR4 = nn.Conv2d(self.dim_embed, 128, kernel_size=1, bias=False)
        
        # We use a linear layer for the pixel in the middle.
        self.DR1 = nn.Linear(self.dim_embed, 128, bias=False)
        self.DR2 = nn.Linear(self.dim_embed, 128, bias=False)
        self.DR3 = nn.Linear(self.dim_embed, 128, bias=False)
        self.DR4 = nn.Linear(self.dim_embed, 128, bias=False)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier: nn.Conv2d = None
        self.merger: nn.Conv2d = None
        
        if self.use_classifier_head:
            # We use the center pixel at the end!
            # self.merger = nn.Conv2d(128 * 4, 128, kernel_size=1, stride=1, padding=0, bias=True)
            # self.classifier = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.merger = nn.Linear(128 * 4, 128, bias=True)
            self.classifier = nn.Linear(128, num_classes, bias=True)
        else:
            # self.merger = nn.Conv2d(128 * 4, dim_embed, kernel_size=1, stride=1, padding=0, bias=True)
            self.merger = nn.Linear(128 * 4, dim_embed, bias=True)
            
        self.mlp_spec1 = nn.Sequential(
            nn.Linear(num_tokens_spectral_branch, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.dim_embed, bias=False),
            nn.Sigmoid(),
        )
        
        self.mlp_spec2 = nn.Sequential(
            nn.Linear(num_tokens_spectral_branch, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.dim_embed, bias=False),
            nn.Sigmoid(),
        )
        
        self.mlp_spec3 = nn.Sequential(
            nn.Linear(num_tokens_spectral_branch, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.dim_embed, bias=False),
            nn.Sigmoid(),
        )
        
        self.mlp_spec4 = nn.Sequential(
            nn.Linear(num_tokens_spectral_branch, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.dim_embed, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x: T, coalescent_projection_spat: CoalescentProjection = None, coalescent_projection_spec: CoalescentProjection = None, use_spat_encoder: bool = True, use_spec_encoder: bool = True):
        # x.shape: [batch_size, 20, 9, 9]
        
        assert use_spat_encoder or use_spec_encoder
        
        CP_dict_spatial = None
        CP_dict_spectral = None
        
        if coalescent_projection_spat is not None:
            CP_dict_spatial = coalescent_projection_spat.CP_dict
        
        if coalescent_projection_spec is not None:
            CP_dict_spectral = coalescent_projection_spec.CP_dict

        # x: (b, c, h, w)
        # ts:(b, c)
        
        batch_size, num_bands, h, w = x.shape

        if use_spat_encoder:
            # Their shape becomes: [1, 768, 9, 9]
            features_spatial_chosen_layers_list = self.spat_encoder.forward(x, coalescent_projection_dict=CP_dict_spatial)

        spec_weights_list = []
        
        if use_spec_encoder:
            spec_feature = self.spec_encoder.forward(x, coalescent_projection_dict=CP_dict_spectral)
        
            spec_feature = spec_feature[0]      # Its shape: [batch_size, num_tokens_spectral, 128]

            spec_feature = self.pool(spec_feature)  # [batch_size, num_tokens_spectral, 1]
            spec_feature = spec_feature.view(batch_size, -1)  # b, c  -> [1, 100]

            spec_weights_list.append(self.mlp_spec1(spec_feature).view(batch_size, -1, 1, 1))   # Its shape: [batch_size, 768, 1, ,1]
            spec_weights_list.append(self.mlp_spec2(spec_feature).view(batch_size, -1, 1, 1))   # Its shape: [batch_size, 768, 1, ,1]
            spec_weights_list.append(self.mlp_spec3(spec_feature).view(batch_size, -1, 1, 1))   # Its shape: [batch_size, 768, 1, ,1]
            spec_weights_list.append(self.mlp_spec4(spec_feature).view(batch_size, -1, 1, 1))   # Its shape: [batch_size, 768, 1, ,1]
        
        features_fused_list = []
        
        if use_spat_encoder:
            for i in range(len(self.chosen_layers_spatial)):
                x = features_spatial_chosen_layers_list[i]
                if use_spec_encoder:
                    x = x * (1 + spec_weights_list[i])  # [batch_size, dim_embed, 1, ,1] * [1, dim_embed, patch_size, patch_size] -> [batch_size, dim_embed, patch_size, patch_size]  # dim_embed = 768
                    
                x = eo.rearrange(x, 'b d h w -> b h w d')       # [batch_size, patch_size, patch_size, dim_embed]  # dim_embed = 768
                
                features_fused_list.append(x)
        
        # Their shape was [batch_size, patch_size, patch_size, dim_embed]  # dim_embed = 768
        # Their shape become [batch_size, patch_size, patch_size, 128]
        features_fused_list[0] = self.DR1.forward(features_fused_list[0])
        features_fused_list[1] = self.DR1.forward(features_fused_list[1])
        features_fused_list[2] = self.DR1.forward(features_fused_list[2])
        features_fused_list[3] = self.DR1.forward(features_fused_list[3])

        ss_feature = torch.concat(features_fused_list, -1)  # Its shape becomes: [batch_size, patch_size, patch_size, 4 * 128]
        
        # Some operations might be redundant! }
        output = self.merger.forward(ss_feature)
        
        # Classification convolution
        if self.use_classifier_head:
            # ss_feature shape was: [batch_size, patch_size, patch_size, 128]
            output = self.classifier.forward(ss_feature)   # Its shape becomes: [batch_size, patch_size, patch_size, num_classes]
            # output = eo.rearrange(output, 'b h w c -> b (w h) c')   # output.shape = [batch_size, patch_size ** 2, num_classes]

            return output[:, self.center_location, self.center_location, :]   # Its shape: [batch_size, num_classes]
        else:
            # ss_feature shape was: [batch_size, patch_size, patch_size, dim_embed]
            return output[:, self.center_location, self.center_location, :]   # Its shape: [batch_size, dim_embed]
