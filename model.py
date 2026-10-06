import copy
import math
import os
from monai.transforms import concatenate
from torch.nn.utils import spectral_norm
from itertools import permutations
from typing import List, Tuple, Optional
from torch.autograd import Function
from torch.cuda import graph
from torch.utils import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision
import torch.nn as nn
import torch
from dependency import *
from utils import get_parameter_number, count_parameters_flops
import torch.nn.functional as F
from knowledage_graph import *
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import torch_geometric
from loss import *
from torchvision.ops import DeformConv2d
from torch.utils import checkpoint
from safetensors.torch import load_file
from torch.cuda.amp import autocast
import timm
from timm.models.hub import load_model_config_from_hf
# from LSNet import  *


sigmoid = nn.Sigmoid()

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
# from timm.models.layers import CondConv2d

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class SpatialCrossAttentionFA2(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout_p=0.1, ffn_expansion=4, apply_softmax_early=False):
        super().__init__()
        self.heads = num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.apply_softmax_early = apply_softmax_early
        self.in_channels = in_channels

        # QKV projections - using 1x1 convs is natural for spatial features,
        # or nn.Linear after flattening
        # Option 1: Using nn.Linear (requires flattening/unflattening)
        self.q_proj1 = nn.Linear(in_channels, in_channels)
        self.k_proj1 = nn.Linear(in_channels, in_channels)
        self.v_proj1 = nn.Linear(in_channels, in_channels)

        self.q_proj2 = nn.Linear(in_channels, in_channels)
        self.k_proj2 = nn.Linear(in_channels, in_channels)
        self.v_proj2 = nn.Linear(in_channels, in_channels)

        # Output projection for attention part
        # self.attn_out_proj_c = nn.Linear(in_channels, in_channels)
        # self.attn_out_proj_d = nn.Linear(in_channels, in_channels)

        # LayerNorms - applied on the channel dimension after flattening
        self.norm1_c = nn.LayerNorm(in_channels)
        self.norm1_d = nn.LayerNorm(in_channels)
        # self.norm2_c = nn.LayerNorm(in_channels)
        # self.norm2_d = nn.LayerNorm(in_channels)

        # FFN
        # self.ffn_c = nn.Sequential(
        #     nn.Linear(in_channels, in_channels * ffn_expansion),
        #     nn.GELU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(in_channels * ffn_expansion, in_channels)
        # )
        # self.ffn_d = nn.Sequential(
        #     nn.Linear(in_channels, in_channels * ffn_expansion),
        #     nn.GELU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(in_channels * ffn_expansion, in_channels)
        # )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, c_feat, d_feat):  # c_feat, d_feat: (B, C, H, W)
        B, C, H, W = c_feat.shape
        N = H * W  # Number of spatial tokens

        # --- Process c_feat ---
        res_c = c_feat
        c_flat = c_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        c_flat_norm = self.norm1_c(c_flat)

        q1 = self.q_proj1(c_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1,
                                                                                     3)  # (B, H_heads, N, D_head)
        k1 = self.k_proj1(c_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1,
                                                                                     3)  # (B, H_heads, N, D_head)
        v1 = self.v_proj1(c_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1,
                                                                                     3)  # (B, H_heads, N, D_head)

        attn_c_raw = (q1 @ k1.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N) Self-attention over spatial tokens

        # --- Process d_feat ---
        res_d = d_feat
        d_flat = d_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        d_flat_norm = self.norm1_d(d_flat)

        q2 = self.q_proj2(d_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k2 = self.k_proj2(d_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v2 = self.v_proj2(d_flat_norm).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_d_raw = (q2 @ k2.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N)

        # --- FA² style combination ---
        if self.apply_softmax_early:
            attn_c_eff = torch.softmax(attn_c_raw, dim=-1)
            attn_d_eff = torch.softmax(attn_d_raw, dim=-1)
        else:
            attn_c_eff = attn_c_raw
            attn_d_eff = attn_d_raw

        m_c = attn_d_eff @ attn_c_eff.transpose(-2, -1)  # (B,H,N,N) @ (B,H,N,N) -> (B,H,N,N)
        m_d = attn_c_eff @ attn_d_eff.transpose(-2, -1)

        if not self.apply_softmax_early:
            m_c = torch.softmax(m_c, dim=-1)
            m_d = torch.softmax(m_d, dim=-1)

        # Weighted sum of values
        out_c_attn_flat = (m_c @ v1).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        out_d_attn_flat = (m_d @ v2).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        # Output projection from attention
        # out_c_attn_flat = self.attn_out_proj_c(out_c_attn_flat)
        # out_d_attn_flat = self.attn_out_proj_d(out_d_attn_flat)

        # First residual (on flattened features)
        c_flat = c_flat + self.dropout(out_c_attn_flat)
        d_flat = d_flat + self.dropout(out_d_attn_flat)

        # c_feat_ffn_input = self.norm2_c(c_flat)
        # d_feat_ffn_input = self.norm2_d(d_flat)
        # FFN
        # out_c_ffn_flat = self.ffn_c(c_feat_ffn_input)
        # out_d_ffn_flat = self.ffn_d(d_feat_ffn_input)
        # # Second residual
        # out_c_flat_final = c_flat + self.dropout(out_c_ffn_flat)
        # out_d_flat_final = d_flat + self.dropout(out_d_ffn_flat)

        # Reshape back to (B, C, H, W)
        out_c = c_flat.transpose(1, 2).view(B, C, H, W)
        out_d = d_flat.transpose(1, 2).view(B, C, H, W)

        return out_c, out_d


class CrossAttentionFA2_Conv(nn.Module):
    def __init__(self, in_channels, num_heads=8, reduction_factor=8, dropout_p=0.1, ffn_expansion=4,
                 apply_softmax_early=False):
        super().__init__()

        assert in_channels % reduction_factor == 0, "in_channels must be divisible by reduction_factor"
        self.projected_dim_kqv = in_channels // reduction_factor

        assert self.projected_dim_kqv % num_heads == 0, \
            "Projected KQV dimension after reduction must be divisible by num_heads. " \
            f"Projected_dim_kqv: {self.projected_dim_kqv}, num_heads: {num_heads}"

        self.heads = num_heads
        self.head_dim_kqv = self.projected_dim_kqv // num_heads
        self.scale = self.head_dim_kqv ** -0.5 + 1e-6
        self.apply_softmax_early = apply_softmax_early
        self.in_channels = in_channels

        self.q_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.k_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.v_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)

        self.q_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.k_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.v_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)

        # self.attn_out_proj_c = nn.Linear(self.projected_dim_kqv, in_channels)
        # self.attn_out_proj_d = nn.Linear(self.projected_dim_kqv, in_channels)
        self.attn_out_proj_c = nn.Sequential(
            nn.Conv2d(self.projected_dim_kqv, in_channels, kernel_size=1, bias=False),
        )
        self.attn_out_proj_d = nn.Sequential(
            nn.Conv2d(self.projected_dim_kqv, in_channels, kernel_size=1, bias=False),
        )

        # LayerNorms - applied on the channel dimension of the flattened features (B, N, C)
        # self.norm1_c = nn.LayerNorm(in_channels)
        # self.norm1_d = nn.LayerNorm(in_channels)
        # self.norm2_c = nn.LayerNorm(in_channels)  # For FFN input
        # self.norm2_d = nn.LayerNorm(in_channels)  # For FFN input
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout_p)

    def _project_and_split_heads_spatial(self, x_norm_flat, q_conv, k_conv, v_conv, B, N):
        # x_norm_flat: (B, N, C_in)
        x_permuted = x_norm_flat.transpose(1, 2)  # (B, C_in, N) for Conv1D

        q_projected = q_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)
        k_projected = k_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)
        v_projected = v_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)

        # Reshape for multi-head attention: (B, N, num_heads, head_dim_kqv) -> (B, num_heads, N, head_dim_kqv)
        q = q_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        k = k_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        v = v_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        return q, k, v

    def forward(self, c_feat_spatial, d_feat_spatial):  # Inputs: (B, C, H, W)
        B, C, H, W = c_feat_spatial.shape
        N = H * W  # Number of spatial tokens
        assert C == self.in_channels

        # --- Flatten and Norm c_feat ---
        c_flat = c_feat_spatial.flatten(2).transpose(1, 2)  # (B, N, C)
        # c_flat_norm = self.norm1_c(c_flat)
        q1, k1, v1 = self._project_and_split_heads_spatial(c_flat, self.q_proj1, self.k_proj1, self.v_proj1, B, N)

        # --- Flatten and Norm d_feat ---
        d_flat = d_feat_spatial.flatten(2).transpose(1, 2)  # (B, N, C)
        # d_flat_norm = self.norm1_d(d_flat)
        q2, k2, v2 = self._project_and_split_heads_spatial(d_flat, self.q_proj2, self.k_proj2, self.v_proj2, B, N)

        v_dtype = v1.dtype

        with autocast(enabled=False):
            # 将输入手动转为 float32
            q1, k1, v1 = q1.float(), k1.float(), v1.float()
            q2, k2, v2 = q2.float(), k2.float(), v2.float()
            attn_c_raw = (q1 @ k1.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N)
            attn_d_raw = (q2 @ k2.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N)

            attn_c_raw.clamp_(-30, 30)  # 30 已经是一个很大的数了，exp(30) 很大
            attn_d_raw.clamp_(-30, 30)

            if self.apply_softmax_early:
                attn_c_eff = torch.softmax(attn_c_raw, dim=-1)
                attn_d_eff = torch.softmax(attn_d_raw, dim=-1)
            else:
                attn_c_eff = attn_c_raw
                attn_d_eff = attn_d_raw

            m_c = attn_d_eff @ attn_c_eff.transpose(-2, -1)  # (B,H,N,N)
            m_d = attn_c_eff @ attn_d_eff.transpose(-2, -1)  # (B,H,N,N)

            if not self.apply_softmax_early:
                m_c = torch.softmax(m_c, dim=-1)
                m_d = torch.softmax(m_d, dim=-1)

            # Result is (B, num_heads, N, self.head_dim_kqv)
            out_c_attn_heads = m_c @ v1
            out_d_attn_heads = m_d @ v2

            out_c_attn_heads = out_c_attn_heads.to(v_dtype)
            out_d_attn_heads = out_d_attn_heads.to(v_dtype)

        # Concatenate heads: (B, N, num_heads * self.head_dim_kqv) = (B, N, self.projected_dim_kqv)
        out_c_attn_projected_flat = out_c_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.projected_dim_kqv)
        out_d_attn_projected_flat = out_d_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.projected_dim_kqv)

        # Output projection: map from self.projected_dim_kqv back to in_channels (C)
        # out_c_attn_flat = self.attn_out_proj_c(out_c_attn_projected_flat)  # (B, N, C)
        # out_d_attn_flat = self.attn_out_proj_d(out_d_attn_projected_flat)  # (B, N, C)

        # First residual (on flattened features, using original un-normed flattened input)
        # c_flat_res = c_flat + self.dropout(out_c_attn_flat)
        # d_flat_res = d_flat + self.dropout(out_d_attn_flat)

        # Reshape back to (B, C, H, W)
        out_c_spatial = out_c_attn_projected_flat.transpose(1, 2).view(B, self.projected_dim_kqv, H, W)
        out_d_spatial = out_d_attn_projected_flat.transpose(1, 2).view(B, self.projected_dim_kqv, H, W)
        out_c_spatial = self.attn_out_proj_c(out_c_spatial)
        out_d_spatial = self.attn_out_proj_d(out_d_spatial)

        out_c_spatial = c_feat_spatial + out_c_spatial * torch.sigmoid(self.alpha)
        out_d_spatial = d_feat_spatial + out_d_spatial * torch.sigmoid(self.beta)
        # out_c_spatial = c_feat_spatial + out_c_spatial
        # out_d_spatial = d_feat_spatial + out_d_spatial

        return out_c_spatial, out_d_spatial


class CrossAttentionFA2_abition(nn.Module):
    def __init__(self, in_channels, num_heads=8, reduction_factor=8, dropout_p=0.1, ffn_expansion=4,
                 apply_softmax_early=False):
        super().__init__()

        assert in_channels % reduction_factor == 0, "in_channels must be divisible by reduction_factor"
        self.projected_dim_kqv = in_channels // reduction_factor

        assert self.projected_dim_kqv % num_heads == 0, \
            "Projected KQV dimension after reduction must be divisible by num_heads. " \
            f"Projected_dim_kqv: {self.projected_dim_kqv}, num_heads: {num_heads}"

        self.heads = num_heads
        self.head_dim_kqv = self.projected_dim_kqv // num_heads
        self.scale = self.head_dim_kqv ** -0.5
        self.apply_softmax_early = apply_softmax_early
        self.in_channels = in_channels

        self.q_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.k_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.v_proj1 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)

        self.q_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.k_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)
        self.v_proj2 = nn.Conv1d(in_channels, self.projected_dim_kqv, kernel_size=1, bias=False)

        # self.attn_out_proj_c = nn.Linear(self.projected_dim_kqv, in_channels)
        # self.attn_out_proj_d = nn.Linear(self.projected_dim_kqv, in_channels)
        self.attn_out_proj_c = nn.Sequential(
            nn.Conv2d(self.projected_dim_kqv, in_channels, kernel_size=1, bias=False),
        )
        self.attn_out_proj_d = nn.Sequential(
            nn.Conv2d(self.projected_dim_kqv, in_channels, kernel_size=1, bias=False),
        )

        # LayerNorms - applied on the channel dimension of the flattened features (B, N, C)
        # self.norm1_c = nn.LayerNorm(in_channels)
        # self.norm1_d = nn.LayerNorm(in_channels)
        # self.norm2_c = nn.LayerNorm(in_channels)  # For FFN input
        # self.norm2_d = nn.LayerNorm(in_channels)  # For FFN input
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout_p)

    def _project_and_split_heads_spatial(self, x_norm_flat, q_conv, k_conv, v_conv, B, N):
        # x_norm_flat: (B, N, C_in)
        x_permuted = x_norm_flat.transpose(1, 2)  # (B, C_in, N) for Conv1D

        q_projected = q_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)
        k_projected = k_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)
        v_projected = v_conv(x_permuted).transpose(1, 2)  # (B, N, C_proj_kqv)

        # Reshape for multi-head attention: (B, N, num_heads, head_dim_kqv) -> (B, num_heads, N, head_dim_kqv)
        q = q_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        k = k_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        v = v_projected.view(B, N, self.heads, self.head_dim_kqv).permute(0, 2, 1, 3)
        return q, k, v

    def forward(self, c_feat_spatial, d_feat_spatial):  # Inputs: (B, C, H, W)
        B, C, H, W = c_feat_spatial.shape
        N = H * W  # Number of spatial tokens
        assert C == self.in_channels, "Input channel dimension mismatch"

        # --- Flatten and Norm c_feat ---
        c_flat = c_feat_spatial.flatten(2).transpose(1, 2)  # (B, N, C)
        # c_flat_norm = self.norm1_c(c_flat)
        q1, k1, v1 = self._project_and_split_heads_spatial(c_flat, self.q_proj1, self.k_proj1, self.v_proj1, B, N)

        # --- Flatten and Norm d_feat ---
        d_flat = d_feat_spatial.flatten(2).transpose(1, 2)  # (B, N, C)
        # d_flat_norm = self.norm1_d(d_flat)
        q2, k2, v2 = self._project_and_split_heads_spatial(d_flat, self.q_proj2, self.k_proj2, self.v_proj2, B, N)

        attn_c_raw = (q1 @ k1.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N)
        attn_d_raw = (q2 @ k2.transpose(-2, -1)) * self.scale  # (B, H_heads, N, N)


        attn_c_eff = torch.softmax(attn_c_raw, dim=-1)
        attn_d_eff = torch.softmax(attn_d_raw, dim=-1)

        # Weighted sum of values V. V has head_dim = self.projected_dim_kqv / num_heads.
        # Result is (B, num_heads, N, self.head_dim_kqv)
        out_c_attn_heads = attn_c_eff @ v1
        out_d_attn_heads = attn_d_eff @ v2

        # Concatenate heads: (B, N, num_heads * self.head_dim_kqv) = (B, N, self.projected_dim_kqv)
        out_c_attn_projected_flat = out_c_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.projected_dim_kqv)
        out_d_attn_projected_flat = out_d_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.projected_dim_kqv)

        # Output projection: map from self.projected_dim_kqv back to in_channels (C)
        # out_c_attn_flat = self.attn_out_proj_c(out_c_attn_projected_flat)  # (B, N, C)
        # out_d_attn_flat = self.attn_out_proj_d(out_d_attn_projected_flat)  # (B, N, C)

        # First residual (on flattened features, using original un-normed flattened input)
        # c_flat_res = c_flat + self.dropout(out_c_attn_flat)
        # d_flat_res = d_flat + self.dropout(out_d_attn_flat)

        # Reshape back to (B, C, H, W)
        out_c_spatial = out_c_attn_projected_flat.transpose(1, 2).view(B, self.projected_dim_kqv, H, W)
        out_d_spatial = out_d_attn_projected_flat.transpose(1, 2).view(B, self.projected_dim_kqv, H, W)
        out_c_spatial = self.attn_out_proj_c(out_c_spatial)
        out_d_spatial = self.attn_out_proj_d(out_d_spatial)

        out_c_spatial = c_feat_spatial + out_c_spatial * self.alpha
        out_d_spatial = d_feat_spatial + out_d_spatial * self.beta
        # out_c_spatial = c_feat_spatial + out_c_spatial
        # out_d_spatial = d_feat_spatial + out_d_spatial

        return out_c_spatial, out_d_spatial

class SpatialFA2_Conv2d(nn.Module):
    def __init__(self, in_channels, num_heads=8, reduction_factor=8, dropout_p=0.1, apply_softmax_early=False):
        super().__init__()

        # --- Dimension Definitions ---
        assert in_channels % reduction_factor == 0, "in_channels must be divisible by reduction_factor for Q/K projection"
        self.projected_dim_qk = in_channels // reduction_factor  # Reduced dimension for Q and K

        assert self.projected_dim_qk % num_heads == 0, \
            "Projected Q/K dimension must be divisible by num_heads. " \
            f"Projected_dim_qk: {self.projected_dim_qk}, num_heads: {num_heads}"

        assert in_channels % num_heads == 0, \
            "in_channels must be divisible by num_heads for V projection"

        self.heads = num_heads
        self.head_dim_qk = self.projected_dim_qk // num_heads  # Head dimension for Q and K
        self.head_dim_v = in_channels // num_heads  # Head dimension for V (no reduction)
        self.scale = self.head_dim_qk ** -0.5  # Scale factor uses QK head dimension

        self.in_channels = in_channels
        self.apply_softmax_early = apply_softmax_early

        # --- Projections using 1x1 Conv2d ---
        # Q and K projections reduce dimensionality
        self.q_proj1 = nn.Conv2d(in_channels, self.projected_dim_qk, kernel_size=1, bias=False)
        self.k_proj1 = nn.Conv2d(in_channels, self.projected_dim_qk, kernel_size=1, bias=False)
        # V projection does NOT reduce dimensionality
        self.v_proj1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.q_proj2 = nn.Conv2d(in_channels, self.projected_dim_qk, kernel_size=1, bias=False)
        self.k_proj2 = nn.Conv2d(in_channels, self.projected_dim_qk, kernel_size=1, bias=False)
        self.v_proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # --- Output Projection ---
        # The attended value vector will have `in_channels` dimension (since V was not reduced).
        # This Conv2d layer fuses the multi-head outputs.
        self.attn_out_proj_c = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.attn_out_proj_d = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout_p)

        # Optional: Add LayerNorm or GroupNorm if desired. The original code removed it.
        # self.norm_c = nn.GroupNorm(1, in_channels) # Example using GroupNorm which works on (B,C,H,W)
        # self.norm_d = nn.GroupNorm(1, in_channels)

    def forward(self, c_feat_spatial, d_feat_spatial):  # Inputs: (B, C, H, W)
        B, C, H, W = c_feat_spatial.shape
        N = H * W  # Number of spatial tokens
        assert C == self.in_channels, "Input channel dimension mismatch"

        # Optional: Apply normalization before projections
        # c_feat_norm = self.norm_c(c_feat_spatial)
        # d_feat_norm = self.norm_d(d_feat_spatial)
        c_feat_norm = c_feat_spatial  # If no norm
        d_feat_norm = d_feat_spatial  # If no norm

        # --- Project features for c_feat ---
        q1_proj = self.q_proj1(c_feat_norm)  # (B, C_qk, H, W)
        k1_proj = self.k_proj1(c_feat_norm)  # (B, C_qk, H, W)
        v1_proj = self.v_proj1(c_feat_norm)  # (B, C, H, W) -> V keeps original channels

        # Reshape for multi-head attention
        # (B, C_proj, H, W) -> (B, C_proj, N) -> (B, N, C_proj) -> (B, N, heads, head_dim) -> (B, heads, N, head_dim)
        q1 = q1_proj.flatten(2).view(B, self.projected_dim_qk, N).transpose(1, 2).view(B, N, self.heads,
                                                                                       self.head_dim_qk).permute(0, 2,
                                                                                                                 1, 3)
        k1 = k1_proj.flatten(2).view(B, self.projected_dim_qk, N).transpose(1, 2).view(B, N, self.heads,
                                                                                       self.head_dim_qk).permute(0, 2,
                                                                                                                 1, 3)
        v1 = v1_proj.flatten(2).view(B, self.in_channels, N).transpose(1, 2).view(B, N, self.heads,
                                                                                  self.head_dim_v).permute(0, 2, 1, 3)

        # --- Project features for d_feat ---
        q2_proj = self.q_proj2(d_feat_norm)
        k2_proj = self.k_proj2(d_feat_norm)
        v2_proj = self.v_proj2(d_feat_norm)

        q2 = q2_proj.flatten(2).view(B, self.projected_dim_qk, N).transpose(1, 2).view(B, N, self.heads,
                                                                                       self.head_dim_qk).permute(0, 2,
                                                                                                                 1, 3)
        k2 = k2_proj.flatten(2).view(B, self.projected_dim_qk, N).transpose(1, 2).view(B, N, self.heads,
                                                                                       self.head_dim_qk).permute(0, 2,
                                                                                                                 1, 3)
        v2 = v2_proj.flatten(2).view(B, self.in_channels, N).transpose(1, 2).view(B, N, self.heads,
                                                                                  self.head_dim_v).permute(0, 2, 1, 3)

        # --- Attention Calculation ---
        attn_c_raw = (q1 @ k1.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn_d_raw = (q2 @ k2.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        if self.apply_softmax_early:
            attn_c_eff = torch.softmax(attn_c_raw, dim=-1)
            attn_d_eff = torch.softmax(attn_d_raw, dim=-1)
        else:
            attn_c_eff = attn_c_raw
            attn_d_eff = attn_d_raw

        m_c = attn_d_eff @ attn_c_eff.transpose(-2, -1)
        m_d = attn_c_eff @ attn_d_eff.transpose(-2, -1)

        if not self.apply_softmax_early:
            m_c = torch.softmax(m_c, dim=-1)
            m_d = torch.softmax(m_d, dim=-1)

        # --- Weighted Sum and Output ---
        # m_c is (B, heads, N, N), v1 is (B, heads, N, head_dim_v)
        # Result is (B, heads, N, head_dim_v)
        out_c_attn_heads = m_c @ v1
        out_d_attn_heads = m_d @ v2

        # Concatenate heads and reshape back to spatial format
        # (B, heads, N, head_dim_v) -> (B, N, heads, head_dim_v) -> (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out_c_attn_spatial = out_c_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.in_channels).transpose(1, 2).view(
            B, self.in_channels, H, W)
        out_d_attn_spatial = out_d_attn_heads.permute(0, 2, 1, 3).reshape(B, N, self.in_channels).transpose(1, 2).view(
            B, self.in_channels, H, W)

        # Final output projection
        out_c_proj = self.attn_out_proj_c(out_c_attn_spatial)
        out_d_proj = self.attn_out_proj_d(out_d_attn_spatial)

        # Residual connection
        out_c_final = c_feat_spatial + self.dropout(out_c_proj)
        out_d_final = d_feat_spatial + self.dropout(out_d_proj)

        return out_c_final, out_d_final




class CR_layer(nn.Module):
    def __init__(self, in_channels):
        super(CR_layer, self).__init__()
        self.conv_clic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels / 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels / 2, in_channels / 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels / 4, 1, kernel_size=1)
        )
        self.conv_derm = nn.Sequential(
            nn.Conv2d(in_channels, in_channels / 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels / 2, in_channels / 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels / 4, 1, kernel_size=1)
        )

    def forward(self, x_clic, x_derm):
        return self.conv_clic(x_clic), self.conv_derm(x_derm)


class DownsampleResidualBlock(nn.Module):
    expansion = 4  # 这是ResNet-50 bottleneck结构的标准扩展因子

    def __init__(self, in_channels, out_channels_bottleneck):
        super().__init__()

        # 最终输出的通道数
        out_channels_final = out_channels_bottleneck * self.expansion

        # --- 主路径 (main path) ---
        # 第一个1x1卷积，步长为2，用于下采样和调整通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels_bottleneck, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_bottleneck)

        # 第二个3x3卷积
        self.conv2 = nn.Conv2d(out_channels_bottleneck, out_channels_bottleneck, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels_bottleneck)

        # 第三个1x1卷积，用于恢复通道数
        self.conv3 = nn.Conv2d(out_channels_bottleneck, out_channels_final, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_final)

        self.relu = nn.ReLU(inplace=True)

        # --- 跳跃连接路径 (shortcut path) ---
        # 需要一个下采样层来匹配主路径的输出尺寸和通道
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_final, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels_final),
        )

    def forward(self, x):
        identity = self.downsample(x)  # 首先处理跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 将主路径和跳跃连接路径相加
        out = self.relu(out)

        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # --- 主路径 (main path) ---
        # 第一个1x1卷积，步长为2，用于下采样和调整通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out

class ChannelAdjustinBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.shortcut(x)
        return out

class ChannelAdjustingResidualBlock(nn.Module):
    expansion = 4  # ResNet-50 bottleneck 结构的扩展因子

    def __init__(self, in_channels, out_channels_bottleneck):

        super().__init__()

        # 最终输出的通道数
        out_channels_final = out_channels_bottleneck * self.expansion

        # --- 主路径 (main path) ---
        # 注意：所有 stride 都是 1
        self.conv1 = nn.Conv2d(in_channels, out_channels_bottleneck, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_bottleneck)

        self.conv2 = nn.Conv2d(out_channels_bottleneck, out_channels_bottleneck, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels_bottleneck)

        self.conv3 = nn.Conv2d(out_channels_bottleneck, out_channels_final, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_final)

        self.relu = nn.ReLU(inplace=True)

        # --- 跳跃连接路径 (shortcut path) ---
        if in_channels != out_channels_final:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_final, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels_final),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)  # 首先处理跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 将主路径和跳跃连接路径相加
        out = self.relu(out)

        return out

class SpatialFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # CR Layer in paper: reduce channels to 1
        self.cr_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, 1),
        )

    def forward(self, x_c, x_d):  # 简化版，先只融合两个输入
        attn_c_map = self.cr_layer(x_c)  # (B, 1, H, W)
        attn_d_map = self.cr_layer(x_d)  # (B, 1, H, W)

        attn_maps = torch.cat([attn_c_map, attn_d_map], dim=1)  # (B, 2, H, W)
        attn_probs = F.softmax(attn_maps, dim=1)  # (B, 2, H, W)
        # 广播并加权求和
        fused_feat = x_c * attn_probs[:, 0:1, :, :] + x_d * attn_probs[:, 1:2, :, :]
        return fused_feat


class GatedCrossSourceFusionBlock(nn.Module):

    def __init__(self, in_channels, inter_channels_ratio=4):
        """
        Args:
            in_channels (int): Number of channels for the input and output feature maps.
            inter_channels_ratio (int): Reduction ratio for the intermediate channels in the context encoder and gate generators.
        """
        super(GatedCrossSourceFusionBlock, self).__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // inter_channels_ratio

        # 1. Unified Context Encoder
        # Takes the sum of all inputs and creates a context-aware feature map.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 2. Independent Gating Units
        # Each gate generator takes the context and creates a gate for one specific input.
        # We use Sigmoid to allow for independent control of information flow.
        self.gate_d = self._make_gate_generator(in_channels, inter_channels)
        self.gate_c = self._make_gate_generator(in_channels, inter_channels)
        self.gate_f = self._make_gate_generator(in_channels, inter_channels)

        # 4. Final Non-linear Fusion
        # A 1x1 convolution to fuse the concatenated gated features.
        # It reduces channels from 3 * in_channels back to in_channels.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _make_gate_generator(self, in_channels, inter_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, d_refined, c_refined, f_prev=None):
        """
        Forward pass of the GCSF block.

        Args:
            d_refined (torch.Tensor): Dermoscopy features. Shape: (B, C, H, W)
            c_refined (torch.Tensor): Clinical features. Shape: (B, C, H, W)
            f_prev (torch.Tensor, optional): Fused features from the previous stage.
                                             Shape: (B, C, H, W). Defaults to None for the first stage.
        Returns:
            torch.Tensor: The fused feature map for the current stage. Shape: (B, C, H, W)
        """
        # Handle the first stage case where f_prev is not available.
        # We create a zero tensor to keep the computation graph and architecture consistent.
        if f_prev is None:
            f_prev = torch.zeros_like(d_refined)

        # 1. Generate Unified Context
        # Summing is a simple and effective way to aggregate information.
        context_input = d_refined + c_refined + f_prev
        context = self.context_encoder(context_input)

        # 2. Generate Gates from the Context
        gate_d = self.gate_d(context)
        gate_c = self.gate_c(context)
        gate_f = self.gate_f(context)

        # 3. Apply Gates for Information Flow Control
        d_gated = d_refined * gate_d
        c_gated = c_refined * gate_c
        f_gated = f_prev * gate_f

        # 4. Final Fusion
        # Concatenate the gated features along the channel dimension.
        fusion_input = torch.cat([d_gated, c_gated, f_gated], dim=1)

        # Fuse and reduce channels using the 1x1 convolution.
        f_current = self.fusion_conv(fusion_input)

        return f_current

class Feature2Node(nn.Module):
    """ 将输入的特征图映射到图谱的概念节点上 """
    def __init__(self, in_channels, node_feature_dim, num_nodes=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.node_feature_dim = node_feature_dim

        # 投影层，将三个输入源融合成一个统一的特征表示
        self.proj = nn.Conv2d(in_channels * 3, in_channels, 1)

        # 为每个节点学习一个查询向量
        self.query_vectors = nn.Parameter(torch.randn(num_nodes, in_channels))

        # MLP将池化后的特征映射到最终的节点特征维度
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, node_feature_dim),
            nn.ReLU(),
            nn.Linear(node_feature_dim, node_feature_dim)
        )

    def forward(self, d_feat, c_feat, f_prev):
        b, c, h, w = d_feat.shape

        # 1. 融合输入特征图
        combined_feat = torch.cat([d_feat, c_feat, f_prev], dim=1)
        combined_feat = F.relu(self.proj(combined_feat)) # (B, C, H, W)

        # 2. 使用注意力池化提取节点特征
        # 将特征图展平
        feat_map_flat = combined_feat.view(b, self.in_channels, h * w) # (B, C, N)

        # 计算注意力权重 (查询向量与特征图像素的点积)
        # self.query_vectors: (num_nodes, C)
        # feat_map_flat.transpose(1, 2): (B, N, C)
        attention_weights = torch.bmm(feat_map_flat.transpose(1, 2), self.query_vectors.unsqueeze(0).repeat(b, 1, 1).transpose(1, 2))
        # -> (B, N, C) @ (B, C, num_nodes) -> (B, N, num_nodes)
        attention_weights = F.softmax(attention_weights, dim=1) # (B, N, num_nodes)

        # 加权求和得到节点初始特征
        # feat_map_flat: (B, C, N)
        # attention_weights: (B, N, num_nodes)
        initial_node_features = torch.bmm(feat_map_flat, attention_weights) # (B, C, num_nodes)
        initial_node_features = initial_node_features.permute(0, 2, 1) # (B, num_nodes, C)

        # 3. 通过MLP进行投影
        node_features = self.mlp(initial_node_features) # (B, num_nodes, node_feature_dim)

        return node_features



class KnowledgeAwareProjector(nn.Module):
    def __init__(self, in_channels, concept_channels, num_concepts, initial_queries):
        """
        Args:
            in_channels (int): Channel dimension of the visual features.
            concept_channels (int): Channel dimension of the output concept embeddings.
            num_concepts (int): Number of concepts (e.g., 8).
            initial_queries (torch.Tensor): A tensor of shape (num_concepts, in_channels)
                                            pre-initialized with knowledge. This will be made
                                            a learnable parameter.
        """
        super().__init__()
        self.in_channels = in_channels
        self.concept_channels = concept_channels
        self.num_concepts = num_concepts

        # 1. Use the knowledge-initialized tensor as the base for our queries.
        #    Make it a learnable parameter so it can be finetuned.
        if initial_queries.shape != (num_concepts, in_channels):
            raise ValueError(f"Shape of initial_queries must be ({num_concepts}, {in_channels})")
        self.base_concept_queries = nn.Parameter(initial_queries)

        # 2. Dynamic Adapter
        # This small MLP learns to dynamically adjust the base queries based on image content.
        self.adapter_pool = nn.AdaptiveAvgPool2d(1)
        self.adapter_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            # Output is an adjustment vector (delta) for each query
            nn.Linear(in_channels // 4, num_concepts * in_channels)
        )

        # 3. Final projection MLP
        self.projection_mlp = nn.Sequential(
            nn.Linear(in_channels, concept_channels),
            nn.LayerNorm(concept_channels),
            nn.ReLU()
        )

    def forward(self, agg_feat):
        # agg_feat is the aggregated visual feature map, shape (B, C_in, H, W)
        b, c, h, w = agg_feat.shape

        # --- Generate Dynamic Queries ---
        # 1. Get image-specific context
        global_context = self.adapter_pool(agg_feat).view(b, c)

        # 2. Generate dynamic adjustments (deltas)
        query_deltas = self.adapter_mlp(global_context).view(b, self.num_concepts, c)

        # 3. Create the final dynamic queries
        # Add the dynamic adjustment to the knowledge-initialized base queries
        # base_concept_queries (N, C) -> (1, N, C) -> broadcast to (B, N, C)
        dynamic_queries = self.base_concept_queries.unsqueeze(0) + query_deltas

        # --- Attention Pooling ---
        agg_feat_flat = agg_feat.view(b, c, -1)
        attn_scores = torch.bmm(dynamic_queries, agg_feat_flat)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_features = torch.bmm(attn_weights, agg_feat_flat.permute(0, 2, 1))

        # --- Final Projection ---
        final_concept_embeddings = self.projection_mlp(attended_features)

        return final_concept_embeddings


class MultiScaleFeatureAggregator(nn.Module):
    """Aggregates input features at multiple scales."""

    def __init__(self, in_channels, out_channels, scales=(1, 2, 3, 6)):
        super().__init__()
        # Input to this module is concatenated features (D, C, F_prev)
        input_dim = in_channels * 3
        self.stages = nn.ModuleList()
        # The first stage reduces the channel dimension
        self.stages.append(nn.Sequential(
            nn.Conv2d(input_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        # Subsequent stages perform adaptive pooling at different scales
        for scale in scales:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(scale, scale)),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

        # Final convolution to fuse features from all scales
        self.bottleneck = nn.Conv2d(out_channels * (len(scales) + 1), out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, d_feat, c_feat, f_prev_feat):
        x = torch.cat([d_feat, c_feat, f_prev_feat], dim=1)

        # Initial projection
        base_feat = self.stages[0](x)
        h, w = base_feat.shape[2:]

        pyramid_feats = [base_feat]
        for i, stage in enumerate(self.stages[1:]):
            # Upsample the pooled feature back to the original size
            pooled = stage(base_feat)
            pyramid_feats.append(F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False))

        # Concatenate and fuse all scale features
        agg_feat = self.relu(self.bottleneck(torch.cat(pyramid_feats, dim=1)))
        return agg_feat


class Feature2ConceptProjector_V2(nn.Module):
    """
    An enhanced projector using multi-scale features and dynamic queries.
    """

    def __init__(self, in_channels, concept_channels, num_concepts=8):
        super().__init__()
        self.in_channels = in_channels
        self.concept_channels = concept_channels
        self.num_concepts = num_concepts

        # 1. Multi-scale Aggregator
        self.aggregator = MultiScaleFeatureAggregator(in_channels, in_channels)

        # 2. Dynamic Query Generator
        # It takes the global context of the image to generate queries
        self.query_generator_pool = nn.AdaptiveAvgPool2d(1)
        self.query_generator_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            # Output: num_concepts * in_channels (for 8 queries)
            nn.Linear(in_channels // 2, num_concepts * in_channels)
        )

        # 3. Final MLP to project the attended features
        self.projection_mlp = nn.Sequential(
            # Input will be 2*in_channels because we concatenate global and local info
            nn.Linear(in_channels * 2, concept_channels),
            nn.LayerNorm(concept_channels),
            nn.ReLU()
        )

    def forward(self, d_feat, c_feat, f_prev_feat):
        # 1. Aggregate features at multiple scales
        agg_feat = self.aggregator(d_feat, c_feat, f_prev_feat)  # Shape: (B, C_in, H, W)
        b, c, h, w = agg_feat.shape

        # 2. Generate Dynamic Queries
        global_context = self.query_generator_pool(agg_feat).view(b, c)  # (B, C_in)
        dynamic_queries = self.query_generator_mlp(global_context)  # (B, num_concepts * C_in)
        # Reshape to (B, num_concepts, C_in)
        dynamic_queries = dynamic_queries.view(b, self.num_concepts, c)

        # --- Attention Pooling (Global Info) ---
        agg_feat_flat = agg_feat.view(b, c, -1)
        # Attention scores: (B, num_concepts, C_in) @ (B, C_in, H*W) -> (B, num_concepts, H*W)
        attn_scores = torch.bmm(dynamic_queries, agg_feat_flat)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (B, num_concepts, H*W)

        # Attended features (global representation for each concept)
        # (B, num_concepts, H*W) @ (B, H*W, C_in) -> (B, num_concepts, C_in)
        global_attended_features = torch.bmm(attn_weights, agg_feat_flat.permute(0, 2, 1))

        # --- Top-K Pooling (Local Highlight Info) ---
        # For each concept's attention map, find the most salient local features
        k = 5  # Number of top local features to consider
        top_k_values, top_k_indices = torch.topk(attn_weights, k, dim=-1)  # (B, num_concepts, k)

        # Gather the corresponding features from agg_feat_flat
        # This requires some advanced indexing
        batch_indices = torch.arange(b)[:, None, None].expand(-1, self.num_concepts, k)
        concept_indices = torch.arange(self.num_concepts)[None, :, None].expand(b, -1, k)

        # Flatten top_k_indices for easier indexing
        # Note: This part is a bit complex. A simpler alternative is to just use the global features.
        # But for max performance, this is how you'd do it.
        # Let's use a simpler, more robust way: weighted average of top-k features
        top_k_weights = F.softmax(top_k_values, dim=-1)

        # We need to gather features corresponding to top_k_indices
        # This is a bit tricky with bmm. Let's use a loop for clarity, can be vectorized.
        local_highlight_features = []
        for i in range(b):
            # indices for this sample: (num_concepts, k)
            # features for this sample: (C_in, H*W) -> (H*W, C_in)
            sample_features_t = agg_feat_flat[i].permute(1, 0)
            # Use gather
            indices = top_k_indices[i].unsqueeze(-1).expand(-1, -1, c)  # (num_concepts, k, C_in)
            gathered_feats = torch.gather(sample_features_t.unsqueeze(0).expand(self.num_concepts, -1, -1), 1,
                                          indices)  # (num_concepts, k, C_in)

            # Weighted average of top-k features
            weighted_local_feats = torch.sum(gathered_feats * top_k_weights[i].unsqueeze(-1),
                                             dim=1)  # (num_concepts, C_in)
            local_highlight_features.append(weighted_local_feats)

        local_highlight_features = torch.stack(local_highlight_features, dim=0)  # (B, num_concepts, C_in)

        # --- 4. Combine Global and Local and Project ---
        combined_features = torch.cat([global_attended_features, local_highlight_features], dim=-1)

        # Final projection to concept space
        final_concept_embeddings = self.projection_mlp(combined_features)  # Shape: (B, num_concepts, C_concept)

        return final_concept_embeddings



class DepthwiseSeparableConv(nn.Module):
    """
    A robust and reusable Depthwise Separable Convolution block.

    This block is a common building block for efficient neural networks
    like MobileNet. It consists of a Depthwise Convolution followed by
    a Pointwise Convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to all four sides of the input.
            dilation (int): Spacing between kernel elements.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(DepthwiseSeparableConv, self).__init__()

        # --- Depthwise Convolution ---
        # Each input channel is convolved with its own set of filters.
        # 'groups=in_channels' is the key to making it a depthwise convolution.
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,  # Output channels are the same as input channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # This makes it depthwise
            bias=bias
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.relu_depthwise = nn.ReLU(inplace=True)

        # --- Pointwise Convolution ---
        # A standard 1x1 convolution to mix channel information.
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.relu_pointwise = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W').
        """
        # Apply depthwise convolution part
        out = self.depthwise(x)
        out = self.bn_depthwise(out)
        out = self.relu_depthwise(out)

        # Apply pointwise convolution part
        out = self.pointwise(out)
        out = self.bn_pointwise(out)
        out = self.relu_pointwise(out)

        return out


class Feature2ConceptProjector(nn.Module):
    def __init__(self, in_channels, concept_channels, num_concepts=8):
        super().__init__()
        self.in_channels = in_channels
        self.concept_channels = concept_channels
        self.num_concepts = num_concepts

        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, in_channels))
        self.projection_mlp = nn.Sequential(
            nn.Linear(in_channels, concept_channels),
            nn.LayerNorm(concept_channels),
            nn.ReLU()
        )

    def forward(self, agg_feat):
        b, c, h, w = agg_feat.shape
        agg_feat_flat = agg_feat.view(b, c, -1)
        attn_scores = torch.bmm(self.concept_queries.repeat(b, 1, 1), agg_feat_flat)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_features = torch.bmm(attn_weights, agg_feat_flat.permute(0, 2, 1))
        initial_concept_embeddings = self.projection_mlp(attended_features)
        return initial_concept_embeddings

class SpatioChannelRouter_V1(nn.Module):
    def __init__(self, concept_channels, num_sources=3, feature_channels=64):
        super().__init__()
        self.num_sources = num_sources
        self.feature_channels = feature_channels
        self.router_mlp = nn.Sequential(
            nn.Linear(concept_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_sources * feature_channels)  # Output dim is much larger
        )

    def forward(self, kg_guidance):
        routing_logits = self.router_mlp(kg_guidance)
        routing_logits = routing_logits.view(-1, self.num_sources, self.feature_channels)
        routing_weights = F.softmax(routing_logits, dim=1)  # (B, 3, C_feat)
        return routing_weights.unsqueeze(-1).unsqueeze(-1)


class HRF_Block_V4(nn.Module):
    def __init__(self, in_channels, concept_channels, num_concepts=8, device='cpu'):
        super().__init__()
        self.device = device
        self.num_concepts = num_concepts
        self.concept_channels = concept_channels

        self.kg_pooler = nn.AdaptiveAvgPool1d(1)
        self.router = SpatioChannelRouter_V1(concept_channels, num_sources=3, feature_channels=in_channels)  # 3 sources: D, C, F_prev

        self.fusion_processor = DepthwiseSeparableConv(in_channels, in_channels)

        self.projector = Feature2ConceptProjector(in_channels, concept_channels, num_concepts)  # Can use V1 or V2

        self.kg_updater = nn.Sequential(
            nn.Linear(concept_channels * 2, concept_channels),
            nn.ReLU(),
            nn.Linear(concept_channels, concept_channels)
        )
        self.kg_norm = nn.LayerNorm(concept_channels)

    def forward(self, d_feat, c_feat, f_prev, kg_prev):
        """
        d_feat, c_feat are features that have already passed through cross-attention.
        """
        if f_prev is None: f_prev = torch.zeros_like(d_feat)
        if kg_prev is None:
            b, _, c_concept = d_feat.shape[0], d_feat.shape[1], self.concept_channels
            kg_prev = torch.zeros(b, self.num_concepts, c_concept, device=self.device)
        kg_guidance = self.kg_pooler(kg_prev.transpose(1, 2)).squeeze(-1)
        routing_weights = self.router(kg_guidance)
        f_routed = (d_feat * routing_weights[:, 0] +
                    c_feat * routing_weights[:, 1] +
                    f_prev * routing_weights[:, 2])
        f_final = f_routed + self.fusion_processor(f_routed)
        kg_initial_curr = self.projector(f_final)
        update_info = self.kg_updater(torch.cat([kg_initial_curr, kg_prev], dim=-1))
        kg_curr = self.kg_norm(kg_prev + update_info)  # Residual update

        return f_final, kg_curr





def inverse_sigmoid(y):
    """Calculates the logit for a given probability y."""
    return torch.log(y / (1 - y))


class ExpectationScoringHead_V1(nn.Module):
    def __init__(self, in_features=512, bottleneck_dim=256, num_classes_list=None):  # class_list
        super().__init__()
        self.num_tasks = len(num_classes_list)
        self.num_checklist_tasks = self.num_tasks - 1
        checklist_num_classes = num_classes_list[1:]

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim), Swish_Module(), nn.Dropout(0.3)
        )
        self.task_classifiers = nn.ModuleList([
            nn.Linear(bottleneck_dim, num_classes) for num_classes in checklist_num_classes
        ])

        # --- 关键修改：可学习的 Suspicion Logits ---
        # 1. 定义先验知识 (目标概率)
        prior_suspicion_probs = {
            'pn': torch.tensor([0.01, 0.5, 0.9]),  # absent, typical, atypical
            'str': torch.tensor([0.01, 0.5, 0.9]),  # absent, regular, irregular
            'pig': torch.tensor([0.01, 0.5, 0.9]),
            'rs': torch.tensor([0.01, 0.9]),
            'dag': torch.tensor([0.01, 0.5, 0.9]),
            'bwv': torch.tensor([0.01, 0.9]),
            'vs': torch.tensor([0.01, 0.5, 0.9]),
        }

        # 2. 将先验概率转换为logits，并创建可学习的Parameter
        self.suspicion_logits = nn.ParameterDict()
        for task_name, probs in prior_suspicion_probs.items():
            # 使用inverse_sigmoid进行初始化
            initial_logits = inverse_sigmoid(torch.clamp(probs, 1e-6, 1 - 1e-6))
            self.suspicion_logits[task_name] = nn.Parameter(initial_logits, requires_grad=True)

        self.task_names = ['pn', 'str', 'pig', 'rs', 'dag', 'bwv', 'vs']
        self.register_buffer('clinical_weights', torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]))

        self.rule_path_mlp = nn.Sequential(
            nn.Linear(1, 64),
            Swish_Module(),
            nn.Linear(64, num_classes_list[0])  # 直接输出logits
        )
        self.vision_path_mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            Swish_Module(),
            nn.Linear(bottleneck_dim // 2, num_classes_list[0])  # 直接输出logits
        )
        self.logit_gate = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, x):
        shared_embedding = self.bottleneck(x)
        checklist_logits = [classifier(shared_embedding) for classifier in self.task_classifiers]

        suspicion_scores = []
        for i, task_name in enumerate(self.task_names):
            logits = checklist_logits[i]
            probs = F.softmax(logits, dim=1)

            # --- 关键修改：从logits生成可学习的weights ---
            # 对学习的logits应用sigmoid，得到0-1之间的权重
            current_suspicion_weights = torch.sigmoid(self.suspicion_logits[task_name])

            expected_score = torch.sum(probs * current_suspicion_weights, dim=1)
            suspicion_scores.append(expected_score)

        suspicion_scores_tensor = torch.stack(suspicion_scores, dim=1)

        # --- 后面部分完全不变 ---
        final_score = torch.sum(suspicion_scores_tensor * self.clinical_weights, dim=1, keepdim=True)
        rule_logits = self.rule_path_mlp(final_score)
        vision_logits = self.vision_path_mlp(shared_embedding)
        weights = F.softmax(self.logit_gate, dim=0)
        final_diag_logit = weights[0] * rule_logits + weights[1] * vision_logits
        output_logits = [final_diag_logit] + checklist_logits

        return output_logits


class MLCNN(nn.Module):
    def __init__(self, class_list, device, ):
        super().__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool

        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool

        # self.fusion_layer4 = copy.deepcopy(self.model_clinic.layer4)

        self.hyper_module_0 = CrossAttentionFA2_Conv(in_channels=64, num_heads=1, reduction_factor=16)
        self.hyper_module_1 = CrossAttentionFA2_Conv(in_channels=256, num_heads=1, reduction_factor=16)
        self.hyper_module_2 = CrossAttentionFA2_Conv(in_channels=512, num_heads=1, reduction_factor=16)
        self.hyper_module_3 = CrossAttentionFA2_Conv(in_channels=1024, num_heads=1, reduction_factor=16)
        #
        self.fusion_transition_0_to_1 = ChannelAdjustingResidualBlock(64, 64)
        self.fusion_transition_1_to_2 = DownsampleResidualBlock(256, 128)
        self.fusion_transition_2_to_3 = DownsampleResidualBlock(512, 256)


        self.fusion_block_0 = HRF_Block_V4(in_channels=64, concept_channels=128, device=device).to(device)
        self.fusion_block_1 = HRF_Block_V4(in_channels=256, concept_channels=128, device=device).to(device)
        self.fusion_block_2 = HRF_Block_V4(in_channels=512, concept_channels=128, device=device).to(device)
        self.fusion_block_3 = HRF_Block_V4(in_channels=1024, concept_channels=128, device=device).to(device)

        self.clinic_classifier = ExpectationScoringHead_V1(in_features=2048, bottleneck_dim=128, num_classes_list=class_list)
        self.derm_classifier = ExpectationScoringHead_V1(in_features=2048, bottleneck_dim=128, num_classes_list=class_list)
        self.fusion_classifier = ExpectationScoringHead_V1(in_features=1024, bottleneck_dim=128, num_classes_list=class_list)
        self.head_classifier = ExpectationScoringHead_V1(in_features=5120, bottleneck_dim=128, num_classes_list=class_list)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)

        print(">>> total parameters:", sum(p.numel() for p in self.parameters()))

    def run_hyper_module(self, module, feat1, feat2):
        return module(feat1, feat2)

    def forward(self, x):
        x_clic, x_derm = x

        # 特征提取
        x_clic = self.conv1_cli(x_clic)
        x_clic = self.bn1_cli(x_clic)
        x_clic = self.relu_cli(x_clic)
        x_clic = self.maxpool_cli(x_clic)
        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)

        if self.training:
            # 使用checkpoint，只在训练模式下
            x_clic, x_derm = checkpoint.checkpoint(self.run_hyper_module, self.hyper_module_0, x_clic, x_derm,
                                                   preserve_rng_state=True)
        else:
            x_clic, x_derm = self.hyper_module_0(x_clic, x_derm)
        #
        # fusion0:torch.Size([32, 64, 56, 56]), kg0:torch.Size([32, 8, 128])
        fusion_0, kg_0 = self.fusion_block_0(x_clic, x_derm, f_prev=None, kg_prev=None)
        # fusion_0 = self.fusion_block_0(x_clic, x_derm, f_prev=None)
        fusion_0 = self.fusion_transition_0_to_1(fusion_0)

        # layer1
        x_clic = self.layer1_cli(x_clic)
        x_derm = self.layer1_derm(x_derm)
        if self.training:
            x_clic, x_derm = checkpoint.checkpoint(self.run_hyper_module, self.hyper_module_1, x_clic, x_derm,
                                                   preserve_rng_state=True)
        else:
            x_clic, x_derm = self.hyper_module_1(x_clic, x_derm)

        # fusion1:torch.Size([32, 256, 56, 56]), kg1:torch.Size([32, 8, 128])
        fusion_1, kg_1 = self.fusion_block_1(x_clic, x_derm, f_prev=fusion_0, kg_prev=kg_0)
        # fusion_1 = self.fusion_block_1(x_clic, x_derm, f_prev=fusion_0)
        fusion_1 = self.fusion_transition_1_to_2(fusion_1)

        # layer2
        x_clic = self.layer2_cli(x_clic)
        x_derm = self.layer2_derm(x_derm)
        if self.training:
            x_clic, x_derm = checkpoint.checkpoint(self.run_hyper_module, self.hyper_module_2, x_clic, x_derm,
                                                   preserve_rng_state=True)
        else:
            x_clic, x_derm = self.hyper_module_2(x_clic, x_derm)

        # fusion2:torch.Size([32, 512, 28, 28]), kg2:torch.Size([32, 8, 128])
        fusion_2, kg_2 = self.fusion_block_2(x_clic, x_derm, f_prev=fusion_1, kg_prev=kg_1)
        # fusion_2 = self.fusion_block_2(x_clic, x_derm, f_prev=fusion_1)
        fusion_2 = self.fusion_transition_2_to_3(fusion_2)

        # layer3
        x_clic = self.layer3_cli(x_clic)
        x_derm = self.layer3_derm(x_derm)
        # x_clic, x_derm = self.hyper_module_3(x_clic, x_derm)
        if self.training:
            x_clic, x_derm = checkpoint.checkpoint(self.run_hyper_module, self.hyper_module_3, x_clic, x_derm,
                                                   preserve_rng_state=True)
        else:
            x_clic, x_derm = self.hyper_module_3(x_clic, x_derm)

        fusion_3, kg_3 = self.fusion_block_3(x_clic, x_derm, f_prev=fusion_2, kg_prev=kg_2)

        # layer4
        x_clic = self.layer4_cli(x_clic)
        x_derm = self.layer4_derm(x_derm)

        x_clic = self.avgpool_cli(x_clic)
        x_clic = x_clic.view(x_clic.size(0), -1)
        x_derm = self.avgpool_derm(x_derm)
        x_derm = x_derm.view(x_derm.size(0), -1)

        # fusion_3 = self.fusion_layer4(fusion_3)
        fusion_3 = self.avg_pool(fusion_3).view(fusion_3.size(0), -1)
        fusion_head = torch.cat((x_clic, x_derm, fusion_3), dim=1)
        logit_fusion = self.fusion_classifier(fusion_3)
        logit_clinic = self.clinic_classifier(x_clic)
        logit_derm = self.clinic_classifier(x_derm)
        logit_head = self.head_classifier(fusion_head)

        return [
            (
                logit_derm, logit_clinic, logit_fusion, logit_head,
             ),

        ]

    def criterion(self, logit, truth):

        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss

    def criterion_diag(self, logit, truth):

        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logit, truth)

        return loss


    def metric(self, logit, truth):
        prob = F.sigmoid(logit)
        # _, prediction = torch.max(logit.data, 1)
        _, prediction = torch.max(prob, 1)

        acc = torch.sum(prediction == truth)
        return acc


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError











