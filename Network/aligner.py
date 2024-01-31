# Trainer for MaskGIT aligner
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

# from Trainer.trainer import Trainer
from Trainer.vit import MaskGIT
from Network.transformer import MaskTransformer, PreNorm

from Network.Taming.models.vqgan import VQModel

class AlignerAttention(nn.Module):
    def __init__(self, proj_weight, dropout_p = 0.):
        """ Initialize the Aligner Attention module. 
            :param:
                attn -> Attention: Attention module in transformer layer       
        """
        super().__init__()
        self.embed_dim = proj_weight.shape[-1]
        self.proj_q, self.proj_k, self.proj_v = [nn.Parameter(i, requires_grad = False) for i in proj_weight.chunk(3)] # Each of size (embed_dim, embed_dim)
        self.dropout_p = dropout_p

    def forward(self, x, tok):
        """ Forward pass through the Attention module.
            :param:
                x   -> torch.Tensor: Input tensor to transformer layers (S, embed_dim)
                tok -> torch.Tensor: Input tok tensor (N, T, embed_dim)
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        batch_size = x.shape[0]
        tok = tok.expand(batch_size, *tok.shape)

        q = F.linear(x, self.proj_q) # (N, S, embed_dim)
        k = F.linear(tok, self.proj_k) # (N, T, embed_dim)
        v = F.linear(tok, self.proj_v) # (N, T, embed_dim)
        attention_weight = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim) # (N, S, T)
        attention_weight = F.softmax(attention_weight, dim = -1)
        attention_weight = F.dropout(attention_weight, self.dropout_p, train = True) 
        attention_value = torch.matmul(attention_weight, v) # (N, S, embed_dim)
        return attention_value, attention_weight

class TransformerEncoderAligner(nn.Module):
    def __init__(self, transformer, dim, num_prefix_tok):
        """ Initialize the Attention module.
            :param:
                transformer   -> TransformerEncoder: transformer encoder to align to
                num_prefix_tok -> int : number of prefix tokens
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.transformer = transformer
        self.layers = nn.ModuleList([])
        self.gates = nn.ParameterList([])
        self.prefix_token = nn.Parameter(torch.rand(num_prefix_tok, dim))

        # Initialize aligner layers and gates per transformer layer
        for attn, _ in self.transformer.layers:
            self.layers.append(
                PreNorm(dim, AlignerAttention(attn.fn.mha.in_proj_weight))
            )
            self.gates.append(
                nn.Parameter(torch.Tensor([0.]))
            )

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention weights
                l_p_attn -> list(torch.Tensor): list of prefix attention weights
        """
        l_attn = []
    
        for i, p_attn in enumerate(self.layers):
            attn, ff = self.transformer.layers[i]
            gate = self.gates[i]
            attention_value, attention_weight = attn(x)
            prefix_attn_value, prefix_attn_weight = p_attn(x, tok = self.prefix_token)
            attention_value = attention_value + gate*prefix_attn_value
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn
    
class MaskAligner(nn.Module):
    """
    Aligner + MaskGit model
    """
    def __init__(self, pretrain_model, num_prefix_tok, hidden_dim):
        super().__init__()
        for params in pretrain_model.parameters():
            params.requires_grad = False

        self.pretrain_model = pretrain_model
        self.nclass = pretrain_model.nclass
        self.patch_size = pretrain_model.patch_size
        self.codebook_size = pretrain_model.codebook_size
        self.tok_emb = pretrain_model.tok_emb
        self.pos_emb = pretrain_model.pos_emb
        self.num_prefix_tok = num_prefix_tok # Prefix token dimensions
        # First layer before the Transformer block
        self.first_layer = pretrain_model.first_layer

        self.transformer = TransformerEncoderAligner(pretrain_model.transformer, hidden_dim, num_prefix_tok)

        # Last layer after the Transformer block
        self.last_layer = pretrain_model.last_layer

        # Bias for the last linear output
        self.bias = pretrain_model.bias

    def forward(self, img_token, y=None, drop_label=None, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h = img_token.size()

        cls_token = y.view(b, -1) + self.codebook_size + 1  # Shift the class token by the amount of codebook

        cls_token[drop_label] = self.codebook_size + 1 + self.nclass  # Drop condition
        input = torch.cat([img_token.view(b, -1), cls_token.view(b, -1)], -1)  # concat visual tokens and class tokens
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.patch_size * self.patch_size, :self.codebook_size + 1], attn

        return logit[:, :self.patch_size*self.patch_size, :self.codebook_size+1]

