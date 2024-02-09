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
from Network.transformer import MaskTransformer, PreNorm, TransformerEncoder

from Network.Taming.models.vqgan import VQModel
from dataset.datasets import ArtDataset, SDDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class AlignerAttention(nn.Module):
    def __init__(self, proj_weight, num_heads, proj_bias = None, dropout_p = 0., mask = None):
        """ Initialize the Aligner Attention module. 
            :param:
                attn -> Attention: Attention module in transformer layer       
        """
        super().__init__()
        self.embed_dim = proj_weight.shape[-1]
        self.proj_q, self.proj_k, self.proj_v = [nn.Parameter(i, requires_grad = False) for i in proj_weight.chunk(3)] # Each of size (embed_dim, embed_dim)
        if proj_bias is not None:
            self.bias_q, self.bias_k, self.bias_v = [nn.Parameter(i, requires_grad = False) for i in proj_bias.chunk(3)]
        else:
            self.bias_q, self.bias_k, self.bias_v = None, None, None
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.head_dim = self.embed_dim//num_heads

    def forward(self, x, tok):
        """ Forward pass through the Attention module.
            :param:
                x   -> torch.Tensor: Input tensor to transformer layers (S, embed_dim)
                tok -> torch.Tensor: Input tok tensor (N, T, embed_dim)
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        batch_size, seq_length, _ = x.shape
        tok = tok.expand(batch_size, *tok.shape)
        
        q = F.linear(x, self.proj_q, self.bias_q) # (N, S, embed_dim)
        k = F.linear(tok, self.proj_k, self.bias_k) # (N, T, embed_dim)
        v = F.linear(tok, self.proj_v, self.bias_v) # (N, T, embed_dim)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, tok.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch_size, tok.shape[1], self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3) # batch, head, seq, dim
        v = v.permute(0, 2, 1, 3)

        attention = F.scaled_dot_product_attention(q, k, v)

        return attention

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
        self.aligner_layers = nn.ModuleList([])
        self.aligner_gates = nn.ParameterList([])
        # self.prefix_token = nn.Parameter(torch.rand(num_prefix_tok, dim))
        self.prefix_token = nn.Embedding(num_prefix_tok, dim)
        # Initialize aligner layers and gates per transformer layer
        for attn, _ in self.transformer.layers:
            self.aligner_layers.append(
                PreNorm(dim, AlignerAttention(attn.fn.mha.in_proj_weight, attn.fn.mha.num_heads, attn.fn.mha.in_proj_bias))
            )
            self.aligner_gates.append(
                nn.Parameter(torch.zeros(1, attn.fn.mha.num_heads, 1, 1))
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
        B, T, C = x.shape
        l_attn = []
    
        for i, p_attn in enumerate(self.aligner_layers):
            attn, ff = self.transformer.layers[i]
            gate = self.aligner_gates[i]
            _, attention_weight = attn(x) #attention_weight = softmax(qk.T)
            v = F.linear(attn.norm(x), p_attn.fn.proj_v, p_attn.fn.bias_v)
            v = v.reshape(B, T, attn.fn.n_head, attn.fn.head_dim)
            v = v.permute(0, 2, 1, 3)
            attention_value = attention_weight @ v


            prefix_attn = p_attn(x, tok = self.prefix_token.weight)
            attention_value = attention_value + gate*prefix_attn
            
            attention_value = attention_value.transpose(1, 2).contiguous().view(B, T, C)
            attention_value = attn.fn.mha.out_proj(attention_value) # Includes weight and bias
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn
    
class NonConditionalMaskTransformer(MaskTransformer):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        super().__init__(img_size, hidden_dim, codebook_size, depth, heads, mlp_dim, dropout, nclass)

    def set_weights(self, weight, pos_emb_weight, bias_weight):
        self.tok_emb = nn.Embedding(self.codebook_size+1, self.hidden_dim, _weight = weight)
        self.pos_emb = nn.Parameter(pos_emb_weight)
        self.bias = nn.Parameter(bias_weight)

    def forward(self, img_token, y = None, drop_label=None, return_attn=False):
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


        input = img_token.view(b, -1)
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
            return logit, attn

        return logit

class MaskGITAligner(MaskGIT):
    """
    Trainer for pre-trained MaskGIT with aligner. Freezes maskgit layers and only trains on prefix tokens. 
    """
    def __init__(self, args):
        super().__init__(args)
        self.freeze_layers()
        self.num_prefix_tok = args.num_prefix_tok
        self.vit.transformer = self.get_aligner()
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = NonConditionalMaskTransformer(
                img_size=self.args.img_size, hidden_dim=768, codebook_size=1024, depth=24, heads=16, mlp_dim=3072, dropout=0.1     # Small
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
            )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                self.args.initial_epoch = self.args.global_epoch
                # Load network
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.set_weights(model.tok_emb.weight[:1025], model.pos_emb.data[:, :-1,:].detach().clone(), model.bias[:-1, :1025].detach().clone())
            model = model.to(self.args.device)

            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            # Load config
            config = OmegaConf.load(os.path.join(self.args.vqgan_folder, "model.yaml"))
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

        
    def freeze_layers(self):
        for params in self.vit.parameters():
            params.requires_grad = False

    def get_aligner(self):
        return TransformerEncoderAligner(self.vit.transformer, self.vit.hidden_dim, self.num_prefix_tok).to(self.args.device)

    def get_data(self):

        train_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, split = "train")
        val_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, split = "val")
        
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.args.is_multi_gpus else None
        test_sampler = DistributedSampler(val_dataset, shuffle=True) if self.args.is_multi_gpus else None
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.bsize,
                                  shuffle=False if self.args.is_multi_gpus else True,
                                  num_workers=self.args.num_workers, pin_memory=True,
                                  drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.args.bsize,
                                  shuffle=False if self.args.is_multi_gpus else True,
                                  num_workers=self.args.num_workers, pin_memory=True,
                                  drop_last=True, sampler=test_sampler)
        
        return train_loader, val_loader

class ConditionalMaskGITAligner(MaskGIT):
    """
    Trainer for pre-trained MaskGIT with aligner. Freezes maskgit layers and only trains on prefix tokens. 
    """
    def __init__(self, args):
        super().__init__(args)
        self.freeze_layers()
        self.num_prefix_tok = args.num_prefix_tok
        #if isinstance(self.vit.transformer, TransformerEncoder):
        self.get_aligner() 
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))
        
    def freeze_layers(self):
        for params in self.vit.parameters():
            params.requires_grad = False

    def get_aligner(self):
        ckpt = self.args.vit_folder
        ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
        if self.args.is_master:
            print("load ckpt from:", ckpt)
        # Read checkpoint file
        checkpoint = torch.load(ckpt, map_location='cpu')
     
        if self.args.resume:
            self.vit.transformer = TransformerEncoderAligner(self.vit.transformer, self.vit.hidden_dim, self.num_prefix_tok).to(self.args.device)
        if self.args.is_finetuned:
            self.vit.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                # labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, random.randint(0, 10)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == 1024).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, 1024, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision
                    if w != 0:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = self.vit(code.clone(), labels, drop_label=~drop)
                logit = torch.nan_to_num(logit, 0)
                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob, validate_args=False)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0, 1023)
            x = self.ae.decode_code(_code)

        self.vit.train()
        return x, l_codes, l_mask

    
    def get_data(self):
        if self.args.data == "wikiart":

            train_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, split = "train")
            val_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, split = "val")
            
            
            train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.args.is_multi_gpus else None
            test_sampler = DistributedSampler(val_dataset, shuffle=True) if self.args.is_multi_gpus else None
            
            train_loader = DataLoader(train_dataset, batch_size=self.args.bsize,
                                    shuffle=False if self.args.is_multi_gpus else True,
                                    num_workers=self.args.num_workers, pin_memory=True,
                                    drop_last=True, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=self.args.bsize,
                                    shuffle=False if self.args.is_multi_gpus else True,
                                    num_workers=self.args.num_workers, pin_memory=True,
                                    drop_last=True, sampler=test_sampler)
            
            return train_loader, val_loader
        elif self.args.data == "sd":
            
            train_dataset = SDDataset(self.args.data_folder, self.args.artist)
            #val_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, split = "val")
            
            train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.args.is_multi_gpus else None
            #test_sampler = DistributedSampler(val_dataset, shuffle=True) if self.args.is_multi_gpus else None
            train_loader = DataLoader(train_dataset, batch_size=self.args.bsize,
                                    shuffle=False if self.args.is_multi_gpus else True,
                                    num_workers=self.args.num_workers, pin_memory=True,
                                    drop_last=True, sampler=train_sampler)
            return train_loader, None