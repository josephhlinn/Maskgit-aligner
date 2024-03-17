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
from Network.transformer import MaskTransformer, PreNorm, Attention, TransformerEncoder, FeedForward

from Network.Taming.models.vqgan import VQModel
from Network.Taming.modules.discriminator.model import NLayerDiscriminator, GANLoss, PixelDiscriminator
from dataset.datasets import ArtDataset, SDDataset, ImagePairDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


    
class AttentionV2(Attention):
    """
    Drop-in for the Attention class
    """
    def __init__(self, embed_dim, num_heads, dropout=0., peft_type = None, r = None, alpha = None):
        """ Initialize the Attention module with PEFTs.
            :param:
                embed_dim -> int: embedding dimension
                num_heads -> int: num heads in MHA
                dropout -> float: dropout rate between 0 and 1
                peft_type -> List[str]: list of applied PEFTs --> Can apply LoRA and prefix-based PEFT at the same time
                r -> int: rank for LoRA, only used when "lora" in peft_type
                alpha -> float: scaling factor in lora for weight matrices     
        """
        super().__init__(embed_dim, num_heads, dropout)
        # Get projection weights and biases from parent class, set requires_grad to False
        self.proj_q, self.proj_k, self.proj_v = [nn.Parameter(i, requires_grad = False) for i in self.mha.in_proj_weight.chunk(3)] # Each of size (embed_dim, embed_dim)
        
        if self.mha.in_proj_bias is not None:
            self.bias_q, self.bias_k, self.bias_v = [nn.Parameter(i, requires_grad = False) for i in self.mha.in_proj_bias.chunk(3)]
        else:
            self.bias_q, self.bias_k, self.bias_v = None, None, None

        self.embed_dim = embed_dim
        self.head_dim = embed_dim//num_heads
        self.num_heads = num_heads
        self.add_lora = False
        # if peft_type == "aligner" or peft_type == "adapter":
        if "aligner" in peft_type or "adapter" in peft_type:
            self.adapter_gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        # if peft_type == "lora":
        if "lora" in peft_type:
            self.add_lora = True
            self.lora_weight_proj_q = nn.ParameterDict({
                "A": nn.Parameter(
                    nn.init.kaiming_uniform_(torch.empty(self.proj_q.shape[0], r), a = math.sqrt(5))
                    ),
                "B": nn.Parameter(
                    nn.init.zeros_(torch.empty(r, self.proj_q.shape[1]))
                    )
            }
            )
            self.lora_scaling = alpha/r
            self.lora_weight_proj_k = nn.ParameterDict({
                "A": nn.Parameter(
                    nn.init.kaiming_uniform_(torch.empty(self.proj_k.shape[0], r), a = math.sqrt(5))
                    ),
                "B": nn.Parameter(
                    nn.init.zeros_(torch.empty(r, self.proj_k.shape[1]))
                    )
            }
            )
            # self.lora_weight_proj_v = nn.ParameterDict({
            #     "A": nn.Parameter(
            #         nn.init.kaiming_uniform_(torch.empty(self.proj_v.shape[0], r), a = math.sqrt(5))
            #         ),
            #     "B": nn.Parameter(
            #         nn.init.zeros_(torch.empty(r, self.proj_v.shape[1]))
            #         )
            # }
            # )


    def forward(self, x, tok = None):
        """ Forward pass through the Attention module.
            :param:
                x   -> torch.Tensor: Input tensor to transformer layers (S, embed_dim)
                tok -> torch.Tensor: Input tok tensor (N, T, embed_dim)
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        batch_size, seq_length, _ = x.shape

        # Self Attention
        q = F.linear(x, self.proj_q, self.bias_q) # (N, S, embed_dim)
        k = F.linear(x, self.proj_k, self.bias_k) # (N, T, embed_dim)
        v = F.linear(x, self.proj_v, self.bias_v) # (N, T, embed_dim)
        if self.add_lora:
            q += F.linear(x, self.lora_weight_proj_q["A"] @ self.lora_weight_proj_q["B"])  * self.lora_scaling
            k += F.linear(x, self.lora_weight_proj_k["A"] @ self.lora_weight_proj_k["B"])  * self.lora_scaling
            # v += F.linear(x, self.lora_weight_proj_v["A"] @ self.lora_weight_proj_v["B"])  * self.lora_scaling

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3) # batch, head, seq, dim
        v = v.permute(0, 2, 1, 3)

        attention = F.scaled_dot_product_attention(q, k, v)

        if tok is not None: 
            tok = tok.expand(batch_size, *tok.shape)
            
            # Prefix Cross Attention
            kp = F.linear(tok, self.proj_k, self.bias_k) # (N, T, embed_dim)
            vp = F.linear(tok, self.proj_v, self.bias_v) # (N, T, embed_dim)        
            kp = kp.reshape(batch_size, tok.shape[1], self.num_heads, self.head_dim)
            vp = vp.reshape(batch_size, tok.shape[1], self.num_heads, self.head_dim)
            kp = kp.permute(0, 2, 1, 3) # batch, head, seq, dim
            vp = vp.permute(0, 2, 1, 3)
            cross_attn = F.scaled_dot_product_attention(q, kp, vp)
            attention = attention + self.adapter_gate*cross_attn
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        attention_value = self.mha.out_proj(attention) # Includes weight and bias
        return attention_value, attention
 
class TransformerEncoderV2(TransformerEncoder):
    """
    Drop in for the TransformerEncoder module
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., peft_type = None, num_tokens = None,  r = None, alpha = None):
        super().__init__(dim, depth, heads, mlp_dim, dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionV2(dim, heads, dropout=dropout, peft_type = peft_type, r = r, alpha = alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.peft_type = peft_type
        # if peft_type == "aligner":
        if "aligner" in peft_type:
            self.adapter_wte = nn.Embedding(num_tokens, dim)
        # if peft_type == "adapter":
        if "adapter" in peft_type:
            self.adapter_wte = nn.ModuleList([])
            for _ in range(depth):
                self.adapter_wte.append(nn.Embedding(num_tokens, dim))
        self.num_tokens = num_tokens
        self.dim = dim

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for idx, (attn, ff) in enumerate(self.layers):
            tok = None
            # if self.peft_type == "aligner":
            if "aligner" in self.peft_type:
                tok = self.adapter_wte.weight
            # if self.peft_type == "adapter":
            if "adapter" in self.peft_type:
                tok = self.adapter_wte[idx].weight
            
            attention_value, attention_weight = attn(x, tok = tok)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn


class MaskTransformerV2(MaskTransformer):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000, peft_type = None, num_tokens = None, r = None, alpha = None):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """
        super().__init__(img_size, hidden_dim, codebook_size, depth, heads, mlp_dim, dropout, nclass)
        self.transformer = TransformerEncoderV2(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, peft_type = peft_type, num_tokens = num_tokens, r = r, alpha = alpha)

class MaskGIT_PEFT(MaskGIT):
    """
    Trainer for pre-trained MaskGIT with aligner. Freezes maskgit layers and only trains on prefix tokens. 
    """
    def __init__(self, args):
        super().__init__(args)
        self.freeze_layers()
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = MaskTransformerV2(
                img_size=self.args.img_size, hidden_dim=768, codebook_size=1024, depth=24, heads=16, mlp_dim=3072, dropout=0.1, peft_type = self.args.peft_type, num_tokens = self.args.num_prefix_tok, r = self.args.r, alpha = self.args.alpha    # Small
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

    def train_one_epoch(self, log_iter=100):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        if self.args.objective == "recon":
            for x, y in bar:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

                # Drop xx% of the condition for cfg
                drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

                
                # VQGAN encoding to img tokens
                with torch.no_grad():
                    emb, _, [_, _, code] = self.ae.encode(x)
                    code = code.reshape(x.size(0), self.patch_size, self.patch_size)
                if self.args.random_input:
                    random_input = torch.randint(0, 1024, code.shape, dtype = code.dtype, device = self.args.device)
                    masked_code, mask = self.get_mask_code(random_input, mode = self.args.sched_mode, value=self.args.mask_value)
                else:
                # Mask the encoded tokens
                    masked_code, mask = self.get_mask_code(code, mode = self.args.sched_mode, value=self.args.mask_value)
                # masked_code, mask = self.get_mask_code(random_input, value=self.args.mask_value)
                with torch.cuda.amp.autocast():                             # half precision
                    # pred = self.vit(random_input, y, drop_label = drop_label)
                    pred = self.vit(masked_code, y, drop_label=drop_label)  # The unmasked tokens prediction
                    # Cross-entropy loss
                    loss = self.criterion(pred.reshape(-1, 1024 + 1), code.view(-1)) / self.args.grad_cum

                # update weight if accumulation of gradient is done
                update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
                if update_grad:
                    self.optim.zero_grad()

                self.scaler.scale(loss).backward()  # rescale to get more precise loss

                if update_grad:
                    self.scaler.unscale_(self.optim)                      # rescale loss
                    nn.utils.clip_grad_norm_(self.vit.parameters(), 1)  # Clip gradient
                    self.scaler.step(self.optim)
                    self.scaler.update()

                cum_loss += loss.cpu().item()
                window_loss.append(loss.data.cpu().numpy().mean())
                # logs
                if update_grad and self.args.is_master:
                    self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

                if self.args.iter % log_iter == 0 and self.args.is_master:
                    # Generate sample for visualization
                    gen_sample = self.sample(nb_sample=10)[0]
                    gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                    # Show reconstruction                  
                    unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                    reco_sample = self.reco(x=x[:8], code=code[:8], unmasked_code=unmasked_code[:8], mask=mask[:10])
                    reco_sample = vutils.make_grid(reco_sample.data, nrow=8, padding=2, normalize=True)
                    self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)


                    # Save Network
                    self.save_network(model=self.vit, path=self.args.writer_log+f"/{self.args.model_version}_current.pth",
                                    iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
                self.args.iter += 1
        elif self.args.objective == "transform":
            for x, y, label in bar:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                label = label.to(self.args.device)
                x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN
                y = 2 * y - 1
                # Drop xx% of the condition for cfg
                drop_label = torch.empty(label.size()).uniform_(0, 1) < self.args.drop_label

                # VQGAN encoding to img tokens
                with torch.no_grad():
                    emb, _, [_, _, x_code] = self.ae.encode(x)
                    x_code = x_code.reshape(x.size(0), self.patch_size, self.patch_size)
                    emb_y, _, [_, _, y_code] = self.ae.encode(y)
                    y_code = y_code.reshape(y.size(0), self.patch_size, self.patch_size)
                
                # Mask the encoded tokens
                # masked_code, mask = self.get_mask_code(code, mode = self.args.sched_mode, value=self.args.mask_value)
                # masked_code, mask = self.get_mask_code(random_input, value=self.args.mask_value)
                
                with torch.cuda.amp.autocast():                             # half precision
                    pred = self.vit(x_code, label, drop_label=drop_label)  # The unmasked tokens prediction
                    # Cross-entropy loss
                    loss = self.criterion(pred.reshape(-1, 1024 + 1), y_code.view(-1)) / self.args.grad_cum

                # update weight if accumulation of gradient is done
                update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
                if update_grad:
                    self.optim.zero_grad()

                self.scaler.scale(loss).backward()  # rescale to get more precise loss

                if update_grad:
                    self.scaler.unscale_(self.optim)                      # rescale loss
                    nn.utils.clip_grad_norm_(self.vit.parameters(), 1)  # Clip gradient
                    # for name, params in self.vit.named_parameters():
                    #     if params.requires_grad:
                    #         if params.grad.isnan().sum().item() != 0:
                    #             raise ValueError(f"NaN in gradient layer {name}")
                    self.scaler.step(self.optim)
                    self.scaler.update()

                cum_loss += loss.cpu().item()
                window_loss.append(loss.data.cpu().numpy().mean())
                # logs
                if update_grad and self.args.is_master:
                    self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

                if self.args.iter % log_iter == 0 and self.args.is_master:
                    # Generate sample for visualization
                    gen_sample = self.sample(nb_sample=10)[0]
                    gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                    # Show reconstruction
                    unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                    prob = torch.softmax(pred, -1)
                    # Sample the code from the softmax prediction
                    distri = torch.distributions.Categorical(probs=prob, validate_args=False)
                    pred_code = distri.sample()
                    reco_sample = self.restyle(x, y, pred_code)
                    reco_sample = vutils.make_grid(reco_sample.data, nrow=8, padding=2, normalize=True)
                    self.log_add_img("Images/Restyle", reco_sample, self.args.iter)

                    # Save Network
                    self.save_network(model=self.vit, path=self.args.writer_log+f"/{self.args.model_version}_current.pth",
                                    iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

                self.args.iter += 1

        return cum_loss / n
        
    def freeze_layers(self):
        for name, params in self.vit.named_parameters():
            params.requires_grad = "adapter" in name or "lora" in name

    def restyle(self, x=None, y=None, pred=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x, y]
        with torch.no_grad():
            pred = pred.view(pred.size(0), self.patch_size, self.patch_size).type(torch.LongTensor).to(self.args.device)
            __y = self.ae.decode_code(torch.clamp(pred, 0, 1023))
            l_visual.append(__y)

        return torch.cat(l_visual, dim=0)

    def get_data(self):
        if self.args.data == "wikiart":

            train_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, objective = self.args.objective, split = "train")
            val_dataset = ArtDataset(self.args.data_folder, self.args.artist, self.args.num_train_images, self.args.img_size, objective = self.args.objective, split = "val")
            
            
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
        
        elif self.args.data == "image_pair":
            
            train_dataset = ImagePairDataset(self.args.data_folder, self.args.num_classes, self.args.img_per_class, split = "train", objective = self.args.objective, transform = None)
            val_dataset = ImagePairDataset(self.args.data_folder, self.args.num_classes, self.args.img_per_class, split = "val", objective= self.args.objective)
            
            train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.args.is_multi_gpus else None
            test_sampler = DistributedSampler(val_dataset, shuffle=True) if self.args.is_multi_gpus else None
            train_loader = DataLoader(train_dataset, batch_size=self.args.bsize,
                                    shuffle=False if self.args.is_multi_gpus else True,
                                    num_workers=self.args.num_workers, pin_memory=True,
                                    drop_last=True, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=self.args.bsize, shuffle = False, num_workers=self.args.num_workers, pin_memory=True,
                                    drop_last=True, sampler=test_sampler)
            return train_loader, val_loader
        
class MaskGITWithDiscriminator(MaskGIT_PEFT):
    def __init__(self, args):
        super().__init__(args)
        self.discriminator = self.get_discriminator()
        self.criterion_d = GANLoss(gan_mode = 'wgangp')
        self.optimizer_D = self.get_optim(self.discriminator, self.args.lr, betas=(0.9, 0.96))

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.initial_epoch + self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            train_loss_g, train_loss_d = self.train_one_epoch()

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss_g = self.all_gather(train_loss_g, torch.cuda.device_count())
                train_loss_d = self.all_gather(train_loss_d, torch.cuda.device_count())

            # Save model
            if e % 25 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.writer_log+f"/{self.args.model_version}_epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Generator Train/GlobalLoss', train_loss_g, self.args.global_epoch)
                self.log_add_scalar('Discriminator Train/GlobalLoss', train_loss_d, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Generator Loss {train_loss_g:.4f},"
                      f" Discriminator Loss {train_loss_d:.4f}"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1
        
    def get_discriminator(self, input_nc = 2, ndf = 32, n_layers = 3):
        model = NLayerDiscriminator(input_nc, ndf, n_layers)
        # model = PixelDiscriminator(input_nc, ndf)
        if self.args.discriminator_folder is not None:
            checkpoint = torch.load(self.args.discriminator_folder, map_location='cpu')
            # Update the current epoch and iteration
            self.args.iter += checkpoint['iter']
            self.args.global_epoch += checkpoint['global_epoch']
            self.args.initial_epoch = self.args.global_epoch
            # Load network
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model.to(self.args.device)
        
    
    def set_requires_grad(self, nets, requires_grad = False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def loss_D(self, fake_xy, real_xy):
        pred_fake = self.discriminator(fake_xy.detach())
        loss_d_fake = self.criterion_d(pred_fake, False)

        pred_real = self.discriminator(real_xy.type(torch.float16))
        loss_d_real = self.criterion_d(pred_real, True)

        loss = 0.5* (loss_d_fake + loss_d_real)
        return loss

        
    def loss_G(self, fake_xy, pred_logits, y):
        pred_fake = self.discriminator(fake_xy)
        loss_g = self.criterion(pred_logits.reshape(-1, 1024 + 1), y.view(-1))
        loss_d = self.criterion_d(pred_fake, True)
        if self.args.loss_lambda is None:
            loss_lambda = loss_g/loss_d
            loss = loss_g + loss_lambda * loss_d
        else:
        #  loss_L1 = self.criterionL1(pred_img, y) * self.args.lambda_L1
            loss = loss_g + self.args.loss_lambda * loss_d
        return loss


    def train_one_epoch(self, log_iter=10):
        """ Train the model for 1 epoch """
        self.vit.train()
        self.discriminator.train()
        cum_loss_d = 0.
        cum_loss_g = 0.
        window_loss_d = deque(maxlen=self.args.grad_cum)
        window_loss_g = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        if self.args.objective == "recon":
            for x, y in bar:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

                # Drop xx% of the condition for cfg
                drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

                
                # VQGAN encoding to img tokens
                with torch.no_grad():
                    emb, _, [_, _, code] = self.ae.encode(x)
                    code = code.reshape(x.size(0), self.patch_size, self.patch_size)
                if self.args.random_input:
                    random_input = torch.randint(0, 1024, code.shape, dtype = code.dtype, device = self.args.device)
                    masked_code, mask = self.get_mask_code(random_input, mode = self.args.sched_mode, value=self.args.mask_value)
                else:
                # Mask the encoded tokens
                    masked_code, mask = self.get_mask_code(code, mode = self.args.sched_mode, value=self.args.mask_value)
                # masked_code, mask = self.get_mask_code(random_input, value=self.args.mask_value)
                with torch.cuda.amp.autocast():                             # half precision
                    # pred = self.vit(random_input, y, drop_label = drop_label)
                    pred = self.vit(masked_code, y, drop_label=drop_label)  # The unmasked tokens prediction
                    # Cross-entropy loss
                    loss = self.criterion(pred.reshape(-1, 1024 + 1), code.view(-1)) / self.args.grad_cum

                # update weight if accumulation of gradient is done
                update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
                if update_grad:
                    self.optim.zero_grad()

                self.scaler.scale(loss).backward()  # rescale to get more precise loss

                if update_grad:
                    self.scaler.unscale_(self.optim)                      # rescale loss
                    nn.utils.clip_grad_norm_(self.vit.parameters(), 1)  # Clip gradient
                    self.scaler.step(self.optim)
                    self.scaler.update()

                cum_loss += loss.cpu().item()
                window_loss.append(loss.data.cpu().numpy().mean())
                # logs
                if update_grad and self.args.is_master:
                    self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

                if self.args.iter % log_iter == 0 and self.args.is_master:
                    # Generate sample for visualization
                    gen_sample = self.sample(nb_sample=10)[0]
                    gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                    # Show reconstruction                  
                    unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                    reco_sample = self.reco(x=x[:8], code=code[:8], unmasked_code=unmasked_code[:8], mask=mask[:10])
                    reco_sample = vutils.make_grid(reco_sample.data, nrow=8, padding=2, normalize=True)
                    self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)


                    # Save Network
                    self.save_network(model=self.vit, path=self.args.writer_log+f"/{self.args.model_version}_current.pth",
                                    iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

        elif self.args.objective == "transform":
            for x, y, label in bar:
                # Encode images into codebook vectors
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                label = label.to(self.args.device)
                x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN
                y = 2 * y - 1
                # Drop xx% of the condition for cfg
                drop_label = torch.empty(label.size()).uniform_(0, 1) < self.args.drop_label

                # VQGAN encoding to img tokens
                with torch.no_grad():
                    emb, _, [_, _, x_code] = self.ae.encode(x)
                    emb_y, _, [_, _, y_code] = self.ae.encode(y)
                    x_code = x_code.reshape(x.size(0), self.patch_size, self.patch_size)
                    y_code = y_code.reshape(y.size(0), self.patch_size, self.patch_size)
                
                # Generate new images with generator
                with torch.cuda.amp.autocast():                             # half precision
                    logits = self.vit(x_code, label, drop_label=drop_label)  # The unmasked tokens prediction
                    # Sample the code from the softmax prediction
                    ctg = torch.linspace(0, 1024, 1025, device = self.args.device)
                    pred = F.gumbel_softmax(logits, tau = 100, dim = -1, hard = True)
                    pred = pred @ ctg
                    pred = pred.view(pred.shape[0], self.args.patch_size, self.args.patch_size)
                    pred = torch.clamp(pred, 0, 1023)
                    # pred_img = self.ae.decode_code(torch.clamp(pred.type(torch.LongTensor), 0, 1023))
                    # update discriminator
                    # real_xy = torch.cat(x, y, 1)
                    # fake_xy = torch.cat(x, pred_img, 1)
                    real_xy = torch.cat((x_code.unsqueeze(1), y_code.unsqueeze(1)), 1)
                    fake_xy = torch.cat((x_code.unsqueeze(1), pred.unsqueeze(1)), 1)
                    self.set_requires_grad(self.discriminator, True)
                    loss_d = self.loss_D(fake_xy, real_xy) / self.args.grad_cum
                    
                    # Cross-entropy loss
                   
                
                # update weight if accumulation of gradient is done
                # Discriminator update
                update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
                if update_grad:
                    self.optimizer_D.zero_grad()
                    # self.optim.zero_grad()

                self.scaler.scale(loss_d).backward()  # rescale to get more precise loss
                if update_grad:
                    self.scaler.unscale_(self.optimizer_D)                      # rescale loss
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)  # Clip gradient
                    self.scaler.step(self.optimizer_D)
                    self.scaler.update()

                # Generator update
                with torch.cuda.amp.autocast():
                    self.set_requires_grad(self.discriminator, False)
                    # loss_G = self.loss_G(fake_xy, pred_img, y)
                    loss_G = self.loss_G(fake_xy, logits, y_code)

                if update_grad:
                    self.optim.zero_grad()
                
                self.scaler.scale(loss_G).backward()
                if update_grad:
                    self.scaler.unscale_(self.optim)                      # rescale loss
                    nn.utils.clip_grad_norm_(self.vit.parameters(), 1)  # Clip gradient
                    self.scaler.step(self.optim)
                    self.scaler.update()

                cum_loss_g += loss_G.cpu().item()
                cum_loss_d += loss_d.cpu().item()
                window_loss_g.append(loss_G.data.cpu().numpy().mean())
                window_loss_d.append(loss_d.data.cpu().numpy().mean())
                # logs
                if update_grad and self.args.is_master:
                    self.log_add_scalar('Generator Train/Loss', np.array(window_loss_g).sum(), self.args.iter)
                    self.log_add_scalar('Discriminator Train/Loss', np.array(window_loss_d).sum(), self.args.iter)

                if self.args.iter % log_iter == 0 and self.args.is_master:
                    # Generate sample for visualization
                    gen_sample = self.sample(nb_sample=10)[0]
                    gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                    # Show reconstruction

                    reco_sample = self.restyle(x, y, pred)
                    reco_sample = vutils.make_grid(reco_sample.data, nrow=8, padding=2, normalize=True)
                    self.log_add_img("Images/Restyle", reco_sample, self.args.iter)

                    # Save Network
                    self.save_network(model=self.vit, path=self.args.writer_log+f"/{self.args.model_version}_current.pth",
                                    iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
                    self.save_network(model=self.discriminator, path=self.args.writer_log+f"/discriminator_current.pth",
                                    iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            self.args.iter += 1

        return cum_loss_g / n, cum_loss_d / n

