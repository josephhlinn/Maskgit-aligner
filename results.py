# %% [markdown]
# # MaskGIT PEFT Result Analysis

# %%
import os
import random
import math

import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from Trainer.vit import MaskGIT
from Network.aligner import MaskGIT_PEFT, MaskGITWithDiscriminator


# %%
import json

# %%
save_paths = [
  "/home/joseph/linProject/models/mask-aligner/logs/image_pair_cyborg_aligner_recon/20240312_185917",
  "/home/joseph/linProject/models/mask-aligner/logs/image_pair_cyborg_adapter_recon/20240313_090807",
]

# %% [markdown]
# # MaskGIT initialisation

# %%
import datetime
sm_temp = 1.3          # Softmax Temperature
r_temp = 7             # Gumbel Temperature
w = 5                # Classifier Free Guidance

class Args:
    def __init__(self,
                model_version = "maskgit",
                discriminator = False, 
                peft_type = None,
                random_input = False,
                data_folder="./models/mask-aligner/dataset/wikiart/",                         
                vit_folder="./models/mask-aligner/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth",
                vqgan_folder="./models/mask-aligner/pretrained_maskgit/VQGAN/",
                writer_log=f"./models/mask-aligner/logs/",
                save_folder = "./models/mask-aligner/finetuned_maskgit/",
                data = "wikiart",
                artist= "Albrecht_Durer",
                num_classes = 10,
                img_per_class  = 10,
                objective = "transform",
                num_train_images= 96,
                img_size = 256, # Size of the image
                r = None,  # only for LoRA
                alpha = None, # only for LoRA
                num_prefix_tok= 10,
                mask_value = 1024, # Masked token value                                  
                epoch = 10,
                bsize = 8, # batch size
                sched_mode = "arccos",
                seed = 1,                    # Seed for reproducibility
                channel = 3,                 # Number of input channel
                num_workers = 4,            # Number of workers
                iter = 0,                    # 750_000 at 256*256 + 750_000 at 512*512
                global_epoch = 0,                    # 300 epoch w/ bsize 512 + 80 epoch with bsize 128
                lr = 1e-4,                        # Learning rate 
                lambda_L1 = 0.1, 
                grad_cum = 1,
                drop_label = 0.1,           # Drop out label for cfg
                device = "cuda" if torch.cuda.is_available() else "cpu",
                resume = True,          # Set to True for loading model
                debug = False,          # Load only the model (not the dataloader)
                test_only = False,        # Dont launch the testing
                is_master = True,           # Master machine 
                is_multi_gpus = False,             # set to False for colab demo
                is_finetuned = False
                 ):
        super().__init__()
        self.model_version = model_version
        self.discriminator = discriminator
        self.peft_type = peft_type
        self.random_input = random_input
        # Folders
        self.data_folder = data_folder                   
        self.vit_folder= vit_folder
        self.vqgan_folder= vqgan_folder
        self.writer_log = f"{writer_log}{model_version}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        self.save_folder = save_folder

        # Data setup
        self.data = data
        self.artist= artist
        self.num_train_images= num_train_images
        self.num_classes = num_classes
        self.img_per_class  = img_per_class
        self.objective = objective

        # Model setup
        self.num_prefix_tok= num_prefix_tok
        self.mask_value = mask_value                                                            # Value of the masked token
        self.img_size = img_size                                                               # Size of the image
        self.patch_size = img_size // 16                                                   # Number of vizual token
        self.channel = channel                                                                # Number of input channel
        self.r = r
        self.alpha = alpha

        # Training setup
        self.epoch = epoch
        self.bsize = bsize # batch size
        self.sched_mode = sched_mode
        self.seed = seed                                                                  # Seed for reproducibility
        self.num_workers = num_workers                                                              # Number of workers
        self.iter = iter                                                             # 750_000 at 256*256 + 750_000 at 512*512
        self.global_epoch = global_epoch                                                           # 300 epoch w/ bsize 512 + 80 epoch with bsize 128
        self.lr = lr                                                                    # Learning rate 
        self.lambda_L1 = lambda_L1
        self.grad_cum = grad_cum
        self.drop_label = drop_label                                                             # Drop out label for cfg
        self.resume = resume                                                                # Set to True for loading model
        self.device =  device       # Device
  
        self.debug = debug                                                                 # Load only the model (not the dataloader)
        self.test_only = test_only                                                            # Dont launch the testing
        self.is_master = is_master                                                             # Master machine 
        self.is_multi_gpus = is_multi_gpus                                                        # set to False for colab demo
        self.is_finetuned = is_finetuned
    
    def to_json(self):
        # js = json.dumps(self, default = lambda o: o.__dict__)
        args_dict = vars(args)
        with open(self.writer_log + "params.json", "w") as file:
            json.dump(args_dict, file, indent = 4) 

    def from_json(self, json_file):
        with open(json_file) as f:
            args_file = json.load(f)
        for key, value in args_file.items():
            setattr(self, key, value)                                                                  
for folder in os.listdir("/home/joseph/linProject/models/mask-aligner/logs"):
    if "cyborg" in folder:
        logs = os.path.join("/home/joseph/linProject/models/mask-aligner/logs", folder)
        save_paths = os.listdir(logs)
        save_paths = [os.path.join(logs, i) for i in save_paths]
        for save_path in save_paths:
            if f"results_cfg{w}.jpg" not in os.listdir(save_path):

                args = Args()

                # %%

                json_file = os.path.join(save_path, "params.json")
                args.from_json(json_file)

                for files in os.listdir(save_path):
                    if "_current.pth" in files and "discriminator" not in files:
                        model_path = os.path.join("/home/joseph/linProject", args.writer_log[1:], files)
                        args.vit_folder = "/home/joseph/linProject" + model_path
                        args.vqgan_folder = "/home/joseph/linProject" + args.vqgan_folder[1:]

                # %%
                args.debug = True

                # %%
                # Fixe seed
                if args.seed > 0:
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    random.seed(args.seed)
                    torch.backends.cudnn.enable = False
                    torch.backends.cudnn.deterministic = True

                # Instantiate the MaskGIT

                maskgit = MaskGIT_PEFT(args)

                # %%
                def viz(x, nrow=10, pad=2, size=(18, 18), save = None):
                    """
                    Visualize a grid of images.

                    Args:
                        x (torch.Tensor): Input images to visualize.
                        nrow (int): Number of images in each row of the grid.
                        pad (int): Padding between the images in the grid.
                        size (tuple): Size of the visualization figure.

                    """
                    nb_img = len(x)
                    min_norm = x.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
                    max_norm = x.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
                    x = (x - min_norm) / (max_norm - min_norm)

                    x = vutils.make_grid(x.float().cpu(), nrow=nrow, padding=pad, normalize=False)
                
                    print(f"saving../home/joseph/linProject{args.writer_log[1:]}{save}")
                    # print(f"/home/joseph/linProject{args.writer_log[1:]}{save}")
                    save_image(x, f"/home/joseph/linProject{args.writer_log[1:]}{save}")
                    

                def decoding_viz(gen_code, mask, maskgit):
                    """
                    Visualize the decoding process of generated images with associated masks.

                    Args:
                        gen_code (torch.Tensor): Generated code for decoding.
                        mask (torch.Tensor): Mask used for decoding.
                        maskgit (MaskGIT): MaskGIT instance.
                    """
                    start = torch.FloatTensor([1, 1, 1]).view(1, 3, 1, 1).expand(1, 3, maskgit.patch_size, maskgit.patch_size) * 0.8
                    end = torch.FloatTensor([0.01953125, 0.30078125, 0.08203125]).view(1, 3, 1, 1).expand(1, 3, maskgit.patch_size, maskgit.patch_size) * 1.4
                    code = torch.stack((gen_code), dim=0).squeeze()
                    mask = torch.stack((mask), dim=0).view(-1, 1, maskgit.patch_size, maskgit.patch_size).cpu()

                    with torch.no_grad():
                        x = maskgit.ae.decode_code(torch.clamp(code, 0, 1023))

                    binary_mask = mask * start + (1 - mask) * end
                    binary_mask = vutils.make_grid(binary_mask, nrow=len(gen_code), padding=1, pad_value=0.4, normalize=False)
                    binary_mask = binary_mask.permute(1, 2, 0)

                    plt.figure(figsize = (18, 2))
                    plt.gca().invert_yaxis()
                    plt.pcolormesh(binary_mask, edgecolors='w', linewidth=.5)
                    plt.axis('off')
                    plt.show()

                    viz(x, nrow=len(gen_code))

                # %%
                randomize = "linear"   # Noise scheduler
                step = 32              # Number of step
                sched_mode = "arccos"  # Mode of the scheduler
                batch = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [1, 7, 282, 604, 724, 179, 681, 367, 850, 999]]
                samples = []
                for labels in batch:
                    labels = labels * 1
                    name = "r_row"
                    labels = torch.LongTensor(labels).to(args.device)
                    # Generate sample
                    gen_sample, gen_code, l_mask = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, w=w, randomize=randomize, r_temp=r_temp, sched_mode=sched_mode, step=step)
                    samples.append(gen_sample)
                viz(torch.cat(samples, 0), nrow=10, size=(18, 18), save = f"results_cfg{w}.jpg")
                    
                # %%
                num_params = 0
                for name, params in maskgit.vit.named_parameters():
                    if params.requires_grad:
                        print(name)
                        if "lora_weight_proj_v" not in name:
                            num_params = num_params + len(params)
                print(num_params)

        # %%


    # %%



