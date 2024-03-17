# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group

from Network.aligner import MaskGIT_PEFT, MaskGITWithDiscriminator
import datetime


import datetime
import json

class Args:
    def __init__(self,
                model_version,
                discriminator = True, 
                peft_type = None,
                random_input = False,
                data_folder="./models/mask-aligner/dataset/wikiart/",                         
                vit_folder="./models/mask-aligner/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth",
                vqgan_folder="./models/mask-aligner/pretrained_maskgit/VQGAN/",
                discriminator_folder = None,
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
                loss_lambda = 100, 
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
        self.discriminator_folder = discriminator_folder
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
        self.loss_lambda = loss_lambda
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

    
def main(args):
    """ Main function:Train or eval MaskGIT """
    if args.discriminator:
        maskgit = MaskGITWithDiscriminator(args) 
    else:
        maskgit = MaskGIT_PEFT(args) 
    # maskgit.nclasses = 1010

    if args.test_only:  # Evaluate the networks
        maskgit.eval()

    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            labels, name = [1, 2, 3, 4, 5, 6, 7, 8, 9, random.randint(0, 10)] * 1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            sm_temp = 1.3          # Softmax Temperature
            r_temp = 7             # Gumbel Temperature
            w = 9                  # Classifier Free Guidance
            randomize = "linear"   # Noise scheduler
            step = 32              # Number of step
            sched_mode = "arccos"  # Mode of the scheduler
            # Generate sample
            gen_sample, _, _ = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                              randomize=randomize, sched_mode=sched_mode, step=step)
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            save_image(gen_sample, f"./models/mask-aligner/saved_img/sched_{sched_mode}_step={step}_temp={sm_temp}"
                                   f"_w={w}_randomize={randomize}_{name}.jpg")
    else:  # Begin training
        maskgit.fit()
        args.to_json()
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            samples = []
            batch = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [1, 7, 282, 604, 724, 179, 681, 367, 850, 999]]
            for labels in batch:
                labels = labels * 1
                name = "r_row"
                labels = torch.LongTensor(labels).to(args.device)
                sm_temp = 1.3          # Softmax Temperature
                r_temp = 7             # Gumbel Temperature
                w = 9                  # Classifier Free Guidance
                randomize = "linear"   # Noise scheduler
                step = 32              # Number of step
                sched_mode = "arccos"  # Mode of the scheduler
                # Generate sample
                gen_sample, _, _ = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                                randomize=randomize, sched_mode=sched_mode, step=step)
                samples.append(gen_sample.float().cpu())
            samples = torch.cat(samples, 0)
            samples = vutils.make_grid(samples, nrow=10, padding=2, size = (18, 18), normalize=True)
           
            # Save image
            save_image(samples, f"{args.writer_log}results.jpg")
        


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import matplotlib.pyplot as plt
import torchvision.utils as vutils

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
    plt.figure(figsize = size)
    plt.axis('off')
    plt.imshow(x.permute(1, 2, 0))
    if save is not None:
        print("saving..")
        # print(f"/home/joseph/linProject{args.writer_log[1:]}{save}")
        plt.savefig(f"/home/joseph/linProject{args.writer_log[1:]}{save}")
    plt.show()

def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    params = [
        ## Next 
        # {
        #     "style": "cyborg",
        #     "peft": ["aligner", "lora"],
        #     "objective": "transform",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 800,
        #     "r": 8,
        #     "epochs": 35
        # },
        # {
        #     "style": "cyborg",
        #     "peft": ["adapter", "lora"],
        #     "objective": "transform",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 800,
        #     "r": 8,
        #     "epochs": 35
        # },
        # {
        #     "style": "cyborg",
        #     "peft": ["aligner", "lora"],
        #     "objective": "transform",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 800,
        #     "r": 24,
        #     "epochs": 35
        # },
        # {
        #     "style": "cyborg",
        #     "peft": ["adapter", "lora"],
        #     "objective": "transform",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 800,
        #     "r": 24,
        #     "epochs": 35
        # },
        ## Next

        {
            "style": "colorfulcomic",
            "peft": ["lora"],
            "objective": "recon",
            "N": 3000,
            "N_class": 20,
            "total_images": 800,
            "r": 8,
            "epochs": 35
        },
        # {
        #     "style": "cyborg",
        #     "peft": ["aligner"],
        #     "objective": "transform",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 800,
        #     "r": 24,
        #     "epochs": 35
        # },
        
        # {
        #     "style": "comic",
        #     "peft": "adapter",
        #     "objective": "recon",
        #     "N": 300,
        #     "N_class": 20,
        #     "total_images": 400,
        #     "r": 48
        # },
        # {
        #     "style": "comic",
        #     "peft": "adapter",
        #     "objective": "recon",
        #     "N": 20,
        #     "N_class": 100,
        #     "total_images": 400,
        #     "r": 48
        # },
        # {
        #     "style": "comic",
        #     "peft": "adapter",
        #     "objective": "recon",
        #     "N": 300,
        #     "N_class": 100,
        #     "total_images": 400,
        #     "r": 48
        # }
    ]
    for param in params: 
        dataset_type = "image_pair"
        withDisc = False
        style = param["style"]
        peft_type = param["peft"]
        objective = param["objective"]

        model_version = f"{dataset_type}_{style}_{peft_type[0]}_{objective}" 
        if withDisc:
            model_version += "_withDisc"
        # if peft_type == "lora":
        #     model_version += "qk_only"
        args = Args(
                    model_version=model_version,
                    discriminator=withDisc, 
                    peft_type= peft_type,
                    random_input = False,                         
                    vit_folder="./models/mask-aligner/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth",
                    # vit_folder = "./models/mask-aligner/logs/image_pair_comic_adapter_transform/20240221_222043/image_pair_comic_adapter_transform_current.pth",
                    vqgan_folder="./models/mask-aligner/pretrained_maskgit/VQGAN/",
                    discriminator_folder = None,
                    writer_log=f"./models/mask-aligner/logs/",
                    save_folder = "./models/mask-aligner/finetuned_maskgit/",
                    # data_folder="./models/mask-aligner/dataset/wikiart/",
                    # data = "wikiart",
                    # artist= "Albrecht_Durer",
                    num_train_images= 96,
                    data_folder = f"./data_gen/images/imagenet/subset/convert2{style}/",
                    data = dataset_type,
                    num_classes = param["N_class"],
                    img_per_class  = int(param["total_images"]/param["N_class"]),
                    objective = objective,
                    img_size = 256, # Size of the image
                    r = param["r"], # only for LoRA
                    alpha = 1, # only for LoRA
                    num_prefix_tok= param["N"],
                    mask_value = 1024, # Masked token value                                  
                    epoch = param["epochs"],
                    bsize = 8, # batch size
                    sched_mode = "arccos",
                    seed = 1,                    # Seed for reproducibility
                    channel = 3,                 # Number of input channel
                    num_workers = 4,            # Number of workers
                    iter = 0,                    # 750_000 at 256*256 + 750_000 at 512*512
                    global_epoch = 0,                    # 300 epoch w/ bsize 512 + 80 epoch with bsize 128
                    lr = 5e-4,                        # Learning rate 
                    loss_lambda = 100,
                    grad_cum = 1,
                    drop_label = 0.1,           # Drop out label for cfg
                    resume = True,          # Set to True for loading model
                    debug = False,          # Load only the model (not the dataloader)
                    test_only = False,        # Dont launch the testing
                    is_master = True,           # Master machine 
                    is_multi_gpus = False,             # set to False for colab demo
                    is_finetuned = False
        )
        if args.seed > 0: # Set the seed for reproducibility
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.deterministic = True

        world_size = torch.cuda.device_count()

        if world_size > 1:  # launch multi training
            print(f"{world_size} GPU(s) found, launch multi-gpus training")
            args.is_multi_gpus = True
            launch_multi_main(args)
        else:  # launch single Gpu training
            print(f"{world_size} GPU found")
            args.is_master = True
            args.is_multi_gpus = False
            main(args)
        
