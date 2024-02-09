# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group

from Network.aligner import MaskGITAligner, ConditionalMaskGITAligner
import datetime

class Args(argparse.Namespace):
    model_name=""
    #data_folder="./models/mask-aligner/dataset/sd_v1-4/"     
    data_folder = "./models/mask-aligner/dataset/wikiart/"                     
    vit_folder="./models/mask-aligner/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth"
    vqgan_folder="./models/mask-aligner/pretrained_maskgit/VQGAN/"
    writer_log=f"./models/mask-aligner/logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data = "wikiart"
    artist = "Albrecht_Durer"   #"cartoon_goldfish"
    num_prefix_tok = 10
    num_train_images = 96
    bsize = 4
    sched_mode= "arccos"
    grad_cum=1
    step = 8
    epoch = 10
    cfg_w = 3
    r_temp = 4.5
    sm_temp = 1
    mask_value = 1024                                                            # Value of the masked token
    img_size = 256                                                               # Size of the image
    path_size = img_size // 16                                                   # Number of vizual token
    seed = 1                                                                     # Seed for reproducibility
    channel = 3                                                                  # Number of input channel
    num_workers = 4                                                              # Number of workers
    iter = 0                                                             # 750_000 at 256*256 + 750_000 at 512*512
    global_epoch = 0                                                           # 300 epoch w/ bsize 512 + 80 epoch with bsize 128
    lr = 5e-5                                                                    # Learning rate 
    drop_label = 0                                                           # Drop out label for cfg
    resume = True                                                                # Set to True for loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # Device
    print(device)
    debug = False                                                                 # Load only the model (not the dataloader)
    test_only = False                                                            # Dont launch the testing
    is_master = True                                                             # Master machine 
    is_multi_gpus = False     
    
def main(args):
    """ Main function:Train or eval MaskGIT """
    maskgit = ConditionalMaskGITAligner(args) 
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


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data",         type=str,   default="wikiart", help="dataset on which dataset to train")
    # parser.add_argument("--artist",       type=str,   default="Albrecht_Durer", help="artist to fine-tune on")
    # parser.add_argument("--num_train_images", type=int, default=5,       help="number of images to train on")
    # parser.add_argument("--data-folder",  type=str,   default="",         help="folder containing the dataset")
    # parser.add_argument("--vqgan-folder", type=str,   default="./models/mask-aligner/pretrained_maskgit/VQGAN/",         help="folder of the pretrained VQGAN")
    # parser.add_argument("--vit-folder",   type=str,   default="./models/mask-aligner/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512.pth",         help="folder where to save the Transformer")
    # parser.add_argument("--writer-log",   type=str,   default="./models/mask-aligner/logs/",         help="folder where to store the logs, defaults to ./runs")
    # parser.add_argument("--sched_mode",   type=str,   default="arccos",   help="scheduler mode whent sampling")
    # parser.add_argument("--grad-cum",     type=int,   default=1,          help="accumulate gradient")
    # parser.add_argument('--channel',      type=int,   default=3,          help="rgb or black/white image")
    # parser.add_argument("--num_workers",  type=int,   default=8,          help="number of workers")
    # parser.add_argument("--step",         type=int,   default=8,          help="number of step for sampling")
    # parser.add_argument('--seed',         type=int,   default=42,         help="fix seed")
    # parser.add_argument("--epoch",        type=int,   default=5,          help="number of epoch")
    # parser.add_argument('--img-size',     type=int,   default=512,        help="image size")
    # parser.add_argument("--bsize",        type=int,   default=2,          help="batch size")
    # parser.add_argument("--mask-value",   type=int,   default=1024,       help="codebook mask token")
    # parser.add_argument("--lr",           type=float, default=1e-4,       help="learning rate to train the transformer")
    # parser.add_argument("--cfg_w",        type=float, default=3,          help="classifier free guidance wight")
    # parser.add_argument("--r_temp",       type=float, default=4.5,        help="Gumbel noise temperature when sampling")
    # parser.add_argument("--sm_temp",      type=float, default=1.,         help="temperature before softmax when sampling")
    # parser.add_argument("--drop-label",   type=float, default=1,          help="drop rate for cfg")
    # parser.add_argument("--num_prefix_tok", type=int, default=10,         help="number of prefix tokens for aligner")
    # parser.add_argument("--test-only",    action='store_true',            help="only evaluate the model")
    # parser.add_argument("--resume",       action='store_true',            help="resume training of the model")
    # parser.add_argument("--debug",        action='store_true',            help="debug")
    # args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.iter = 0
    # args.global_epoch = 0
    # args.resume = True
    args = Args()
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
        
