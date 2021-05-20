"""  Test with pretrained models

We train two models, one for fixed data, one for fresh data. 

For testing
0: Put the testing images under corresponding folder.
    For fixed data, put testing images under /data/Fixed/test/A
    For fresh data, put testing images under /data/Fresh/test/A
1: Select the checkpoint 
   You can use the default.

2: run the testing

Example 
        python test.py --dataset_name Fixed

        python test.py --dataset_name Fresh

Results are in test_results folder.

"""

import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

# import functions from self-written packages
from models import *
from datasets import *

# Options for the training
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Fixed", help="name of the dataset")
parser.add_argument("--checkpoint_epoch", type=int, default="90", help=" Checkpoint epoch to reload")
parser.add_argument("--img_height", type=int, default=256, help="image height in training")
parser.add_argument("--img_width", type=int, default=256, help="image width in training")
parser.add_argument("--input_nc", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator, same as training")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("test_results/%s" % opt.dataset_name, exist_ok=True)
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Reload Model

input_shape = (opt.input_nc, opt.img_height, opt.img_width)
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.checkpoint_epoch)))
G_AB.eval()


Tensor = torch.cuda.FloatTensor

# Test data loader

Test_dataloader = DataLoader(
    ImageDatasetSingle("data/%s" % opt.dataset_name),
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

files_A = sorted(glob.glob(os.path.join("data/"+opt.dataset_name + "/test/A/*.*")))

for i, batch in enumerate(Test_dataloader):

    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    fake_B = G_AB(real_A)
    files_AS = files_A[i]
    print(i)
    filename = files_AS.replace('data/%s/test/A/'% opt.dataset_name,'')
    save_image(fake_B, 'test_results/%s/'% opt.dataset_name + filename, normalize=True)
print("saved in /test_results/%s"%opt.dataset_name)