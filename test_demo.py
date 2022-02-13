from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modules import *
import torchvision.transforms as transforms
import socket
import numpy as np
from datasets import get_theta
from util import PSNR, SSIM, rgb2ycbcr
from os.path import join
import time
import json
import lpips


# Training settings
parser = argparse.ArgumentParser(description='PyTorch See360 model')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./data/real_world')
parser.add_argument('--test_set', type=str, default='British') # British, Louvre, Manhattan, Parliament
parser.add_argument('--save_dir', default='result/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='record/', help='Location to save checkpoint models')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    # transforms.Lambda(lambda x: x.mul(255))
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
)


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')


file_path = join(opt.data_dir, opt.test_set)
save_path = join(opt.save_dir, opt.test_set)


print('===> Building model ')

model = generator_final_v2(input_dim=3, dim=64)


model = torch.nn.DataParallel(model)

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

if cuda:
    model = model.cuda(gpus_list[0])
    loss_fn_alex = loss_fn_alex.cuda(gpus_list[0])

model.eval()
loss_fn_alex.eval()

def eval(angle_left, angle_right, angle_target):

    model_name = 'models/' + opt.test_set + '/GAN_final.pth'

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        # print('===> read model as: ', model_name)

    theta = int(12 / np.abs(angle_right - angle_left) * (angle_target - angle_left))
    code = F.one_hot(torch.tensor(theta), num_classes=12).float()

    if angle_right == 360:
        angle_right = 0
    img1 = Image.open(file_path + '/' + 'img1_crop.png').convert('RGB')
    img2 = Image.open(file_path + '/' + 'img2_crop.png').convert('RGB')

    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    img1 = img1.cuda(gpus_list[0])
    img2 = img2.cuda(gpus_list[0])

    with torch.no_grad():
        img1 = 2.0 * (img1 - 0.5)
        img2 = 2.0 * (img2 - 0.5)

        code = code.view(img1.shape[0], 12).cuda()

        output = model(img1, img2, code)

        output = output * 0.5 + 0.5

    output = output.data[0].cpu().permute(1, 2, 0)

    output = output * 255
    output = output.clamp(0, 255)

    out_name = save_path + '/' + str(angle_target) + '.png'
    Image.fromarray(np.uint8(output)).save(out_name)



##########
## Done ##
##########

##Eval Start!!!!
eval(angle_left=0, angle_right=60, angle_target=30)




