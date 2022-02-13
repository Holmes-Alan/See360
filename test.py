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
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--test_set', type=str, default='UrbanCity360') # UrbanCity360 or Archinterior360
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


file_path = join(opt.data_dir, opt.test_set, 'test_data_v2')
save_path = join(opt.save_dir, opt.test_set)

f = open(join(opt.data_dir, opt.test_set) + '/test_record_v1.json')
test_param = json.load(f)

print('===> Building model ')

model = generator_final(input_dim=3, dim=64)

model = torch.nn.DataParallel(model)

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

if cuda:
    model = model.cuda(gpus_list[0])
    loss_fn_alex = loss_fn_alex.cuda(gpus_list[0])

model.eval()
loss_fn_alex.eval()

def eval(angle_left, angle_right, angle_target, x, y):

    model_name = 'models/' + opt.test_set + '/GAN_final.pth'

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        # print('===> read model as: ', model_name)

    theta = int(12 / np.abs(angle_right - angle_left) * (angle_target - angle_left))
    code = F.one_hot(torch.tensor(theta), num_classes=12).float()

    if angle_right == 360:
        angle_right = 0
    img1 = Image.open(file_path + '/' + str(x) + '_' + str(y) + '_' + str(angle_left) + '.png').convert('RGB')
    img2 = Image.open(file_path + '/' + str(x) + '_' + str(y) + '_' + str(angle_right) + '.png').convert('RGB')
    GT = Image.open(file_path + '/' + str(x) + '_' + str(y) + '_' + str(angle_target) + '.png').convert('RGB')

    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
    GT_s = transform(GT).unsqueeze(0)

    img1 = img1.cuda(gpus_list[0])
    img2 = img2.cuda(gpus_list[0])
    GT_s = GT_s.cuda(gpus_list[0])

    # CCN_v1
    with torch.no_grad():
        img1 = 2.0 * (img1 - 0.5)
        img2 = 2.0 * (img2 - 0.5)
        GT_s = 2.0 * (GT_s - 0.5)

        code = code.view(img1.shape[0], 12).cuda()

        output = model(img1, img2, code)
        # calculate LPIPS
        lpips_dis = loss_fn_alex(GT_s, output)

        output = output * 0.5 + 0.5

    output = output.data[0].cpu().permute(1, 2, 0)

    output = output * 255
    output = output.clamp(0, 255)

    out_name = save_path + '/' + str(x) + '_' + str(y) + '_' + str(angle_target) + '.png'
    Image.fromarray(np.uint8(output)).save(out_name)


    GT = np.array(GT).astype(np.float32)
    GT_Y = rgb2ycbcr(GT)
    output = np.array(output).astype(np.float32)
    output_Y = rgb2ycbcr(output)
    psnr_predicted = PSNR(output_Y, GT_Y, shave_border=8)
    ssim_predicted = SSIM(output_Y, GT_Y, shave_border=8)
    lpips_dis = lpips_dis.cpu().item()
    print(psnr_predicted)
    return psnr_predicted, ssim_predicted, lpips_dis


##########
## Done ##
##########

##Eval Start!!!!
len_file = len(test_param)
final_psnr = 0
final_ssim = 0
final_lpips = 0
for t in range(len_file):
    x = test_param[t]['x']
    y = test_param[t]['y']

    avg_psnr_predicted = 0
    avg_ssim_predicted = 0
    avg_lpips_predicted = 0
    t0 = time.time()
    count = 0

    for r in range(0, 360, 180):
        angle_left = r
        angle_right = r + 180
        for i in range(1, 12, 1):
            count += 1
            angle_target = angle_left + i * 15
            psnr_predicted, ssim_predicted, lpips_predicted = eval(angle_left, angle_right, angle_target, x, y)
            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted
            avg_lpips_predicted += lpips_predicted

    t1 = time.time()

    avg_psnr_predicted = avg_psnr_predicted / count
    avg_ssim_predicted = avg_ssim_predicted / count
    avg_lpips_predicted = avg_lpips_predicted / count

    print("file = {} || PSNR_predicted = {:.4f} || "
          "SSIM_predicted = {:.4f} || "
          "LPIPS_predicted = {:.4f} || "
          "time = {:.4f}".format(t,
                            avg_psnr_predicted,
                            avg_ssim_predicted,
                            avg_lpips_predicted,
                            t1-t0))

    final_psnr = final_psnr + avg_psnr_predicted
    final_ssim = final_ssim + avg_ssim_predicted
    final_lpips = final_lpips + avg_lpips_predicted

final_psnr = final_psnr / len_file
final_ssim = final_ssim / len_file
final_lpips = final_lpips / len_file

print("final PSNR = {:.4f} || final SSIM = {:.4f} || final LPIPS = {:.4f}".format(final_psnr, final_ssim, final_lpips))




