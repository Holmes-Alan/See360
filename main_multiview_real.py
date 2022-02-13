from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from laploss import LapLoss
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from modules import generator_final_v2, discriminator_v2, PDLoss
from transformer_net import VGG19
import numpy as np
from data import get_training_set
import torch.nn.functional as F
import socket
from pytorch_ssim import SSIM as pytorch_ssim


# Training settings
parser = argparse.ArgumentParser(description='PyTorch See360 real scene training code')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--pretrained_iter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./data/360HungHom')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped LR image')
parser.add_argument('--pretrained_G', default='3D/GAN_generator_1345.pth', help='sr pretrained base model')
parser.add_argument('--pretrained_D', default='GAN_discriminator_0.pth', help='sr pretrained base model')
parser.add_argument('--model_type', default='GAN', help='model name')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='logs/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]



def train(epoch):
    G_epoch_loss = 0
    D_epoch_loss = 0
    adv_epoch_loss = 0
    recon_epoch_loss = 0
    G.train()
    D.train()

    vgg_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    for iteration, batch in enumerate(training_data_loader, 1):
        left_img, right_img, target_img, mask_img = batch[0], batch[1], batch[2], batch[3]
        x, y, code = batch[4], batch[5], batch[6]
        minibatch = left_img.size()[0]
        real_label = torch.ones((minibatch, 1680))
        fake_label = torch.zeros((minibatch, 1680))
        # code = F.one_hot(theta, num_classes=12)
        if cuda:
            left_img = left_img.cuda(gpus_list[0])
            right_img = right_img.cuda(gpus_list[0])
            target_img = target_img.cuda(gpus_list[0])
            mask_img = mask_img.cuda(gpus_list[0])
            code = code.cuda(gpus_list[0])
            real_label = real_label.cuda(gpus_list[0])
            fake_label = fake_label.cuda(gpus_list[0])

        target_img = 2.0 * (target_img - 0.5)
        left_img = 2.0 * (left_img - 0.5)
        right_img = 2.0 * (right_img - 0.5)
        mask_img = 2.0 * (mask_img - 0.5)

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = False

        G_optimizer.zero_grad()

        predict = G(left_img, right_img, code)

        PD = torch.mean(PD_loss(predict * 0.5 + 0.5, target_img * 0.5 + 0.5))

        D_fake_feat, D_fake_decision = D(predict, mask_img, left_img, right_img)
        D_real_feat, D_real_decision = D(target_img, mask_img, left_img, right_img)

        GAN_loss = L1_loss(D_fake_decision, real_label)

        recon_loss = lap_loss(predict, target_img) 

        ssim_loss = 1 - ssim(predict, target_img) 

        GAN_feat_loss = L1_loss(D_real_feat.detach(), D_fake_feat)


        G_loss = 1*recon_loss + 1*ssim_loss + 1*PD + 1*GAN_loss + 1*GAN_feat_loss

        G_loss.backward()
        G_optimizer.step()

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = True

        D_optimizer.zero_grad()

        _, D_fake_decision = D(predict.detach(), mask_img.detach(), left_img.detach(), right_img.detach())
        _, D_real_decision = D(target_img, mask_img, left_img, right_img)

        real = real_label * np.random.uniform(0.7, 1.2)
        fake = fake_label + np.random.uniform(0.0, 0.3)


        Dis_loss = (L1_loss(D_real_decision, real)
                    + L1_loss(D_fake_decision, fake)) / 2.0

        # Back propagation
        D_loss = Dis_loss
        D_loss.backward()
        D_optimizer.step()

        # log
        G_epoch_loss += G_loss.data
        D_epoch_loss += D_loss.data
        adv_epoch_loss += (GAN_loss.data)
        recon_epoch_loss += (recon_loss.data)


        print(
            "===> Epoch[{}]({}/{}): G_loss: {:.4f} || D_loss: {:.4f} "
            "|| Adv: {:.4f} || Recon_Loss: {:.4f} || ssim_loss: {:.4f}"
            "|| GAN_feat_loss: {:.4f} || PD_loss: {:.4f}".format(
                epoch, iteration,
                len(training_data_loader), G_loss.data, D_loss.data,
                GAN_loss.data, recon_loss.data, ssim_loss.data,
                GAN_feat_loss.data, PD.data))
    print(
        "===> Epoch {} Complete: Avg. G_loss: {:.4f} D_loss: {:.4f} Recon_loss: {:.4f} Adv: {:.4f}".format(
            epoch, G_epoch_loss / len(training_data_loader), D_epoch_loss / len(training_data_loader),
                   recon_epoch_loss / len(training_data_loader),
                   adv_epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_G = opt.save_folder + opt.model_type + "_generator_{}.pth".format(epoch)
    model_out_D = opt.save_folder + opt.model_type + "_discriminator_{}.pth".format(epoch)
    torch.save(G.state_dict(), model_out_G)
    torch.save(D.state_dict(), model_out_D)
    print("Checkpoint saved to {} and {}".format(model_out_G, model_out_D))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')


G = generator_final_v2(input_dim=3, dim=64)
D = discriminator_v2(num_channels=3, base_filter=64)
VGG = VGG19()
PD_loss = PDLoss(l1_lambda=1.5, w_lambda=0.01)


G = torch.nn.DataParallel(G, device_ids=gpus_list)
D = torch.nn.DataParallel(D, device_ids=gpus_list)
VGG = torch.nn.DataParallel(VGG, device_ids=gpus_list)
PD_loss = torch.nn.DataParallel(PD_loss, device_ids=gpus_list)


L1_loss = nn.L1Loss()
lap_loss = LapLoss(max_levels=5, k_size=5, sigma=2.0)
BCE_loss = nn.BCEWithLogitsLoss()
L2_loss = nn.MSELoss()
ssim = pytorch_ssim()


print('---------- Generator architecture -------------')
print_network(G)
print('---------- Discriminator architecture -------------')
print_network(D)
print('----------------------------------------------')


if opt.pretrained:
    model_G = os.path.join(opt.save_folder + opt.pretrained_G)
    model_D = os.path.join(opt.save_folder + opt.pretrained_D)
    if os.path.exists(model_G):
        # G.load_state_dict(torch.load(model_G, map_location=lambda storage, loc: storage))
        pretrained_dict = torch.load(model_G, map_location=lambda storage, loc: storage)
        model_dict = G.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        G.load_state_dict(model_dict)
        print('Pre-trained Generator model is loaded.')
    if os.path.exists(model_D):
        D.load_state_dict(torch.load(model_D, map_location=lambda storage, loc: storage))
        print('Pre-trained Discriminator model is loaded.')



if cuda:
    G = G.cuda(gpus_list[0])
    D = D.cuda(gpus_list[0])
    PD_loss = PD_loss.cuda(gpus_list[0])
    VGG = VGG.cuda(gpus_list[0])
    L1_loss = L1_loss.cuda(gpus_list[0])
    BCE_loss = BCE_loss.cuda(gpus_list[0])
    L2_loss = L2_loss.cuda(gpus_list[0])
    lap_loss = lap_loss.cuda(gpus_list[0])
    ssim = ssim.cuda(gpus_list[0])


G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999), eps=1e-8)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999), eps=1e-8)


# writer = SummaryWriter(opt.log_folder)
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)