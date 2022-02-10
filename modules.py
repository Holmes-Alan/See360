import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision
from torch import mm
from torchvision import models
import numpy as np
import math
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, norm=None):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding=0, bias=bias)
        reflection_padding = padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.act = torch.nn.PReLU()
        # self.act = torch.nn.ReLU()
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

    def forward(self, x):
        x = self.reflection_pad(x)
        out = self.conv(x)

        if self.norm is not None:
            out = self.bn(out)

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, norm=None):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()
        # self.act = torch.nn.ReLU()
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

    def forward(self, x):
        out = self.deconv(x)

        if self.norm is not None:
            out = self.bn(out)

        return self.act(out)



class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.ReLU(inplace=True)
        self.act2 = torch.nn.ReLU(inplace=True)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)


    def forward(self, x):

        out = self.conv1(x)
        if self.norm is not None:
            out = self.bn(out)
        out = self.act1(out)

        out = self.conv2(out)

        if self.norm is not None:
            out = self.bn(out)

        out = self.act2(out + x)

        return out




class generator_final(nn.Module):
    def __init__(self, input_dim, dim):
        super(generator_final, self).__init__()

        self.feat_1 = ConvBlock(input_dim, dim, 3, 1, 1, norm='instance') # 240*320
        self.feat_2 = ConvBlock(dim, 2 * dim, 4, 2, 1, norm='instance') # 120*160
        self.feat_3 = ConvBlock(2 * dim, 4 * dim, 4, 2, 1, norm='instance') # 60*80
        self.feat_4 = ConvBlock(4 * dim, 8 * dim, 4, 2, 1, norm='instance') # 30*40

        self.compress = nn.Sequential(
            nn.Conv2d(8 * dim, dim, 1, 1, 0), # 512-->1
            nn.ReLU()
        )
        self.attn = AttnBlock(kernel_size=11)
        self.act = nn.Tanh()
        self.code_mask = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
        )
        self.homography = nn.Sequential(
            nn.Linear(600, 48),
            nn.ReLU(),
        )
        self.perspective_left = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 24)
        )
        self.perspective_right = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 24)
        )

        self.postproc_4 = nn.Sequential(
            ConvBlock(16 * dim, 8 * dim, 1, 1, 0),
            ResnetBlock(8 * dim, 3, 1, 1, norm='instance'),
        )
        self.postproc_3 = nn.Sequential(
            ConvBlock(8 * dim, 4 * dim, 1, 1, 0),
            ResnetBlock(4 * dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(4 * dim, 8 * dim, 1, 1, 0),
        )
        self.postproc_2 = nn.Sequential(
            ConvBlock(4 * dim, 2 * dim, 1, 1, 0),
            ResnetBlock(2 * dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(2 * dim, 4 * dim, 1, 1, 0),
        )
        self.postproc_1 = nn.Sequential(
            ConvBlock(2 * dim, dim, 1, 1, 0),
            ResnetBlock(dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(dim, 2 * dim, 1, 1, 0),
        )

        # self.up1 = DeconvBlock(16 * dim, 8 * dim, 4, 2, 1, norm='instance')  # 60*80
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(8 * dim, 4 * dim, 3, 1, 1, norm='instance')
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(8 * dim, 2 * dim, 3, 1, 1, norm='instance')
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(4 * dim, 1 * dim, 3, 1, 1, norm='instance')
        )

        self.recon = nn.Sequential(
            ResnetBlock(2 * dim, 3, 1, 1, norm='instance'),
            nn.Conv2d(2 * dim, input_dim, 3, 1, 1),
            nn.Tanh()
        )


    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        #theta = theta.view(-1, 2, 3)
        grid = Func.affine_grid(theta, x.size())
        out = Func.grid_sample(x, grid, padding_mode='zeros')
        return out

    def forward(self, left_img, right_img, code):
        # code mask
        mean, std = self.code_mask(code).chunk(2, dim=1)
        # initial
        left_feat_1 = self.feat_1(left_img)
        right_feat_1 = self.feat_1(right_img)
        left_feat_2 = self.feat_2(left_feat_1)
        right_feat_2 = self.feat_2(right_feat_1)
        left_feat_3 = self.feat_3(left_feat_2)
        right_feat_3 = self.feat_3(right_feat_2)
        left_feat_4 = self.feat_4(left_feat_3)
        right_feat_4 = self.feat_4(right_feat_3)

        # 2d transform
        left_compress = self.compress(left_feat_4)
        right_compress = self.compress(right_feat_4)
        fuse_left = self.attn(left_compress, right_compress)
        fuse_right = self.attn(right_compress, left_compress)
        fuse_left = fuse_left.view(fuse_left.shape[0], -1)
        fuse_right = fuse_right.view(fuse_right.shape[0], -1)
        fuse_left = (fuse_left - torch.min(fuse_left)) / (torch.max(fuse_left) - torch.min(fuse_left))
        fuse_right = (fuse_right - torch.min(fuse_right)) / (torch.max(fuse_right) - torch.min(fuse_right))

        theta_left = self.homography(fuse_left) * (1 + std) + mean
        theta_left = self.perspective_left(theta_left)
        theta_right = self.homography(fuse_right) * (1 + std) + mean
        theta_right = self.perspective_right(theta_right)

        theta_left = theta_left.view(theta_left.shape[0], 8, 3)
        theta_right = theta_right.view(theta_right.shape[0], 8, 3)
        a, b = torch.split(theta_left, split_size_or_sections=[2,1], dim=2)
        a = self.act(a)
        theta_left = torch.cat((a, b), dim=2)
        theta_left_1, theta_left_2, theta_left_3, theta_left_4 = theta_left.chunk(4, dim=1)
        a, b = torch.split(theta_right, split_size_or_sections=[2,1], dim=2)
        a = self.act(a)
        theta_right = torch.cat((a, b), dim=2)
        theta_right_1, theta_right_2, theta_right_3, theta_right_4 = theta_right.chunk(4, dim=1)
        # perspective transformation
        left_feat_1_rotated = self.stn(left_feat_1, theta_left_1)
        right_feat_1_rotated = self.stn(right_feat_1, theta_right_1)
        left_feat_2_rotated = self.stn(left_feat_2, theta_left_2)
        right_feat_2_rotated = self.stn(right_feat_2, theta_right_2)
        left_feat_3_rotated = self.stn(left_feat_3, theta_left_3)
        right_feat_3_rotated = self.stn(right_feat_3, theta_right_3)
        left_feat_4_rotated = self.stn(left_feat_4, theta_left_4)
        right_feat_4_rotated = self.stn(right_feat_4, theta_right_4)
        # 2d postprocess
        feat_rotated = torch.cat((left_feat_4_rotated, right_feat_4_rotated), dim=1)
        feat_4 = self.postproc_4(feat_rotated)
        feat_rotated = torch.cat((left_feat_3_rotated, right_feat_3_rotated), dim=1)
        feat_3 = self.postproc_3(feat_rotated)
        feat_rotated = torch.cat((left_feat_2_rotated, right_feat_2_rotated), dim=1)
        feat_2 = self.postproc_2(feat_rotated)
        feat_rotated = torch.cat((left_feat_1_rotated, right_feat_1_rotated), dim=1)
        feat_1 = self.postproc_1(feat_rotated)

        # reconstruction
        dec_feat = self.up1(feat_4)
        dec_feat = torch.cat((dec_feat, feat_3), dim=1)
        dec_feat = self.up2(dec_feat)
        dec_feat = torch.cat((dec_feat, feat_2), dim=1)
        dec_feat = self.up3(dec_feat)
        dec_feat = torch.cat((dec_feat, feat_1), dim=1)

        out = self.recon(dec_feat)

        return out



    def __init__(self, input_dim, dim):
        super(generator_final_v2, self).__init__()

        self.feat_1 = ConvBlock(input_dim, dim, 3, 1, 1, norm='instance') # 240*320
        self.feat_2 = ConvBlock(dim, 2 * dim, 4, 2, 1, norm='instance') # 120*160
        self.feat_3 = ConvBlock(2 * dim, 4 * dim, 4, 2, 1, norm='instance') # 60*80
        self.feat_4 = ConvBlock(4 * dim, 8 * dim, 4, 2, 1, norm='instance') # 30*40

        self.compress = nn.Sequential(
            nn.Conv2d(8 * dim, dim, 1, 1, 0), # 512-->1
            nn.ReLU()
        )
        self.attn = AttnBlock(kernel_size=11)
        self.act = nn.Tanh()
        self.code_mask = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
        )
        self.homography_ = nn.Sequential(
            nn.Linear(700, 48),
            nn.ReLU(),
        )
        self.perspective_left = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 24)
        )
        self.perspective_right = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 24)
        )

        self.postproc_4 = nn.Sequential(
            ConvBlock(16 * dim, 8 * dim, 1, 1, 0),
            ResnetBlock(8 * dim, 3, 1, 1, norm='instance'),
        )
        self.postproc_3 = nn.Sequential(
            ConvBlock(8 * dim, 4 * dim, 1, 1, 0),
            ResnetBlock(4 * dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(4 * dim, 8 * dim, 1, 1, 0),
        )
        self.postproc_2 = nn.Sequential(
            ConvBlock(4 * dim, 2 * dim, 1, 1, 0),
            ResnetBlock(2 * dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(2 * dim, 4 * dim, 1, 1, 0),
        )
        self.postproc_1 = nn.Sequential(
            ConvBlock(2 * dim, dim, 1, 1, 0),
            ResnetBlock(dim, 3, 1, 1, norm='instance'),
            # nn.Conv2d(dim, 2 * dim, 1, 1, 0),
        )

        # self.up1 = DeconvBlock(16 * dim, 8 * dim, 4, 2, 1, norm='instance')  # 60*80
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(8 * dim, 4 * dim, 3, 1, 1, norm='instance')
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(8 * dim, 2 * dim, 3, 1, 1, norm='instance')
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(4 * dim, 1 * dim, 3, 1, 1, norm='instance')
        )

        self.recon = nn.Sequential(
            ResnetBlock(2 * dim, 3, 1, 1, norm='instance'),
            nn.Conv2d(2 * dim, input_dim, 3, 1, 1),
            nn.Tanh()
        )


    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        #theta = theta.view(-1, 2, 3)
        grid = Func.affine_grid(theta, x.size())
        out = Func.grid_sample(x, grid, padding_mode='zeros')
        return out

    def forward(self, left_img, right_img, code):
        # code mask
        mean, std = self.code_mask(code).chunk(2, dim=1)
        # initial
        left_feat_1 = self.feat_1(left_img)
        right_feat_1 = self.feat_1(right_img)
        left_feat_2 = self.feat_2(left_feat_1)
        right_feat_2 = self.feat_2(right_feat_1)
        left_feat_3 = self.feat_3(left_feat_2)
        right_feat_3 = self.feat_3(right_feat_2)
        left_feat_4 = self.feat_4(left_feat_3)
        right_feat_4 = self.feat_4(right_feat_3)

        # 2d transform
        left_compress = self.compress(left_feat_4)
        right_compress = self.compress(right_feat_4)
        fuse_left = self.attn(left_compress, right_compress)
        fuse_right = self.attn(right_compress, left_compress)
        fuse_left = fuse_left.view(fuse_left.shape[0], -1)
        fuse_right = fuse_right.view(fuse_right.shape[0], -1)
        fuse_left = (fuse_left - torch.min(fuse_left)) / (torch.max(fuse_left) - torch.min(fuse_left))
        fuse_right = (fuse_right - torch.min(fuse_right)) / (torch.max(fuse_right) - torch.min(fuse_right))

        theta_left = self.homography_(fuse_left) * (1 + std) + mean
        theta_left = self.perspective_left(theta_left)
        theta_right = self.homography_(fuse_right) * (1 + std) + mean
        theta_right = self.perspective_right(theta_right)

        theta_left = theta_left.view(theta_left.shape[0], 8, 3)
        theta_right = theta_right.view(theta_right.shape[0], 8, 3)
        a, b = torch.split(theta_left, split_size_or_sections=[2,1], dim=2)
        a = self.act(a)
        theta_left = torch.cat((a, b), dim=2)
        theta_left_1, theta_left_2, theta_left_3, theta_left_4 = theta_left.chunk(4, dim=1)
        a, b = torch.split(theta_right, split_size_or_sections=[2,1], dim=2)
        a = self.act(a)
        theta_right = torch.cat((a, b), dim=2)
        theta_right_1, theta_right_2, theta_right_3, theta_right_4 = theta_right.chunk(4, dim=1)
        # perspective transformation
        left_feat_1_rotated = self.stn(left_feat_1, theta_left_1)
        right_feat_1_rotated = self.stn(right_feat_1, theta_right_1)
        left_feat_2_rotated = self.stn(left_feat_2, theta_left_2)
        right_feat_2_rotated = self.stn(right_feat_2, theta_right_2)
        left_feat_3_rotated = self.stn(left_feat_3, theta_left_3)
        right_feat_3_rotated = self.stn(right_feat_3, theta_right_3)
        left_feat_4_rotated = self.stn(left_feat_4, theta_left_4)
        right_feat_4_rotated = self.stn(right_feat_4, theta_right_4)
        # 2d postprocess
        feat_rotated = torch.cat((left_feat_4_rotated, right_feat_4_rotated), dim=1)
        feat_4 = self.postproc_4(feat_rotated)
        feat_rotated = torch.cat((left_feat_3_rotated, right_feat_3_rotated), dim=1)
        feat_3 = self.postproc_3(feat_rotated)
        feat_rotated = torch.cat((left_feat_2_rotated, right_feat_2_rotated), dim=1)
        feat_2 = self.postproc_2(feat_rotated)
        feat_rotated = torch.cat((left_feat_1_rotated, right_feat_1_rotated), dim=1)
        feat_1 = self.postproc_1(feat_rotated)

        # reconstruction
        dec_feat = self.up1(feat_4)
        dec_feat = torch.cat((dec_feat, feat_3), dim=1)
        dec_feat = self.up2(dec_feat)
        dec_feat = torch.cat((dec_feat, feat_2), dim=1)
        dec_feat = self.up3(dec_feat)
        dec_feat = torch.cat((dec_feat, feat_1), dim=1)

        out = self.recon(dec_feat)

        return out



class AttnBlock(nn.Module):
    def __init__(self, kernel_size=11):
        super(AttnBlock, self).__init__()

        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=1)
        self.kernel_size = kernel_size

    def forward(self, im1, im2):
        sample_flrs = self.unfold(im1)
        sample_flrs = sample_flrs.permute(0, 2, 1).contiguous()
        sample_flrs = sample_flrs.view(im1.size(0), -1, im1.size(1), self.kernel_size, self.kernel_size)

        imgs_gp = torch.split(im2, 1, dim=0)
        flrs_gp = torch.split(sample_flrs, 1, dim=0)

        y = []
        for im, flr in zip(imgs_gp, flrs_gp):
            flr = flr[0]

            eps = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                eps = eps.cuda()

            normalized_flr = flr / torch.max(torch.sqrt((flr * flr).sum([1, 2, 3], keepdim=True)), eps)
            tmp = Func.conv2d(im, normalized_flr, stride=1, padding=0)
            tmp = tmp.contiguous()
            out = torch.max(tmp, dim=1, keepdim=True).values

            y.append(out)
        y = torch.cat(y, dim=0)
        y = y.contiguous()

        return y




class discriminator_v2(nn.Module):
    def __init__(self, num_channels, base_filter):
        super(discriminator_v2, self).__init__()

        self.input_conv = nn.Conv2d(4 * num_channels, base_filter, 3, 1, 1)#512*256
        self.conv1 = nn.Conv2d(base_filter, base_filter * 2, 4, 2, 1)
        self.norm1 = nn.InstanceNorm2d(base_filter * 2)
        self.conv2 = nn.Conv2d(base_filter * 2, base_filter * 4, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(base_filter * 4)
        self.conv3 = nn.Conv2d(base_filter * 4, base_filter * 8, 4, 2, 1)
        self.norm3 = nn.InstanceNorm2d(base_filter * 8)
        self.act = nn.LeakyReLU(0.2, False)

        self.weight = nn.Conv2d(base_filter * 8, 1, 3, 1, 1)

        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x, mask, left, right):
        x = torch.cat((x, mask, left, right), 1)
        feat = self.act(self.input_conv(x))
        out1 = self.act(self.norm1(self.conv1(feat)))
        out2 = self.act(self.norm2(self.conv2(out1)))
        out3 = self.act(self.norm3(self.conv3(out2)))

        prob = self.weight(out3)

        b = feat.shape[0]

        prob = prob.view(b, -1)

        feat = feat.view(b, -1)
        out1 = out1.view(b, -1)
        out2 = out2.view(b, -1)
        out3 = out3.view(b, -1)

        out = torch.cat((feat, out1, out2, out3), 1)

        return out, prob

    def forward(self, x, mask, left, right):
        # x = torch.cat((x, y), 1)
        feat1, prob1 = self.encode(x, mask, left, right)
        x = self.down(x)
        mask = self.down(mask)
        left = self.down(left)
        right = self.down(right)
        feat2, prob2 = self.encode(x, mask, left, right)
        # x = self.down(x)
        # feat3, prob3 = self.encode(x)

        feat_out = torch.cat((feat1, feat2), 1)
        prob_out = torch.cat((prob1, prob2), 1)

        return feat_out, prob_out




class PDLoss(nn.Module):
    def __init__(self, l1_lambda=1.5, w_lambda=0.01):
        super(PDLoss, self).__init__()
        self.vgg = Vgg19Conv4().cuda()
        self.criterionL1 = nn.L1Loss()
        self.w_lambda = w_lambda
        self.l1_lambda = l1_lambda

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def w_distance(self, xvgg, yvgg):
        xvgg = xvgg / (torch.sum(xvgg, dim=(2, 3), keepdim=True) + 1e-14)
        yvgg = yvgg / (torch.sum(yvgg, dim=(2, 3), keepdim=True) + 1e-14)

        xvgg = xvgg.view(xvgg.size()[0], xvgg.size()[1], -1).contiguous()
        yvgg = yvgg.view(yvgg.size()[0], yvgg.size()[1], -1).contiguous()

        cdf_xvgg = torch.cumsum(xvgg, dim=-1)
        cdf_yvgg = torch.cumsum(yvgg, dim=-1)

        cdf_distance = torch.sum(torch.abs(cdf_xvgg - cdf_yvgg), dim=-1)
        cdf_loss = cdf_distance.mean()

        return cdf_loss

    def forward(self, x, y):
        # L1loss = self.criterionL1(x, y) * self.l1_lambda
        # L1loss = 0
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        WdLoss = self.w_distance(x_vgg, y_vgg) * self.w_lambda

        return WdLoss


### Define Vgg19 for projected distribution loss
class Vgg19Conv4(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Conv4, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()

        for x in range(23):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.slice1(x)
        return out