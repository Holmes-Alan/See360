import torch
import torchvision

class generator_v5(torch.nn.Module):
    def __init__(self):
        super(generator_v5, self).__init__()
        # coarse convolution
        self.coarse_conv1 = ConvLayer(1, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.coarse_conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.coarse_conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # mask convolution
        self.mask_conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.weight1 = ConvLayer(32, 64, kernel_size=3, stride=1)
        self.mask_conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.weight2 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.mask_conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.weight3 = ConvLayer(128, 256, kernel_size=3, stride=1)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.out_act = torch.nn.Tanh()
        # Non-linearities
        self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, mask, false):
        false_feat = self.act(self.in1(self.coarse_conv1(false)))

        mask_feat = self.act(self.mask_conv1(mask))
        mean, scale = self.weight1(mask_feat).chunk(2, dim=1)
        false_feat = false_feat * (1 + scale) + mean
        false_feat = self.act(self.in2(self.coarse_conv2(false_feat)))

        mask_feat = self.act(self.mask_conv2(mask_feat))
        mean, scale = self.weight2(mask_feat).chunk(2, dim=1)
        false_feat = false_feat * (1 + scale) + mean
        false_feat = self.act(self.in3(self.coarse_conv3(false_feat)))

        mask_feat = self.act(self.mask_conv3(mask_feat))
        mean, scale = self.weight3(mask_feat).chunk(2, dim=1)
        false_feat = false_feat * (1 + scale) + mean
        false_feat = self.res1(false_feat)
        false_feat = self.res2(false_feat)
        false_feat = self.res3(false_feat)
        false_feat = self.res4(false_feat)
        false_feat = self.res5(false_feat)
        false_feat = self.act(self.in4(self.deconv1(false_feat)))
        false_feat = self.act(self.in5(self.deconv2(false_feat)))

        out = self.out_act(self.deconv3(false_feat))

        return out





class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        mean = X.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = X.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        X = (X - mean) / std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(torch.nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        # mean = x.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        # std = x.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        # x = (x - mean) / std
        # y = (y - mean) / std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
