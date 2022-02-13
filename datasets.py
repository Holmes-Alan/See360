import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps, ImageEnhance
import random
import re
import json
import math
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, i):
    if i != 0:
        img = Image.open(filepath).convert('RGB')
    elif i == 0:
        img = Image.open(filepath).convert('YCbCr')
        img = img.getchannel(0)
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    (w, h) = img_in.size
    new_size_in = tuple([int(scale*w), int(scale*h)])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def modcrop(im):
    (w, h) = im.size
    # new_h = h//modulo*modulo
    # new_w = w//modulo*modulo
    # ih = h - new_h
    # iw = w - new_w
    if w >= h:
        dt = (h - w//2)//2
        db = h - w - dt
        ims = im.crop((0, dt, w, h - db))
    else:
        dl = (w - h//2)//2
        dr = w - h - dl
        ims = im.crop((dl, 0, w - dr, h))
        ims = ims.rotate(90)

    new_size_in = tuple([512, 256])
    ims = ims.resize(new_size_in, resample=Image.BICUBIC)

    return ims

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_left, img_right, img_tar, img_mask, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if torch.rand(1).item() < 0.5 and flip_h:
        img_left = ImageOps.flip(img_left)
        img_right = ImageOps.flip(img_right)
        img_tar = ImageOps.flip(img_tar)
        img_mask = ImageOps.flip(img_mask)
        info_aug['flip_h'] = True

    # if rot:
    #     if torch.rand(1).item() < 0.5:
    #         img_left = ImageOps.mirror(img_left)
    #         img_right = ImageOps.mirror(img_right)
    #         img_tar = ImageOps.mirror(img_tar)
    #         info_aug['flip_v'] = True
    #     if torch.rand(1).item() < 0.5:
    #         a = random.randint(1, 3) * 90
    #         img_left = img_left.rotate(a)
    #         img_right = img_right.rotate(a)
    #         img_tar = img_tar.rotate(a)
    #         info_aug['trans'] = True

    return img_left, img_right, img_tar, img_mask, info_aug



class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()

        self.transform = transform
        self.data_augmentation = data_augmentation
        self.data_dir = data_dir
        f = open(self.data_dir + '/train_record.json')
        self.render_params = json.load(f)

    def __getitem__(self, index):
        x = self.render_params[index]['x']
        y = self.render_params[index]['y']
        rotate_left = random.randint(0, 20) * 15 # max=300

        rotate_right = rotate_left + 60

        ratio = 12

        if rotate_right == 360:
            rotate_right = 0


        rotate_target = rotate_left + 5 * random.randint(1, 11)

        left_name = self.data_dir + '/train_data/' + str(x) + '_' + str(y) + '_' + str(rotate_left) + '.png'
        right_name = self.data_dir + '/train_data/' + str(x) + '_' + str(y) + '_' + str(rotate_right) + '.png'
        target_name = self.data_dir + '/train_data/' + str(x) + '_' + str(y) + '_' + str(rotate_target) + '.png'
        mask_name = self.data_dir + '/train_mask/' + str(x) + '_' + str(y) + '_' + str(rotate_target) + '.png'
        left_img = load_img(left_name, 1)
        left_img = rescale_img(left_img, 0.5) # 320*240
        right_img = load_img(right_name, 1)
        right_img = rescale_img(right_img, 0.5) # 320*240
        target_img = load_img(target_name, 1)
        target_img = rescale_img(target_img, 0.5)
        mask_img = load_img(mask_name, 1)
        mask_img = rescale_img(mask_img, 0.5)


        theta = int(0.2 * (rotate_target - rotate_left))
        code = F.one_hot(torch.tensor(theta), num_classes=12)
        code = np.float32(code.numpy())

        x = (np.float32(x) + 6700) / 17700
        y = (np.float32(y) + 1900) / 3000
        # theta = np.float32(theta)
        if self.data_augmentation:
            left_img, right_img, target_img, mask_img, _ = augment(left_img, right_img, target_img, mask_img)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            target_img = self.transform(target_img)
            mask_img = self.transform(mask_img)


        return left_img, right_img, target_img, mask_img, x, y, code

    def __len__(self):
        return len(self.render_params)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            #input = self.transform(input)
            bicubic = self.transform(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)



def _to_radians(deg):
    return deg * (np.pi / 180)


def rot_matrix_x(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = 1.
    mat[1, 1] = np.cos(theta)
    mat[1, 2] = -np.sin(theta)
    mat[2, 1] = np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat

def rot_matrix_y(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 2] = np.sin(theta)
    mat[1, 1] = 1.
    mat[2, 0] = -np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat

def rot_matrix_z(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 1] = -np.sin(theta)
    mat[1, 0] = np.sin(theta)
    mat[1, 1] = np.cos(theta)
    mat[2, 2] = 1.
    return mat

def pad_rotmat(theta):
    """theta = (3x3) rotation matrix"""
    return np.hstack((theta, np.zeros((3,1))))


def get_theta(angles):
    '''Construct a rotation matrix from angles.
    This uses the Euler angle representation. But
    it should also work if you use an axis-angle
    representation.
    '''
    # bs = angles.shape[0]
    # theta = np.zeros((3, 4))
    angles = _to_radians(angles)
    theta = pad_rotmat(np.dot(np.dot(rot_matrix_z(0), rot_matrix_y(angles)), rot_matrix_x(0)))

    return theta