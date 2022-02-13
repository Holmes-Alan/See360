from os.path import join
from torchvision import transforms
from datasets import DatasetFromFolder as data_loader
from datasets_real import DatasetFromFolder as data_loader_real
from torch.utils.data import DataLoader


def transform():
    return transforms.Compose([
        # ColorJitter(hue=0.3, brightness=0.3, saturation=0.3),
        # RandomRotation(10, resample=PIL.Image.BILINEAR),
        # transforms.Resize((512, 256), interpolation=2),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def get_training_set(data_dir, data_augmentation):

    train_set = data_loader(data_dir, data_augmentation, transform=transform())

    return train_set


def get_training_set_real(data_dir, data_augmentation):

    train_set = data_loader_real(data_dir, data_augmentation, transform=transform())

    return train_set


