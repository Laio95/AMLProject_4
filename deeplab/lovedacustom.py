import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import OrderedDict
import numpy as np

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

class loveDAcustom(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])  # stesso nome

        image = Image.open(image_path)
        mask = Image.open(mask_path)  # valori da 0 a 6

        if self.transform:
            image = self.transform(image)
            if isinstance(image, torch.Tensor):
                image_size =( image.shape[1], image.shape[2] ) #H and W  
            else:
                image_size = image.size
            mask = TF.resize(mask, image_size, interpolation=Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask