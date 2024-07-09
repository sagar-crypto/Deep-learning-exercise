from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_dir = self.data.iloc[index, 0]
        image = imread("./" + image_dir)
        transformed_image = gray2rgb(image)
        normalized_image = self._transform(transformed_image)
        normalized_tensor = torch.Tensor(normalized_image)
        label = torch.Tensor([self.data.iloc[index, 1], self.data.iloc[index, 2]])
        return normalized_tensor, label
