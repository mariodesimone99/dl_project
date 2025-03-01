import torch
from torch.utils.data.dataset import Dataset
import numpy as np

import os
import glob

class CityscapesDataset(Dataset):
    def __init__(self, root="../dataset/cityscapes_preprocessed", split="train", labels=7):
        self.root = root
        self.split = split
        self.images = glob.glob(os.path.join(root, split, "image", "*.npy"))
        self.labels = glob.glob(os.path.join(root, split, f"label_{labels}", "*.npy"))
        self.depth = glob.glob(os.path.join(root, split, "depth", "*.npy"))
        self.images.sort()
        self.labels.sort()
        self.depth.sort()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(self.images[idx])).permute(2, 0, 1)
        label = torch.from_numpy(np.load(self.labels[idx]))
        depth = torch.from_numpy(np.load(self.depth[idx])).squeeze(2)
        out_dict = {'segmentation': label, 'depth': depth}
        return image, out_dict
        # return image, label, depth