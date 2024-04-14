from torch.utils.data import Dataset
from skimage import io
import numpy as np
from torchvision.io import read_image
import torch

class COD10KDataset(Dataset):
    
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.length = len(data_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img_tensor = read_image(self.data_paths[idx]['image_path']).type(torch.FloatTensor)
        mask_tensor = read_image(self.data_paths[idx]['mask_path']).type(torch.FloatTensor)
        label = self.data_paths[idx]['camouflaged']
        
        sample = {
            'img': img_tensor,
            'mask': mask_tensor,
            'label': label
        }
        
        return sample
        