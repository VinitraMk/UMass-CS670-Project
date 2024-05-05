from torch.utils.data import Dataset
from skimage import io
import numpy as np
from torchvision.io import read_image
import torch
from torchvision.transforms import transforms
from PIL import Image

class COD10KDataset(Dataset):
    
    def __init__(self, data_paths, task_type = 'semantic_segmentation'):
        self.data_paths = data_paths
        self.length = len(data_paths)
        self.task_type = task_type
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img_tensor = read_image(self.data_paths[idx]['image_path']).type(torch.FloatTensor)
        mask_tensor = read_image(self.data_paths[idx]['mask_path']).type(torch.FloatTensor)
        label = self.data_paths[idx]['camouflaged']
        img_name = self.data_paths[idx]['image_name']

        if self.task_type == 'binary_classification':
            img = np.array(Image.open(self.data_paths[idx]['image_path']))
            return img, int(label), img_name
        
        sample = {
            'img': img_tensor,
            'mask': mask_tensor,
            'label': label,
            'img_name': img_name
        }
        
        return sample
        