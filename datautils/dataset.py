from torch.utils.data import Dataset
from skimage import io
import numpy as np
from torchvision.io import read_image
import torch
from torchvision.transforms import transforms
from PIL import Image
from common.utils import get_config
import os

class COD10KDataset(Dataset):
    
    def __init__(self, data_paths, task_type = 'semantic_segmentation', label_arr = []):
        self.data_paths = data_paths
        self.length = len(data_paths)
        self.task_type = task_type
        self.label_arr = label_arr
        
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
        elif self.task_type == 'multiclass_classification':
            img = np.array(Image.open(self.data_paths[idx]['image_path']))
            if 'Texture' in img_name:
                class_lbl = img_name.split('-')[-4].lower()
            else:
                class_lbl = img_name.split('-')[-2].lower()
            return img, class_lbl, img_name
        
        sample = {
            'img': img_tensor,
            'mask': mask_tensor,
            'label': label,
            'img_name': img_name
        }
        
        return sample

class CamouflagedTestDataset(Dataset):

    def __init__(self, dataset_name = 'cod10k'):
        cfg = get_config()
        dataset_path = os.path.join(cfg["data_dir"], f'TestDataset/{dataset_name.upper()}/Imgs')
        self.data_paths = os.listdir(dataset_path)
        self.length = len(self.data_paths)
        self.img_par = os.path.join(cfg["data_dir"], f'TestDataset/{dataset_name.upper()}/Imgs')
        self.gt_par =  os.path.join(cfg["data_dir"], f'TestDataset/{dataset_name.upper()}/GT')
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_par, self.data_paths[idx])
        mask_path = os.path.join(self.gt_par, self.data_paths[idx]).replace("jpg", "png")
        img = np.array(Image.open(img_path))
        mask_tensor = read_image(mask_path).type(torch.FloatTensor)
        return img, mask_tensor, self.data_paths[idx]


