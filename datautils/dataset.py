from torch.utils.data import Dataset
from skimage import io

class COD10KDataset(Dataset):
    
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.length = len(data_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img_tensor = io.imread(self.data_paths[idx]['image_path'])
        mask_tensor = io.imread(self.data_paths[idx]['mask_path'])
        label = self.data_paths[idx]['camouflaged']
        
        sample = {
            'img': img_tensor,
            'mask': mask_tensor,
            'label': label
        }
        
        return sample
        