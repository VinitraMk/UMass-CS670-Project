from models.custom_models import get_model
from common.utils import load_model, init_config
from datautils.datareader import read_data
from datautils.dataset import COD10KDataset
import random
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from common.utils import get_config
import os
from tqdm import tqdm


class SemanticSegmentation:
    
    def __init__(self, model_path, model_name, num_classes):
        init_config()
        self.model = get_model(num_classes, model_name)
        mf = load_model(model_path)
        self.model.load_state_dict(mf)
        
    def __prepare_data(self, dataset_type = 'Train'):
        data_paths = read_data(dataset_type, True)
        dataset = COD10KDataset(data_paths, 'binary_classification')
        smlen = int(0.01 * len(dataset))
        #ridxs = random.sample(range(len(dataset)), smlen)
        ridxs = list(range(smlen))
        smftr_dataset = Subset(dataset, ridxs)
        return dataset, smftr_dataset
    
    def __image_collate(self, batch):
        batchlist = list(map(list, zip(*batch)))
        return batchlist
    
    def __convert_to_grascale(self, img):
        imin, imax = img.min(), img.max()
        x = (img - imin) / (imax - imin)
        return x
    
    def run(self, args):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(self.__convert_to_grascale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        cfg = get_config()
        
        test_dataset, sm_test_dataset = self.__prepare_data('Test')
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, collate_fn = self.__image_collate)
        target_layers = [self.model.model.layer4[-1]]
        grad_cam = GradCAM(model = self.model.model, target_layers = target_layers)
        target_class = [ClassifierOutputTarget(1)]
        for i_batch, batch in enumerate(tqdm(test_loader, desc = 'Running through test set')):
            img_batch = list(map(transform, batch[0]))
            img_batch = torch.stack(img_batch, 0)
            #print('img batch', img_batch.size())
            #print(op.size())
            grayscale_cam = grad_cam(input_tensor = img_batch, targets = target_class)
            grayscale_cam = grayscale_cam[0, :, :]
            rgb_imgs = list(map(tensor_transform, batch[0]))
            rgb_imgs = np.transpose(np.stack(rgb_imgs, 0), (0, 2, 3, 1))
            rgb_imgs /= 255.0
            #print(grayscale_cam.shape, rgb_imgs.shape)
            cam_img = show_cam_on_image(rgb_imgs[0], grayscale_cam, use_rgb = True)
            cam_mask = grayscale_cam.copy()
            cam_mask[cam_mask > 0.6] = 1
            cam_mask[cam_mask <= 0.6] = 0
            cam_mask = 1 - cam_mask
            op_path = os.path.join(cfg['output_dir'], f'Test-SemSeg/{batch[2][0]}')
            op_path = op_path.replace("jpg", "png")
            #print(cfg['output_dir'], op_path)
            plt.imsave(op_path, cam_mask, cmap='binary')
            #plt.imshow(grayscale_cam)
            #plt.show()
            #plt.imshow(cam_mask)
            #plt.show()

def run_semantic_segmentation_pipeline():
    seg = SemanticSegmentation('./models/best_models/resnet-18-full.pt', 'resnet18', 2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument('--resize_size', type=int, default=336)
    args = parser.parse_args(args=[])
    seg.run(args)
    
#run_semantic_segmentation_pipeline() 