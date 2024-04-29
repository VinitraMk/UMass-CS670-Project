from datautils.datareader import read_data
from datautils.dataset import COD10KDataset
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
import warnings

from experiments.style_transfer import style_transfer
try:
    from experiments.synthetic import run_synthetic_pipeline
except ModuleNotFoundError:
    warnings.warn("Unable to import synthetic data generation! Check package diffusers if using this")
try:
    from experiments.sam_baseline import run_sam_pipeline
except ModuleNotFoundError:
    warnings.warn("Unable to import sam baseline! Check package groundingdino and segment_anything if using this")

def run_style_transfer_pipeline(args):
    pos_data_paths = read_data('Train')

    dataset = COD10KDataset(pos_data_paths)
    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    
    for i_batch, batch in enumerate(dataloader):
        style_transfer(batch['img'],
            (1, 4, 6, 7),
            3,
            6e-2,
            (2000, 512, 12, 1),
            6e-2,
            args)
        if i_batch == 0:
            break
       
    
if __name__ == "__main__":
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="style_transfer")
    parser.add_argument('--device', type=str, default="available")
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--model_name', type=str, default='squeezenet')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--synthetic_path', type=str, default="generated")
    args = parser.parse_args()
    
    if args.mode == "style_transfer":
        run_style_transfer_pipeline(args)
    elif args.mode == "synthetic":
        run_synthetic_pipeline(args)
    elif args.mode == "sam":
        run_sam_pipeline(args)
    
