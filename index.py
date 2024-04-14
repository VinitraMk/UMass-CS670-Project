from datautils.datareader import read_data
from datautils.dataset import COD10KDataset
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm

from experiments.style_transfer import style_transfer

def run_style_transfer_pipeline(args):
    pos_data_paths = read_data('Train')

    dataset = COD10KDataset(pos_data_paths)
    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    
    for i_batch, batch in enumerate(dataloader):
        style_transfer(batch['img'],
            (1, 2, 3, 4),
            3,
            0.01,
            (500, 200, 10, 1),
            0.01,
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
    
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--model_name', type=str, default='squeezenet')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--resize_size', type=int, default=256)
    args = parser.parse_args()
    
    run_style_transfer_pipeline(args)
    
