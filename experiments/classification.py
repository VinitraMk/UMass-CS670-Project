from datautils.datareader import read_data
from torch.utils.data import DataLoader
from datautils.dataset import COD10KDataset
from models.ots_models import get_model

def __prepare_data(args, type = 'Train'):
    data_paths = read_data('Train', True)
    dataset = COD10KDataset(data_paths)
    return dataset

def binary_classification(args):
    train_dataset = __prepare_data(args)
    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False)
    model, _ = get_model("vit")
    for i_batch, batch in enumerate(dataloader):
        op = model(batch)
        print(op.size())
        break
    
    
    