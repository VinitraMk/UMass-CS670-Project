from datautils.datareader import read_data
from torch.utils.data import DataLoader
from datautils.dataset import COD10KDataset
from models.custom_models import get_model
from torchvision.transforms import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Subset
from random import shuffle
from tqdm import tqdm
import os
import random

from common.utils import get_config, save_experiment_output, save_model_helpers, load_model, get_modelinfo

class Classification:

    def __init__(self):
        pass

    def __image_collate(self, batch):
        batchlist = list(map(list, zip(*batch)))
        return batchlist
    
    def __convert_to_grascale(self, img):
        imin, imax = img.min(), img.max()
        x = (img - imin) / (imax - imin)
        return x
    
    def __prepare_data(self, dstype = 'Train', task_type = 'binary_classification'):
        data_paths = read_data(dstype, True)
        dataset = COD10KDataset(data_paths, task_type)
        smlen = int(0.1 * len(dataset))
        ridxs = random.sample(range(len(dataset)), smlen)
        smftr_dataset = Subset(dataset, ridxs)
        return dataset, smftr_dataset
    
    def __get_experiment_chkpt(self, model, optimizer):
    
        cfg = get_config()
        root_dir = cfg["root_dir"]
    
        mpath = os.path.join(root_dir, "models/checkpoints/curr_model.pt")
        opath = os.path.join(root_dir, "models/checkpoints/curr_model_optimizer.pt")
        if os.path.exists(mpath):
            print("Loading saved model")
            saved_model = load_model(mpath)
            saved_optim = load_model(opath)
            #model_dict = saved_model["model_state"]
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optim)
            model_info = get_modelinfo(True)
            return model, optimizer, model_info
        else:
            return model, optimizer, None
    
    def __run_train_loop(self, args, model, optimizer, model_info, train_dataloader, val_dataloader, val_len):
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.resize_size, args.resize_size)),
            transforms.Lambda(self.__convert_to_grascale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
    
        loss_fn = torch.nn.BCEWithLogitsLoss()
        last_epoch = -1
        train_losses = []
        val_losses = []
        val_accs = []
        best_val_loss = 9999
        if model_info != None:
            last_epoch = model_info['last_epoch']
            train_losses, val_losses, val_accs = model_info['trlosshistory'], model_info['vallosshistory'], model_info['valacchistory']
            epoch_arr = list(range(last_epoch + 1, args.max_iter))
        else:
            epoch_arr = list(range(args.max_iter))
    
        for epoch_i in epoch_arr:
            avg_loss = 0
            print(f'Running epoch {epoch_i}')
            model.train()
            for i_batch, batch in enumerate(tqdm(train_dataloader, desc = '\tRunning through training set', position = 0, leave = True)):
                lbls = torch.tensor(batch[1]).type(torch.FloatTensor)
                #inds = torch.tensor(list(range(len(batch[1]))))
                #print(lbls.size(), lbls == 0)
                target = torch.zeros((len(batch[1]), 2))
                target[lbls == 1, 1] = 1.0
                target[lbls == 0, 0] = 1.0
                optimizer.zero_grad()
                #print(len(batch), type(batch[0]), len(batch[0]))
                img_batch = list(map(transform, batch[0]))
                img_batch = torch.stack(img_batch, 0)
                #print(img_batch.size())
                #tr_batch = transform(batch['img'])
                op = model(img_batch)
                loss = loss_fn(op, target)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                #print(op.size())
                #break
    
            train_losses.append(avg_loss / (i_batch + 1))
    
            with torch.no_grad():
                model.eval()
                val_acc = 0.0
                val_loss = 0.0
                for i_batch, batch in enumerate(tqdm(val_dataloader, desc = '\tRunning through validation set', position = 0, leave = True)):
                    lbls = torch.tensor(batch[1]).type(torch.FloatTensor)
                    target = torch.zeros((len(batch[1]), 2))
                    target[lbls == 1, 1] = 1.0
                    target[lbls == 0, 0] = 1.0
    
                    img_batch = list(map(transform, batch[0]))
                    img_batch = torch.stack(img_batch, 0)
                    #print(img_batch.size())
                    #tr_batch = transform(batch['img'])
                    op = model(img_batch)
                    loss = loss_fn(op, target)
                    lbls = torch.argmax(F.sigmoid(op), 1)
                    #print(lbls, op, loss)
                    val_acc += torch.eq(lbls, torch.tensor(batch[1])).sum()
                    val_loss += loss.item()
                val_acc /= val_len
                val_loss /= (i_batch + 1)
                val_losses.append(val_loss)
                val_accs.append(val_acc.item())
            #print(train_losses, val_losses, val_accs)
            chkpt_info = {
                'trlosshistory': train_losses,
                'vallosshistory': val_losses,
                'valacchistory': val_accs,
                'last_epoch': epoch_i
            }
            if val_loss < best_val_loss:
                save_experiment_output(model, chkpt_info, True, True)
            else:
                save_experiment_output(model, chkpt_info)
            save_model_helpers(optimizer.state_dict())
        save_experiment_output(model, chkpt_info, False)
        save_model_helpers(optimizer.state_dict(), False)
        print('\n\n')
        plt.plot(list(range(args.max_iter)), train_losses, color = 'red')
        plt.plot(list(range(args.max_iter)), val_losses, color = 'blue')
        plt.legend(['Training loss', 'Validation loss'])
        plt.title('Loss - Epoch plot')
        plt.show()
        print('\n\n\n')
        plt.clf()
        plt.plot(list(range(args.max_iter)), val_accs)
        plt.title('Validation accuracy plot')
        plt.show()
    
    def __run_test_loop(self, args, model):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.resize_size, args.resize_size)),
            transforms.Lambda(self.__convert_to_grascale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
        full_test_dataset, sm_test_dataset = self.__prepare_data('Test')
        test_len = len(full_test_dataset)
        test_loader = DataLoader(full_test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = self.__image_collate)
        test_acc = 0
        with torch.no_grad():
            model.eval()
            val_acc = 0.0
            val_loss = 0.0
            for i_batch, batch in enumerate(tqdm(test_loader, desc = 'Running through test set', position = 0, leave = True)):
                lbls = torch.tensor(batch[1]).type(torch.FloatTensor)
                target = torch.zeros((len(batch[1]), 2))
                target[lbls == 1, 1] = 1.0
                target[lbls == 0, 0] = 1.0
    
                img_batch = list(map(transform, batch[0]))
                img_batch = torch.stack(img_batch, 0)
                #print(img_batch.size())
                #tr_batch = transform(batch['img'])
                op = model(img_batch)
                lbls = torch.argmax(F.sigmoid(op), 1)
                #print(lbls, op, loss)
                test_acc += torch.eq(lbls, torch.tensor(batch[1])).sum()
            test_acc = test_acc / test_len
            print('\n\nTest accuracy: ', test_acc)
    
    def run_binary_classification_pipeline(self, args):
        full_dataset, smftr_dataset = self.__prepare_data()
        fulllen = len(full_dataset)
        smlen = len(smftr_dataset)
        idxs = list(range(fulllen))
        shuffle(idxs)
        vlen = int(0.2 * fulllen)
        val_idxs = idxs[:vlen]
        tr_idxs = idxs[vlen:]
        train_dataset = Subset(full_dataset, tr_idxs)
        val_dataset = Subset(full_dataset, val_idxs)
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = self.__image_collate)
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = self.__image_collate)
        model = get_model(2, args.model_name)
    
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.99, 0.999))
    
        model, optimizer, model_info = self.__get_experiment_chkpt(model, optimizer)
    
        self.__run_train_loop(args, model, optimizer, model_info, train_dataloader, val_dataloader, vlen)
    
        #model.load_state_dict(torch.load('./models/checkpoints/last_model.pt'))
        print('\n\n')
        self.__run_test_loop(args, model)
        
