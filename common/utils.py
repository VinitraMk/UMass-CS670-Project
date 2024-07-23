from torch.cuda import is_available
from torchvision.io import read_image
from PIL import Image
import torch
import os
import yaml
import json
from datetime import datetime
from torchvision.transforms import transforms

root_dir = ''
config_params = {}

def save_df_to_csv(df, filename, columns = []):
    if len(columns) == 0:
        columns = df.columns.tolist()
    df.to_csv(filename, index = False)
    
def read_imgtensor(imgpath):
    return read_image(imgpath)

def read_imgnp(imgpath):
    img = Image.open(imgpath)
    return img

def img2tensor(imgnparr):
    return torch.from_numpy(imgnparr)
   
def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)

def read_yaml(ypath):
    yml_params = {}
    with open(ypath, "r") as stream:
        try:
            yml_params = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)
    return yml_params

def dump_yaml(ypath, datadict):
    with open(ypath, 'w') as outfile:
        yaml.dump(datadict, outfile, default_flow_style=False)

def init_config():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'source-data')
    op_dir = os.path.join(root_dir, 'output')
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    config_params['root_dir'] = root_dir
    config_params['data_dir'] = data_dir
    config_params['output_dir'] = op_dir
    config_params['use_gpu'] = torch.cuda.is_available()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    config_params['device'] = device
    dump_yaml(config_path, config_params)
    cdir = os.path.join(root_dir, 'models/checkpoints')
    if not(os.path.exists(cdir)):
        os.mkdir(cdir)
    return config_params


def get_exp_params():
    yaml_fp = os.path.join(root_dir, 'run.yaml')
    exp_params = read_yaml(yaml_fp)
    return exp_params

def get_config():
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    return config_params

def save2config(key, val):
    config_params = get_config()
    config_params[key] = val
    config_path = os.path.join(root_dir, 'config.yaml')
    dump_yaml(config_path, config_params)

def get_accuracy(y_pred, y_true):
    c = torch.sum(y_pred == y_true)
    return c / len(y_true)

def get_error(y_pred, y_true):
    c = torch.sum(y_pred != y_true)
    return c / len(y_true)

def save_model_chkpt(model, chkpt_info, is_checkpoint = True, is_best = False):
    config_params = get_config()
    chkpt_filename = ''
    if is_checkpoint:
        fpath = os.path.join(config_params['root_dir'], 'models/checkpoints')
        chkpt_filename = 'curr_model'
    else:
        fpath = os.path.join(config_params['root_dir'], 'models/checkpoints')
        chkpt_filename = 'last_model'
        os.remove(os.path.join(config_params['root_dir'], 'models/checkpoints/curr_model.pt'))
        os.remove(os.path.join(config_params['root_dir'], 'models/checkpoints/curr_model.json'))
    
    mpath = os.path.join(fpath, f'{chkpt_filename}.pt')
    jpath = os.path.join(fpath, f'{chkpt_filename}.json')
    torch.save(
        model.state_dict(),
        mpath
    )
    with open(jpath, 'w') as fp:
        json.dump(chkpt_info, fp)
    if is_best:
        mpath = os.path.join(fpath, 'best_model.pt')
        torch.save(model.state_dict(), mpath)
        
def load_model(model_path):
    config_params = get_config()
    return torch.load(model_path, map_location = torch.device(config_params["device"]))
    
def get_modelinfo(is_chkpt = True):
    model_info = {}
    cfg = get_config()
    if is_chkpt:
        json_path = os.path.join(cfg["root_dir"], "models/checkpoints/curr_model.json")
    else:
        json_path = os.path.join(cfg["root_dir"], f"models/checkpoints/last_model.json")
    with open(json_path, 'r') as fp:
        model_info = json.load(fp)
    return model_info

def get_model_filename(model_name):
    now = datetime.now()
    nowstr = now.strftime("%d%m%Y%H%M%S")
    fname = f'{model_name}_{nowstr}'
    return fname

def save_experiment_output(model, chkpt_info, is_chkpoint = True, is_best = False):
    model_info = {
        'trlosshistory': chkpt_info['trlosshistory'],
        'vallosshistory': chkpt_info['vallosshistory'],
        'valacchistory': chkpt_info['valacchistory'],
        'last_epoch': chkpt_info['last_epoch']
    }
    save_model_chkpt(model, model_info, is_chkpoint, is_best)


def get_saved_model(model, model_filename, is_chkpt = True):
    cfg = get_config()
    if is_chkpt:
        model_dict = load_model(os.path.join(cfg["root_dir"], "models/checkpoints/curr_model.pt"))
    else:
        model_dict = load_model(os.path.join(cfg["root_dir"], f"models/checkpoints/last_model.pt"))
    model_state = model.state_dict()
    for key in model_dict:
        model_state[key] = model_dict[key]
    return model

def save_model_helpers(optimizer_state, is_chkpt = True):
    cfg = get_config()
    if is_chkpt:
        #mhpath = os.path.join(cfg["root_dir"], "models/checkpoints/curr_model_history.pt")
        opath = os.path.join(cfg["root_dir"], "models/checkpoints/curr_model_optimizer.pt")
    else:
        #mhpath = os.path.join(cfg["root_dir"], f"models/checkpoints/last_model_history.pt")
        opath = os.path.join(cfg["root_dir"], f"models/checkpoints/last_model_optimizer.pt")
        os.remove(os.path.join(cfg["root_dir"], "models/checkpoints/curr_model_optimizer.pt"))

    #torch.save(model_history, mhpath)
    torch.save(optimizer_state, opath)

def convert_to_grascale(img):
    imin, imax = img.min(), img.max()
    x = (img - imin) / (imax - imin)
    return x


def get_transforms(crop_size = 1000, rs_size = 256):
    transform = transforms.Compose([
        #transforms.CenterCrop(crop_size),
        #transforms.Resize(rs_size),
        transforms.Lambda(convert_to_grascale),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    inv_transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean = [ 0., 0., 0. ],
        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
            std = [ 1., 1., 1. ]),
        transforms.Lambda(convert_to_grascale),
        #transforms.ToPILImage(),
        ])

    return transform, inv_transform

def get_labels(filepath, label_arr, label_dict):
    li = len(label_arr) - 1
    with open(filepath, 'r') as fp:
        data_paths = fp.readlines()
        for line in data_paths:
            fn = line.split(' ')[0]
            info = fn.split('-')
            if info[-2].lower() not in label_arr:
                li += 1
                label_dict.append({
                    'name': info[-2].lower(),
                    'label_index': li
                })
                label_arr.append(info[-2].lower())
    return label_arr, label_dict
