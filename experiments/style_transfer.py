import torch
from models.ots_models import get_model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import io
from torchvision.io import read_image
from common.utils import convert_to_grascale, get_transforms
import json
from datautils.datareader import read_data
from datautils.dataset import COD10KDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def __get_gram_matrix(features):
    n,c,h,w = features.size()
    features = torch.reshape(features, (n, c, h*w))
    G = torch.matmul(features, features.transpose(1, 2))
    G /= (h * w * c)
    return G


def __content_loss(content_weight, curr_content, orig_content):
    return content_weight * torch.sum((curr_content - orig_content)**2)

def __style_loss(features, style_layers, style_grams, style_weights):

    if torch.cuda.is_available():
        stloss = torch.tensor(0.0).to('cuda')
    else:
        stloss = torch.tensor(0.0)

    for i in range(len(style_layers)):
        stlyr = features[style_layers[i]].clone()
        gm = __get_gram_matrix(stlyr)
        stloss += (style_weights[i] * torch.sum((style_grams[i] - gm)**2))

    return stloss

def __tv_loss(img, tv_weight):
    _, _, h, w = img.size()
    lt = torch.sum((img[:, :, 1:h, :] - img[:, :, :h-1, :])**2)
    rt = torch.sum((img[:, :, :, 1:w] - img[:, :, :, :w-1])**2)
    tvloss = tv_weight * (lt + rt)
    return tvloss

def __get_features(img, model_features):
    features = []
    x = img

    for _, layer in enumerate(model_features._modules.values()):
        op = layer(x)
        features.append(op)
        x = op

    return features

def style_transfer(content_img, style_img, style_layers, content_layer, content_weight, style_weights, tv_weight, args):

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        device = 'cuda'
    else:
        dtype = torch.FloatTensor
        device = 'cpu'

    model, _ = get_model(args.model_name, True)
    model_features = model.features
    model_features.type(dtype)

    for param in model_features.parameters():
        param.requires_grad = False

    style_img = style_img.type(dtype) #read_image('./data/Textures/tree-bark.jpg').type(dtype)
    content_img = content_img.type(dtype) #read_image('./data/Mini-Set/butterfly-image.jpg').type(dtype)
    #print('img sizes', style_img.size(), content_img.size())
    c_transform, c_inv_transform = get_transforms()
    content_img = c_transform(content_img)
    #print('cs after', content_img.shape)
    features = __get_features(content_img, model_features)
    content_trgt = features[content_layer].clone()

    s_transform, _ = get_transforms()
    style_img = s_transform(style_img)[None]
    features = __get_features(style_img, model_features)
    style_grams = []

    for i in style_layers:
        style_grams.append(__get_gram_matrix(features[i].clone()))

    new_img = content_img.clone().type(dtype)
    new_img.requires_grad_(True)
    #print('ns', new_img.shape)
    optimizer = torch.optim.Adam([new_img], lr = args.lr)

    losses = []
    closses = []
    tlosses = []
    slosses = []

    for t in range(args.max_iter):
        #if t < (args.max_iter - 10):
        new_img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()
        features = __get_features(new_img, model_features)
        closs = __content_loss(content_weight, features[content_layer], content_trgt)
        sloss = __style_loss(features, style_layers, style_grams, style_weights)
        tloss = __tv_loss(new_img, tv_weight)
        loss = closs + (100* sloss) + tloss
        losses.append(loss.cpu().detach().numpy())
        closses.append(closs.cpu().detach().numpy())
        tlosses.append(tloss.cpu().detach().numpy())
        slosses.append(sloss.cpu().detach().numpy())
        #print('loss grad', loss.requires_grad)
        loss.backward()
        optimizer.step()

    

        '''
        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            print('Losses: ', closs.item(), sloss.item(), loss.item())
            rescaled_img = c_inv_transform(new_img.data.cpu())
            plt.imshow(rescaled_img)
            plt.show()
        '''

    rescaled_img = c_inv_transform(new_img.data.cpu())
    new_img = new_img.detach().cpu()
    style_img = style_img.cpu()
    content_img = content_img.cpu()
    del style_img, content_img
    '''
    print('Iteration {}'.format(t))
    plt.axis('off')
    print('Losses: ', closs.item(), sloss.item(), loss.item())
    plt.imshow(rescaled_img)
    plt.show()

    print('\n\n\n Loss plots')
    plt.clf()
    plt.plot(list(range(args.max_iter)), losses, color='b')
    plt.plot(list(range(args.max_iter)), closses, color='r')
    plt.plot(list(range(args.max_iter)), slosses, color='g')
    plt.plot(list(range(args.max_iter)), tlosses, color='y')
    plt.legend(['Total loss', 'Content loss', 'Style loss', 'Total variation loss'])
    plt.show()
    '''
    return rescaled_img
            
def run_style_transfer_pipeline(args, texture_name, style_weight, last_batch_run = -1):
    pos_data_paths = read_data('Train')

    dataset = COD10KDataset(pos_data_paths)

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)

    for i_batch, batch in enumerate(dataloader):
        if i_batch > last_batch_run:
            print(f"Processing batch {i_batch}, image: {batch['img_name']} of dimensions: {batch['img'].shape}")
            style_img = read_image(f'./source-data/Textures/{texture_name}.jpg')
            new_img = style_transfer(
                batch['img'],
                style_img,
                [0, 2, 5, 14, 23],
                21,
                1e-4,
                [style_weight]*5,
                1e-5,
                args)
            img_name = batch['img_name'][0]
            img_name = img_name.replace(".jpg", "")
            img_name = f"./source-data/Train/Styled-Image/{img_name}-Texture-{texture_name}.jpg"
            save_image(new_img, img_name)
            last_run = {
                "last_batch": i_batch
            }
            print(f"completing transfer of img {i_batch} with texture {texture_name}")
            with open("./last_run_info.json", "w") as fp:
                json.dump(last_run, fp)

    print(f'Finished modifying train dataset images for {texture_name}')
    last_run = {
        "last_batch": -1
    }
    with open("./last_run_info.json", "w") as fp:
        json.dump(last_run, fp)
    return -1

