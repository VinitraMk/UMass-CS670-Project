import torch
from models.ots_models import get_model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import io
from torchvision.io import read_image
from common.utils import convert_to_grascale, get_transforms, get_config
import json
from datautils.datareader import read_data
from datautils.dataset import COD10KDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
from common.sam_utils import show_mask, show_box
import numpy as np
from common.closed_form_matting import closed_form_matting_with_mask
import torch.nn.functional as F
from skimage.transform import resize
import torch.nn as nn

CHECKPOINT_PATH='./models/weights/sam_vit_h_4b8939.pth'

MODEL_TYPE = "vit_h"


def __get_gram_matrix(features):
    n,c,h,w = features.size()
    features = torch.reshape(features, (n, c, h*w))
    G = torch.matmul(features, features.transpose(1, 2))
    G /= (h * w * c)
    return G

def __content_loss(content_weight, curr_content, orig_content):
    return content_weight * F.mse_loss(curr_content, orig_content)
    #return content_weight * torch.sum((curr_content - orig_content)**2)

def __augmented_style_loss(features, style_layers, style_grams, style_weights, content_mask):
    cfg = get_config()
    if torch.cuda.is_available():
        stloss = torch.tensor(0.0).to('cuda')
    else:
        stloss = torch.tensor(0.0)

    scaled_mask = content_mask.to(cfg['device'])
    #scaled_mask.to(cfg['device'])
    for i in range(len(style_layers)):
        stlyr = features[style_layers[i]].clone()
        stlyr = stlyr * scaled_mask
        gm = __get_gram_matrix(stlyr)
        N = torch.numel(gm) ** 2
        #print(type(N))
        #stloss += (style_weights[i] * torch.sum((style_grams[i] - gm)**2))a
        #print(style_grams[i].size(), gm.size())
        tmploss = F.mse_loss(gm, style_grams[i])
        #stloss += (0.5 * (1 / N) * (torch.sum((gm - style_grams[i])**2, (1, 2))))[0]
        #print('tmploss', tmploss)
        stloss += 0.5 * (1 / N) * tmploss
        scaled_mask = F.interpolate(scaled_mask, scale_factor = 0.5, mode = 'nearest', antialias = False)

    return stloss


def __style_loss(features, style_layers, style_grams, style_weights):

    if torch.cuda.is_available():
        stloss = torch.tensor(0.0).to('cuda')
    else:
        stloss = torch.tensor(0.0)

    for i in range(len(style_layers)):
        stlyr = features[style_layers[i]].clone()
        gm = __get_gram_matrix(stlyr)
        #stloss += (style_weights[i] * torch.sum((style_grams[i] - gm)**2))
        stloss += F.mse_loss(gm, style_grams[i])

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

def __get_mask(np_img, fn):

    cfg = get_config()
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=cfg['device'])
    mask_generator = SamAutomaticMaskGenerator(sam)

    #plt.imshow(np_img)
    #plt.show()
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(np_img)
    w, h, _ = np_img.shape
    #print(image_rgb.shape)


    # Predict mask with bounding box prompt
    bbox_prompt = np.array([0, 0, h, w])
    masks, scores, logits = mask_predictor.predict(
    box=bbox_prompt,
    multimask_output=False
    )

    # Plot the bounding box prompt and predicted mask
    #plt.imshow(np_img)
    #show_mask(masks[0], plt.gca())
    #show_box(bbox_prompt, plt.gca())
    #plt.show()

    #plt.imshow(masks[0], cmap='binary')
    #plt.show()
    plt.imsave(f'./source-data/segmentation/{fn}.png', masks[0], cmap='binary')

    return masks[0]

def __get_tensor_from_sparse_scipy(scipy_sparse_matrix):

    cfg = get_config()

    data = torch.tensor(scipy_sparse_matrix.data, dtype = torch.float32)
    col = torch.tensor(scipy_sparse_matrix.col, dtype = torch.long)
    row = torch.tensor(scipy_sparse_matrix.row, dtype = torch.long)
    indices = torch.stack([row, col]).to("cpu")

    i = torch.LongTensor(indices)
    shape = scipy_sparse_matrix.shape

    return torch.sparse_coo_tensor(indices, data, shape)
    #return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def deep_style_transfer(content_img, style_img, style_layers, content_layer, content_weight, style_weights, lamda, eta, args):

    cfg = get_config()


    style_img = F.interpolate(style_img[None], size = 512, mode = 'bilinear', antialias = False)[0]
    content_img = F.interpolate(content_img[None], size = 512, mode = 'bilinear', antialias = False)[0]

    og_content_img = content_img# / 255.0
    og_style_img = style_img #/ 255.0

    np_style_img = style_img.transpose(0, 2).transpose(0, 1).numpy()
    np_content_img = content_img.transpose(0, 2).transpose(0, 1).numpy()
    style_mask = __get_mask(np_style_img, 'tar').astype(float)
    content_mask = __get_mask(np_content_img, 'in').astype(float)
    print('masks', style_mask.min(), content_mask.max(), content_mask.min(), content_mask.max())
    content_mask_bg = (content_mask == 1).astype(float)
    style_mask_bg = (style_mask == 1).astype(float)
    content_img_bg = np_content_img * np.expand_dims(content_mask_bg, 2)
    style_img_bg = np_style_img * np.expand_dims(style_mask_bg, 2)
    #plt.imshow(content_img_bg/255.0)
    #plt.show()
    #Mc = closed_form_matting_with_mask(np_content_img, content_mask)
    Mc = closed_form_matting_with_mask(content_img_bg, content_mask)
    Mc = __get_tensor_from_sparse_scipy(Mc).to(cfg['device'])
    #print('images', np_style_img.shape, np_content_img.shape, style_mask.shape, content_mask.shape)
    #print('laplacians', Mc.size(), type(Mc))
    #style_mask = torch.from_numpy(style_mask)[None, None, :].float()
    #content_mask = torch.from_numpy(content_mask)[None, None, :].float()
    style_mask_bg = torch.from_numpy(style_mask_bg)[None, None, :].float()
    content_mask_bg = torch.from_numpy(content_mask_bg)[None, None, :].float()
    style_img = torch.from_numpy(style_img_bg).transpose(2, 0).transpose(2, 1) #/ 255.0
    content_img = torch.from_numpy(content_img_bg).transpose(2, 0).transpose(2, 1) #/ 255.0
    #print('content sizes', style_img.size(), content_img.size(), og_content_img.size(), og_style_img.size())
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
    style_features = __get_features(style_img, model_features)
    #style_mask_features = __get_features()
    style_grams = []

    scaled_style_mask = style_mask_bg.to(cfg['device'])
    #scaled_style_mask.to(cfg['device'])
    for i in style_layers:
        fm = style_features[i].clone() * scaled_style_mask
        stg = __get_gram_matrix(fm)
        #print('style layers: ', i, stg.size(), features[i].size())
        style_grams.append(stg)
        scaled_style_mask = F.interpolate(scaled_style_mask, scale_factor = 0.5, mode = 'nearest', antialias = False)

    new_img = content_img[None, :].clone().type(dtype)
    new_img.requires_grad_(True)
    print('ns', new_img.size())
    optimizer = torch.optim.LBFGS([new_img], lr = args.lr)

    losses = []
    t = [0]

    while t[0] <= args.max_iter:
        #if t < (args.max_iter - 10):
        #new_img.data.clamp_(0, 1)
        def closure():
            with torch.no_grad():
                #print('ns', new_img.size())
                new_img.clamp_(0, 1)
            optimizer.zero_grad()
            features = __get_features(new_img, model_features)
            closs = __content_loss(content_weight, features[content_layer], content_trgt)
            #sloss = eta * __style_loss(features, style_layers, style_grams, style_weights)
            sloss = eta * __augmented_style_loss(features, style_layers, style_grams, style_weights, content_mask_bg)
            #print('augmented style loss', sloss.size())
            #tloss = __tv_loss(new_img, tv_weight)
            loss = closs + sloss #+ tloss
            loss.backward()

            '''
            img_vec = new_img.reshape(-1, 3).transpose(0, 1)
            #print('img vec', img_vec.size(), Mc.size(), img_vec.get_device(), Mc.get_device())
            dotsum = 0
            preg_grad = torch.zeros(img_vec.size()).T.to(cfg['device'])
            for i in range(3):
                im_row = img_vec[i, :][None, :]
                grad = torch.sparse.mm(Mc, im_row.T)
                #print('grad size', grad.size())
                preg_grad += grad
                dotprod = torch.einsum('ij,jk->i', im_row, grad)
                dotsum += dotprod[0]
            ploss = lamda * dotsum
            preg_grad = 2 * lamda * preg_grad
            preg_grad = preg_grad.reshape(new_img.size())
            #print('preg', preg_grad.size(), ploss.size(), new_img.grad.size())
            loss += ploss
            new_img.grad += preg_grad
            '''
            t[0] += 1
            if t[0] % args.iter_interval == 0:
                print('\n\nIteration {}'.format(t))
                plt.axis('off')
                print('Losses: ', closs.item(), sloss.item(), loss.item())
                #print(new_img.data.size(), t)
                rescaled_img = c_inv_transform(new_img.data.cpu())
                print('rs min max', rescaled_img.min(), rescaled_img.max())
                #print('rs', rescaled_img.size())
                rescaled_img = rescaled_img.transpose(0, 2).transpose(0, 1)
                #print(rescaled_img.size())
                plt.imshow(rescaled_img)
                plt.show()
            return loss

        optimizer.step(closure)

    print('ns size', new_img.data.min(), new_img.data.max())
    with torch.no_grad():
        new_img.clamp_(0, 1)
    print('ns size', new_img.data.min(), new_img.data.max())
    print('\n\nIteration {}'.format(t))
    plt.axis('off')
    #print('Losses: ', closs.item(), sloss.item(), loss.item())

    rescaled_img = c_inv_transform(new_img.data.cpu())
    print('ns size', rescaled_img.min(), rescaled_img.max(), og_content_img.min(), og_content_img.max())
    #print('og sizes', content_mask_bg.size(), og_content_img.size(), rescaled_img.size())
    #print('rs', rescaled_img.size())
    rescaled_img = (rescaled_img * content_mask_bg[0]) + (og_content_img * (1 - content_mask_bg[0]))
    rescaled_img = rescaled_img.transpose(0, 2).transpose(0, 1)
    plt.imshow(rescaled_img)
    plt.show()

    new_img = new_img.detach().cpu()
    style_img = style_img.cpu()
    content_img = content_img.cpu()
    del style_img, content_img, new_img

    '''
    print('\n\n\n Loss plots')
    plt.clf()
    plt.plot(list(range(args.max_iter)), losses, color='b')
    plt.plot(list(range(args.max_iter)), closses, color='r')
    plt.plot(list(range(args.max_iter)), slosses, color='g')
    plt.plot(list(range(args.max_iter)), tlosses, color='y')
    plt.legend(['Total loss', 'Content loss', 'Style loss', 'Total variation loss'])
    plt.show()
    '''


def style_transfer(content_img, style_img, style_layers, content_layer, content_weight, style_weights, tv_weight, args):

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        device = 'cuda'
    else:
        dtype = torch.FloatTensor
        device = 'cpu'

    model, _ = get_model(args.model_name, True)
    model_features = model.features.eval()
    model_features.type(dtype)

    for layer in model_features.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False

    for param in model_features.parameters():
        param.requires_grad = False

    style_img = style_img.type(dtype) / 255.0 #read_image('./data/Textures/tree-bark.jpg').type(dtype)
    content_img = content_img.type(dtype) / 255.0 #read_image('./data/Mini-Set/butterfly-image.jpg').type(dtype)
    #print('img sizes', style_img.size(), content_img.size())

    style_img = F.interpolate(style_img[None], size = 512, mode = 'bilinear', antialias = False)
    content_img = F.interpolate(content_img[None], size = 512, mode = 'bilinear', antialias = False)

    c_transform, c_inv_transform = get_transforms()
    content_img = c_transform(content_img)
    features = __get_features(content_img, model_features)
    content_trgt = features[content_layer].clone()
    s_transform, _ = get_transforms()
    style_img = s_transform(style_img)#[None]
    features = __get_features(style_img, model_features)
    style_grams = []

    for i in style_layers:
        style_grams.append(__get_gram_matrix(features[i].clone()))

    new_img = content_img.clone().type(dtype)
    new_img.requires_grad_(True)
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
        #print('featrs', new_img.size(), features[content_layer].size(), content_trgt.size())
        closs = __content_loss(content_weight, features[content_layer], content_trgt)
        sloss = __style_loss(features, style_layers, style_grams, style_weights)
        tloss = __tv_loss(new_img, tv_weight)
        loss = closs + (100 * sloss) + tloss
        losses.append(loss.cpu().detach().numpy())
        closses.append(closs.cpu().detach().numpy())
        tlosses.append(tloss.cpu().detach().numpy())
        slosses.append(sloss.cpu().detach().numpy())
        #print('loss grad', loss.requires_grad)
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print('\n\nIteration {}'.format(t))
            plt.axis('off')
            #print('Losses: ', closs.item(), sloss.item(), loss.item())
            rescaled_img = c_inv_transform(new_img.data.cpu())
            rescaled_img = rescaled_img.transpose(0, 2).transpose(0, 1)
            plt.imshow(rescaled_img)
            plt.show()

    print('\n\nIteration {}'.format(t))
    plt.axis('off')
    rescaled_img = c_inv_transform(new_img.data.cpu())
    rescaled_img = rescaled_img.transpose(0, 2).transpose(0, 1)
    plt.imshow(rescaled_img)
    plt.show()
    print('\n\nContent loss: ', closs.item(), '| Style Loss: ', sloss.item(), '| Total variation loss:', loss.item())
    
    new_img = new_img.detach().cpu()
    style_img = style_img.cpu()
    content_img = content_img.cpu()
    del style_img, content_img, new_img


    '''
    print('\n\n\n Loss plots')
    plt.clf()
    plt.plot(list(range(args.max_iter)), losses, color='b')
    plt.plot(list(range(args.max_iter)), closses, color='r')
    plt.plot(list(range(args.max_iter)), slosses, color='g')
    plt.plot(list(range(args.max_iter)), tlosses, color='y')
    plt.legend(['Total loss', 'Content loss', 'Style loss', 'Total variation loss'])
    plt.show()
    '''
    #return rescaled_img


def run_style_transfer_pipeline(args,
    texture_name, style_weight, last_batch_run = -1):

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


