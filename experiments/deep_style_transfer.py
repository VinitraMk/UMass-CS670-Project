import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import numpy as np
import copy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from common.sam_utils import show_mask, show_box
from common.closed_form_matting import closed_form_matting_with_mask
from common.utils import get_config

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

device = get_config()['device']

CHECKPOINT_PATH='./models/weights/sam_vit_h_4b8939.pth'

MODEL_TYPE = "vit_h"


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class AugmentedStyleLoss(nn.Module):
    """
    AugmentedStyleLoss exploits the semantic information of images.
    See Luan et al. for the details.
    """

    def __init__(self, target_feature, target_masks, input_masks):
        super(AugmentedStyleLoss, self).__init__()
        self.input_masks = [mask.detach() for mask in input_masks]
        self.targets = [
            gram_matrix(target_feature * mask).detach() for mask in target_masks
        ]

    def forward(self, input):
        gram_matrices = [
            gram_matrix(input * mask.detach()) for mask in self.input_masks
        ]
        self.loss = sum(
            F.mse_loss(gram, target)
            for gram, target in zip(gram_matrices, self.targets)
        )
        return input


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image#.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def get_mask(np_img, fn):

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)
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

def get_tensor_from_sparse_scipy(scipy_sparse_matrix):

    cfg = get_config()

    data = torch.tensor(scipy_sparse_matrix.data, dtype = torch.float32)
    col = torch.tensor(scipy_sparse_matrix.col, dtype = torch.long)
    row = torch.tensor(scipy_sparse_matrix.row, dtype = torch.long)
    indices = torch.stack([row, col]).to("cpu")

    i = torch.LongTensor(indices)
    shape = scipy_sparse_matrix.shape

    return torch.sparse_coo_tensor(indices, data, shape)
    #return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def main(style_filename = './source-data/Textures/red-sand.jpg', content_filename = './source-data/Mini-Set/bgm-bird-2.jpg'):

    #device = get_config()['device']
    print('device', device)
    style_img = image_loader(style_filename)
    print(style_img.min(), style_img.max())
    content_img = image_loader(content_filename)
    print(content_img.min(), content_img.max())
    style_img = F.interpolate(style_img, size = 512, mode = 'bilinear', antialias = False)
    content_img = F.interpolate(content_img, size = 512, mode = 'bilinear', antialias = False)
    og_content_img = content_img.clone()

    np_style_img = style_img[0].transpose(0, 2).transpose(0, 1).numpy()
    np_content_img = content_img[0].transpose(0, 2).transpose(0, 1).numpy()
    style_mask = get_mask(np_style_img, 'tar')#.astype(float)
    content_mask = get_mask(np_content_img, 'in')#.astype(float)
    print('masks', style_mask.min(), content_mask.max(), content_mask.min(), content_mask.max())
    content_mask_bg = (content_mask == 1).astype(float)
    style_mask_bg = (style_mask == 1).astype(float)
    content_img_bg = np_content_img * np.expand_dims(content_mask_bg, 2)
    style_img_bg = np_style_img * np.expand_dims(style_mask_bg, 2)
    #plt.imshow(content_img_bg/255.0)
    #plt.show()
    #Mc = closed_form_matting_with_mask(np_content_img, content_mask)
    Mc = closed_form_matting_with_mask(content_img_bg, content_mask)
    Mc = get_tensor_from_sparse_scipy(Mc).to(device)
    #print('images', np_style_img.shape, np_content_img.shape, style_mask.shape, content_mask.shape)
    #print('laplacians', Mc.size(), type(Mc))
    #style_mask = torch.from_numpy(style_mask)[None, None, :].float()
    #content_mask = torch.from_numpy(content_mask)[None, None, :].float()
    style_mask_bg = torch.from_numpy(style_mask_bg)[None, None, :].float()
    content_mask_bg = torch.from_numpy(content_mask_bg)[None, None, :].float()
    style_img = torch.from_numpy(style_img_bg).transpose(2, 0).transpose(2, 1) #/ 255.0
    content_img = torch.from_numpy(content_img_bg).transpose(2, 0).transpose(2, 1) #/ 255.0
    style_img = style_img.to(device, torch.float)[None, :]
    content_img = content_img.to(device, torch.float)[None, :]

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    '''
    plt.ion()

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')
    '''

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device, torch.float)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # desired depth layers to compute style/content losses :
    # content_layers_default = ['conv_4']
    # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    input_img = content_img.clone()
    # if you want to use white noise by using the following code:
    #
    # .. code-block:: python
    #
    #    input_img = torch.randn(content_img.data.size())

    # add the original input image to the figure:
    #plt.figure()
    #imshow(input_img, title='Input Image')

    style_weight = 1e6
    content_weight = 1e1

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, 300, style_weight, content_weight)
    
    output = output.to("cpu")
    output = (output * content_mask_bg) + (og_content_img[0] * (1 - content_mask_bg))
    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

