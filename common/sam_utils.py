import matplotlib.pyplot as plt
import numpy as np
from common.utils import get_config
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from PIL import Image

CHECKPOINT_PATH='./models/weights/sam_vit_h_4b8939.pth'

MODEL_TYPE = "vit_h"

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_box(box, ax, processed_dim=False):
    x0, y0 = box[0], box[1]
    if processed_dim:
        w, h = box[2], box[3]
    else:
        w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
def show_mask(mask, ax):
    color = np.array([0, 0, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_all_masks(masks):
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones(
        (
            sorted_masks[0]["segmentation"].shape[0],
            sorted_masks[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for mask in sorted_masks:
        m = mask["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_sam_mask(img, img_fn):

    np_img = img.transpose(0, 2).transpose(0, 1).numpy()
    img_path = os.path.join(f'./source-data/segmentation/{img_fn}.png')
    print('otuside if', img_path)
    if os.path.exists(img_path):
        print('inside if', img_path)
        mask = np.array(Image.open(img_path))
        return mask[:, :, 0] / 255, np_img

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
    plt.imsave(f'./source-data/segmentation/{img_fn}.png', masks[0], cmap='binary')

    return masks[0], np_img




