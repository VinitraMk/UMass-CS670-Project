import matplotlib.pyplot as plt
import numpy as np

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





