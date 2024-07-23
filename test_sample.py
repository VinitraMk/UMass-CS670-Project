from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

rgb_img = np.array(Image.open('./source-data/TestDataset/CHAMELEON/Imgs/animal-43.jpg'))
mask_img = np.array(Image.open('./output/Test-SemSeg/CHAMELEON/animal-43.png'))[:, :, :3]
overlay = mask_img * rgb_img

print(rgb_img.shape, mask_img.shape)

plt.imshow(overlay)
plt.show()