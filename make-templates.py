import os
import numpy as np
from PIL import Image

data_path = './data/RedLights2011_Medium'
template_imgs_dir = './templates/red-light'
template_img_files = sorted(os.listdir(template_imgs_dir))
template_img_files = [f for f in template_img_files if '.jpg' in f]

for i, filename in enumerate(template_img_files):
    I = Image.open(os.path.join(data_path, filename))
    template = Image.open(os.path.join(template_imgs_dir, filename))

    I = np.asarray(I)
    template = np.asarray(template)

    mean = np.mean(I, axis=(0, 1))
    std  = np.std(I, axis=(0, 1))

    template = (template - mean) / std

    print(filename, mean, std, np.mean(template), np.std(template))

    np.save(os.path.join(template_imgs_dir, f'template{i}'), # chop off '.jpg'
            template)


data_path = './data/RedLights2011_Medium'