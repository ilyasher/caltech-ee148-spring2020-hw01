import os
import numpy as np
import json
from PIL import Image

imgs_path = './data/RedLights2011_Medium/'
imgs = ['010', '044', '105', '102']

preds_path = './data/predictions'
with open(os.path.join(preds_path,'preds.json'),'r') as f:
    preds = json.load(f)

# Returns a green outline to be pasted on top of detections
def make_outline(h, w):
    outline = np.zeros(shape=(h, w, 3))
    outline[:, :, 1] = 255
    outline[2:-2, 2:-2, 1] = 0
    return outline

for num in imgs:
    filename = f'RL-{num}.jpg'
    boxes = preds[filename]
    path = os.path.join(imgs_path, filename)
    I = Image.open(path)
    I = np.asarray(I).copy()

    for box in boxes:
        y0, x0, y1, x1 = tuple(box)
        cutout = I[y0:y1+1, x0:x1+1, :]
        cutout = np.maximum(cutout, make_outline(y1-y0+1, x1-x0+1))
        I[y0:y1+1, x0:x1+1, :] = cutout
    Image.fromarray(I).show()

