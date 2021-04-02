import os
import numpy as np
import json
from PIL import Image

templates_dir = './templates/red-light'
template_files = sorted(os.listdir(templates_dir))
templates = [np.load(os.path.join(templates_dir, f)) for f in template_files if 'template' in f]

def detect_red_light(I, visualize=False):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    cpy = I.copy()

    # Convert to float so that we can normalize
    I = I.astype(np.float32)

    # Normalize image
    # It would *probably* be better to normalize with respect
    # to the rest of the dataset instead.
    mean = np.mean(I, axis=(0, 1))
    std  = np.std(I, axis=(0, 1))
    I = (I - mean) / std

    # Resize a np.array to the given width and height
    def resize(arr, w, h):
        shape = list(arr.shape)
        shape[:2] = [h, w]
        ret = np.zeros(shape)
        old_h, old_w = arr.shape[:2]
        for x in range(w):
            for y in range(h):
                old_x = int(x / w * old_w)
                old_y = int(y / h * old_h)
                ret[y][x] = arr[old_y][old_x]
        return ret

    # Return a list of template side lengths that we will try
    # to match in the image.
    def make_templates_sizes():
        # I used to have something more complicated but
        # it's not really necessary for this assignment
        return [6, 9, 15, 24, 36, 54, 72]

    # Return (distinct templates) cross (side lengths)
    def make_templates_sized(templates):
        templates_sized = list()
        sizes = make_templates_sizes()
        for size in sizes:
            for template in templates:
                templates_sized.append(resize(template, size, size))
        return templates_sized

    # consider resizing image instead

    h, w, _ = I.shape
    for template in make_templates_sized(templates):

        kernel_size = template.shape[0]
        stride = int(kernel_size / 5) + 1

        # The dot product above which we accept the detection.
        threshold = 2.5 * (kernel_size ** 2)

        for x in range(0, w-kernel_size, stride):
            for y in range(0, h-kernel_size, stride):

                # Patch of the original image
                target = I[y:y+kernel_size, x:x+kernel_size, :]

                # How similar it is to the template
                dot = np.sum(target * template)

                # Object detected
                if dot > threshold:

                    # Sorry for this illegible mess
                    # This is me trying to deal with overlapping boxes.
                    # If the old box's center is within this box,
                    #   we eat the old box.
                    new_box = [y, x, y+kernel_size-1, x+kernel_size-1]
                    old_bounding_boxes = bounding_boxes.copy()
                    for i, box in enumerate(old_bounding_boxes):
                        tl_y, tl_x, br_y, br_x = tuple(box)
                        old_center_y = int((tl_y + br_y) / 2)
                        old_center_x = int((tl_x + br_x) / 2)
                        if old_center_y >= y and old_center_y <= new_box[2] and\
                           old_center_x >= x and old_center_x <= new_box[3]:
                            new_box = [min(tl_y, y), min(tl_x, x),
                                       max(br_y, new_box[2]),
                                       max(br_x, new_box[3])]
                            bounding_boxes.remove(box)
                    bounding_boxes.append(new_box)

    # Returns a green outline to be pasted on top of detections
    def make_outline(h, w):
        outline = np.zeros(shape=(h, w, 3))
        outline[:, :, 1] = 255
        outline[2:-2, 2:-2, 1] = 0
        return outline

    if visualize:
        for box in bounding_boxes:
            y0, x0, y1, x1 = tuple(box)
            cutout = cpy[y0:y1+1, x0:x1+1, :]
            cutout = np.maximum(cutout, make_outline(y1-y0+1, x1-x0+1))
            cpy[y0:y1+1, x0:x1+1, :] = cutout
        Image.fromarray(cpy).show()

    ''' END YOUR CODE '''

    for box in bounding_boxes:
        assert len(box) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './data/predictions'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for file_name in file_names:

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_name))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_name] = detect_red_light(I, visualize=False)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds, f)
