import os
import numpy as np
import json
from PIL import Image

templates_dir = './templates/red-light'
template_files = sorted(os.listdir(templates_dir))
templates = [np.load(os.path.join(templates_dir, f)) for f in template_files if 'template' in f]

def detect_red_light(I):
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

    ''' BEGIN YOUR CODE '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    cpy = I.copy()
    I = I.astype(np.float32)

    # Subtract out average
    mean = np.mean(I, axis=(0, 1))
    std  = np.std(I, axis=(0, 1))
    I = (I - mean) / std

    def resize(arr, w, h):
        shape = list(arr.shape)
        shape[:2] = [w, h]
        ret = np.zeros(shape)
        old_w, old_h = arr.shape[:2]
        for x in range(w):
            for y in range(h):
                old_x = int(x / w * old_w)
                old_y = int(y / h * old_h)
                ret[x][y] = arr[old_x][old_y]
        return ret

    def make_templates_sizes(max_size):
        template_sizes = list()
        size = 4
        while size < max_size:
            template_sizes.append(int(size))
            size *= 1.5
        return [4, 6, 9, 15, 24, 36, 54, 72]
        return template_sizes

    def make_templates_sized(templates):
        templates_sized = list()
        sizes = make_templates_sizes(min(I.shape[:2]))
        for template in templates:
            for size in sizes:
                templates_sized.append(resize(template, size, size))
        return templates_sized

    w, h, _ = I.shape
    for template in make_templates_sized(templates[:2]):
        kernel_size = template.shape[0]
        stride = int(kernel_size / 5) + 1
        threshold = 2.5 * (kernel_size ** 2)
        for x in range(0, w-kernel_size, stride):
            for y in range(0, h-kernel_size, stride):
                target = I[x:x+kernel_size, y:y+kernel_size, :]
                dot = np.sum(target * template)
                if dot > threshold:
                    print(x, y, int(dot), dot / (kernel_size ** 2))
                    cpy[x:x+kernel_size, y:y+kernel_size, 1] = 128

    Image.fromarray(cpy).show()

    """
    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        bounding_boxes.append([tl_row,tl_col,br_row,br_col])
    """

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
for file_name in file_names[:10]:

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_name))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_name] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds, f)
