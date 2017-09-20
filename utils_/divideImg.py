import numpy as np


def divideImg(image, nb_slices):
    """
    input::
    image: in format of 3d array (height, widht, channel)
    nb_slices: scaler that determins how many sub images needed to be created
    
    output::
    img_dict: a dictionay containing nb_slices elements, with key as name of sub image, value as the corresponding array
    """
    
    # determine the number of rows and cols
    factors = [x for x in range(1, nb_slices+1) if nb_slices%x==0]
    nb_sqrt = np.sqrt(nb_slices)
    if nb_sqrt in factors:
        nb_rows = nb_cols = np.int(np.sqrt(nb_slices))
    else:
        idx = np.int(len(factors)/2) - 1
        a = factors[idx]
        b = np.int(nb_slices/a)

        nb_rows = min(a, b)
        nb_cols = max(a,b)
    # print nb_rows, nb_cols 
    
    # obtain the shape of image
    height, width = np.shape(image)[:2]
    
    unit_height = height/nb_rows
    unit_width  = width/nb_cols
    
    # slice the image from left to right
    img_dict = {}
    for i in range(nb_rows):
        for j in range(nb_cols):
            k = 'slice_{}_{}'.format(i+1,j+1)
            v = image[i*unit_height:(i+1)*unit_height, j*unit_width:(j+1)*unit_width, :]
            img_dict[k] = v
    
    return img_dict, nb_rows, nb_cols
