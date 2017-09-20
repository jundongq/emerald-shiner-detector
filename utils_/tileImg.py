import numpy as np


def tile_images(img_dict, nb_rows, nb_cols):

	"""
	input::
	img_dict: a dictionary with key as the img slice name, value as the 3D matrix of each img slice
	nb_rows: number of rows of img slice
	nb_cols: number of cols of img slice

	output:
	tiled image: with format of (width, height, channels)
	"""
	keys = img_dict.keys()

	(unit_height, unit_width) = np.shape(img_dict[keys[0]])[:2]
	
	tiled_width  = nb_cols*unit_width
	tiled_height = nb_rows*unit_height
	
	tiled_img = np.zeros((tiled_height, tiled_width, 3))

	sorted_keys = sorted(keys)
	for i in range(nb_rows):
		for j in range(nb_cols):
			k = 'slice_{}_{}'.format(i+1,j+1)
			tiled_img[i*unit_height:(i+1)*unit_height, j*unit_width:(j+1)*unit_width, :] = img_dict[k]

	return tiled_img
