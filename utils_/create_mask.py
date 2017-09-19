import numpy as np

def create_mask(image, bounding_boxes):
	"""
	input::
	image: as a numpy ndarray, in format (height, widht, channel)
	bounding_boxes: an ndarray with each row containing a list of corners of a bounding box
	output::
	mask_image: as a binary mask with elements inside of bounding boxes equal to 1, others are 0.
	"""
	mask_image = np.zeros(np.shape(image)[:2])

	for i in range(len(bounding_boxes)):
		mask_image[bounding_boxes[i][0]:bounding_boxes[i][2], bounding_boxes[i][1]:bounding_boxes[i][3]] = 1
	return mask_image