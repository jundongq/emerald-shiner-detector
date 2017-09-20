import numpy as np



def coordinates_extract_transform(image_slice, image_slice_key, boxes, scores, score_criterion=0.1, enlarge_bndbox=True, enlarge_extent=0):
	"""
	input::
	as sliced image path, and bounding boxes coordinates (0 ~ 1)
	image_path: the directory of each sliced image (patch) of an orignal large image
	boxes: a tuple containing 100 relative coordinates in each image, in form of (xmin, ymin, xmax, ymax)
	scores: iou value

	output::
	as a numpy asrray, the bounding boxes coordinates in the orignal image
	"""

	# image = Image.open(image_path)
	im_height, im_width = np.shape(image_slice)[:2]

	# select boxes that satisfy the score_criterion
	selected_boxes = boxes[0][scores[0] > score_criterion]

	if len(selected_boxes) != 0:

		if enlarge_bndbox:
			# patch_bndbox_coordinates contains the coordinates of all bounding boxes in the input image
			# each row of patch_bndbox_coordinates contain four corners of a single bounding box
			patch_bndbox_coordinates = np.asarray([map(int,(selected_boxes[i][0]*im_height-enlarge_extent, selected_boxes[i][1]*im_width-enlarge_extent, \
				selected_boxes[i][2]*im_height+enlarge_extent, selected_boxes[i][3]*im_width+enlarge_extent)) for i in range(len(selected_boxes))])
		else:
			patch_bndbox_coordinates = np.asarray([map(int,(selected_boxes[i][0]*im_height, selected_boxes[i][1]*im_width, \
				selected_boxes[i][2]*im_height, selected_boxes[i][3]*im_width)) for i in range(len(selected_boxes))])

		# patch_idx = image_path.split('/')[-1].split('.')[0][-5:]
		patch_idx = image_slice_key[-3:]
		row_id    = np.int(patch_idx[0])
		col_id    = np.int(patch_idx[-1])
		# print 'patch_idx is:', patch_idx
	    
		if row_id == 1 & col_id == 1:
			patch_bndbox_coordinates = patch_bndbox_coordinates
		
		else:
			patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0]+(row_id-1)*im_height, patch_bndbox_coordinates[:, 1]+(col_id-1)*im_width, 
				patch_bndbox_coordinates[:, 2]+(row_id-1)*im_height, patch_bndbox_coordinates[:, 3]+(col_id-1)*im_width]).T

	else:
		pass # there is no bounding boxes found in the image slice.

	return patch_bndbox_coordinates

