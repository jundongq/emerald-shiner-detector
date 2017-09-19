def coordinates_extract_transform(image_slice, image_slice_key, boxes, scores, score_criterion=0.1, enlarge_bndbox=True, enlarge_extent=5):
	"""
	input::
	as sliced image path, and bounding boxes coordinates (0 ~ 1)
	image_path: the directory of each sliced image (patch) of an orignal large image
	boxes: a tuple containing 100 relative coordinates in each image, in form of (xmin, ymin, xmax, ymax)
	scores: iou value

	output::
	as a numpy asrray, the bounding boxes coordinates in the orignal image
	"""

	im_height, im_width = np.shape(image_slice)[:2]

	# select boxes that satisfy the score_criterion
	selected_boxes = boxes[0][scores[0] > score_criterion]

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
	if patch_idx == '1_1':
		patch_bndbox_coordinates = patch_bndbox_coordinates

	elif patch_idx == '1_2':
		patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0], patch_bndbox_coordinates[:, 1]+im_width, 
			patch_bndbox_coordinates[:, 2], patch_bndbox_coordinates[:, 3]+im_width]).T

	elif patch_idx == '1_3':
		patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0], patch_bndbox_coordinates[:, 1]+2*im_width, 
			patch_bndbox_coordinates[:, 2], patch_bndbox_coordinates[:, 3]+2*im_width]).T

	elif patch_idx == '2_1':
		patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0]+im_height, patch_bndbox_coordinates[:, 1], 
			patch_bndbox_coordinates[:, 2]+im_height, patch_bndbox_coordinates[:, 3]]).T

	elif patch_idx == '2_2':
		patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0]+im_height, patch_bndbox_coordinates[:, 1]+im_width, 
			patch_bndbox_coordinates[:, 2]+im_height, patch_bndbox_coordinates[:, 3]+im_width]).T

	elif patch_idx == '2_3':
		patch_bndbox_coordinates = np.asarray([patch_bndbox_coordinates[:, 0]+im_height, patch_bndbox_coordinates[:, 1]+2*im_width, 
			patch_bndbox_coordinates[:, 2]+im_height, patch_bndbox_coordinates[:, 3]+2*im_width]).T


	return patch_bndbox_coordinates