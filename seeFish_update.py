import cv2
import csv
import json
import glob
import os
import argparse
import numpy as np

'''
Modified on Jan. 31 2018. 

The change involves modifying function 'create_mask' to 'create_inquiry_mask'. Using score threshold
to filter out bounding boxes of low possibility, so it is less noisy for identifying fish.
'''


def find_fish(img, cnts, remove_noise=True):
	"""Input as contours, this function uses aspect ratio of rotated bounding box
	to identify fish. Remove noise by filling all other contours """
	
	aspectRatios = []
	rotatedBoxes = []
	# centroids    = []

	# find bounding rectangle for each contour
	for cnt in cnts:

		# define rotated rectangle
		# rect containes (center(x,y), (width, height), angle of rotation)
		rect = cv2.minAreaRect(cnt)

			# box contains fours vertices of a rotated rect, useful to draw the rotated rectange
		box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
		box = np.int0(box)

		rotatedBoxes.append(box)

		# convert np.ndarray box as a list
		box_list = box.tolist()

		# check if there is any repeated coordiantes in the box. If repeated, its a dot, not fish.
		if len(box_list) == len(set([tuple(element_pair) for element_pair in box_list])): 

			# rotatedBoxes.append(box)

			# compute distance between one vertice and others
			dist = []

			for i in range(1,4):
				d = np.sqrt((box_list[0][0]-box_list[i][0])**2 + (box_list[0][1]-box_list[i][1])**2)
				dist.append(d)

			# The shorted dist is width, the sencond longest is length, the longest is diagonal
			# Sort the list dist in ascending order
			w = sorted(dist)[0]
			h = sorted(dist)[1]

			# if only there are two more pixels for a blob, it can be considered as candidates of a fish.
			if w > 2 and h >2:
				aspectRatio = h/w
				aspectRatios.append(aspectRatio)
			else: 
				aspectRatios.append(-2)
		# to make sure the number of apsect ratios corresponds to number of contours
		else:
			aspectRatios.append(-1)

	print 'Total number of aspectRatios:', len(aspectRatios)

	fish_index = np.argmax(aspectRatios)

	fish_box = rotatedBoxes[fish_index]
	print 'Fish Aspect Ratio:', aspectRatios[fish_index]
	
	global cx, cy
	cx = 0
	cy = 0
	if aspectRatios:
		fish_index = np.argmax(aspectRatios)
		fish_box   = rotatedBoxes[fish_index]
		# print fish_box
		M  = cv2.moments(cnts[fish_index])
		if M['m00'] != 0:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
		else:
			pass

	return cx, cy


def create_inquiry_mask(image, bounding_boxes, scores, score_threshold):
    """
    input::
    
    image: as a numpy ndarray, in format (height, widht, channel)
    bounding_boxes: an ndarray with each row containing a list of corners of a bounding box
    scores: a list with each element indicating the probability that corresponding bounding box containing fish
    score_threshold: a value set to filter our bounding boxes of low probability
    
    
    output::
    
    mask_image: as a binary mask with elements inside of bounding boxes equal to 1, others are 0.
    """
    
    mask_image = np.zeros(np.shape(image)[:2])
    
    # the input bounding boxes are the ones with scores higher than 0.15 while doing inference
    selected_idx = [s>score_threshold for s in scores]
    bounding_boxes_selected = [box for (box, idx) in zip(bounding_boxes, selected_idx) if idx]
    

    # if there is no bounding_boxes found in the image, let the motion detection take care of fish detection
    bounding_boxes_coor_list = [x for sublist in bounding_boxes_selected for x in sublist]
    if sum(bounding_boxes_coor_list) == 0:
        mask_image = np.ones(np.shape(image)[:2])

    else:
        for i in range(len(bounding_boxes_selected)):
            bounding_boxes_selected[i] = [np.int(x) for x in bounding_boxes_selected[i]]
            if bounding_boxes_selected[i] == [0, 0, 0, 0]:
                mask_image = mask_image
            else:
                # keep the bounding box enlargements
                mask_image[bounding_boxes_selected[i][0]:bounding_boxes_selected[i][2], bounding_boxes_selected[i][1]:bounding_boxes_selected[i][3]] = 1

                # remove the enlargements of bounding boxes
                enlargment = 0 # was set in 'object_detection_output_bndbox_coordinates.py' on 12/13/2017
                mask_image[bounding_boxes_selected[i][0]+enlargment:bounding_boxes_selected[i][2]-enlargment, bounding_boxes_selected[i][1]+enlargment:bounding_boxes_selected[i][3]-enlargment] = 1
    return mask_image


def blocks_mask(block_vortices, nb_polygons):
	
	polys_ = block_vortices.copy()
	for i in range(nb_polygons):
		for j in range(4):
			polys_[i][j][1] = 1070-block_vortices[i][j][1]

	mask = np.ones((1070, 1914), dtype=np.int8)
	for i in polys_.tolist():
		cv2.fillConvexPoly(mask, np.array(i), 0)
	return mask


def run(dir, score_threshold, nb_polygons, video_motion_detection=True):
	"""
	Input::
	
	dir:
	a directory that contains all related images. For example 'FishBehavior_11_09_9' is 
	a directory containing all imgs of 'FishBehavior_11_09_9.mp4'.
	
	score_threshold:
	the score criterion that is used to filtered out returned fish bounding boxes
	
	Output::
	a csv file containing fish mass center in pixel unit. 'fishCentroids_{}.csv'
	"""

	PWD = os.getcwd()
	DATA_DIR = os.path.join(PWD, dir)
	
	if video_motion_detection:
		MOTION_DETECTION = 'motion_detection_MOG/fgbg_MOG_results/sampledFrames'
	else:
		MOTION_DETECTION = 'motion_detection_MOG'
	
	FISH_DETECTION = 'UndistortedPreprocessed/fishDetection_emeraldShiner_inference_graph_faster_rcnn_resnet101_coco_12_13_2017'

	COMBINED_IMG_DIR = os.path.join(DATA_DIR, 'combined_imgs')
	if not os.path.isdir(COMBINED_IMG_DIR):
		os.mkdir(COMBINED_IMG_DIR)

	# the path to all imgs contained all moving objects in each frame (results from motion detection)
	IMAGES_PATH = glob.glob(os.path.join(DATA_DIR, MOTION_DETECTION) + '/*.png')
	print "There are %d images" %(len(IMAGES_PATH))
	#### load coordinates of bounding boxes, and corresponding scores
	with open(os.path.join(DATA_DIR, FISH_DETECTION, 'bounding_box_coordinates.json'), 'r') as f:
		box_coordinates = json.load(f)

	with open(os.path.join(DATA_DIR, FISH_DETECTION, 'bounding_box_scores.json'), 'r') as f:
		box_scores = json.load(f)

	# Block ploygons in the corresponding videos
	nb_polygons = nb_polygons
	polys = np.loadtxt(os.path.join(dir, 'polygons.txt'))
	polys = polys.reshape((nb_polygons, 4, 2)).astype(np.int)

	centroids = []
	c_x, c_y = 0, 0 
	for i, img_path in enumerate(IMAGES_PATH):
	
		if i > len(IMAGES_PATH)-1:
			print 'Done!'
			break

		else:

			print i
			im = cv2.imread(img_path)
			img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

			#### removing noise, optional
			kernel_1 = np.ones((2,2), np.uint8)
			erosion  = cv2.erode(img, kernel_1)
			kernel_2 = np.ones((2,2), np.uint8)
			dilation = cv2.dilate(erosion, kernel_1)
		
			# binarize the images for input of create_mask
			ret,thresh = cv2.threshold(dilation,200,255,2)
			bounding_boxes = box_coordinates[i].values()[0]
			bounding_boxes_scores = box_scores[i].values()[0]
			mask_image = create_inquiry_mask(thresh, bounding_boxes, bounding_boxes_scores, score_threshold)
			bloc_mask = blocks_mask(polys, nb_polygons)
			# apply mask_image on the binary image, to remove all noise outside of the bounding_boxes
			conbined_img = np.multiply(thresh,mask_image).astype(np.uint8)
			conbined_img = np.multiply(conbined_img,bloc_mask).astype(np.uint8)
			# print np.shape(conbined_img)
			# plt.figure(figsize=(12,8))
			# plt.imshow(thresh)
			# plt.figure(figsize=(12,8))
			# plt.imshow(conbined_img)
			conbined_img_save = cv2.cvtColor(conbined_img, cv2.COLOR_GRAY2RGB).astype(np.uint8)
			cv2.imwrite(os.path.join(COMBINED_IMG_DIR, "conbined_img_%05d.png" % (i+1)), conbined_img_save)

			# find contours on the conbined images
			cnts, hierarchy = cv2.findContours(conbined_img,1,2)
			if len(cnts) != 0:
				c_x, c_y = find_fish(conbined_img, cnts, remove_noise=True)
			centroids.append([c_x, c_y])

			with open(os.path.join(DATA_DIR, 'fishCentroids_{}.csv'.format(score_threshold)), 'wb') as f:
				writer = csv.writer(f, delimiter=',')
				writer.writerow(['cx', 'cy'])
				for line in centroids:
					if None not in line:
						writer.writerow(line)

if __name__ == "__main__":
	#### import all images
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dir', required=True, help='dirctory that contains UndistortedPreprocessed')
	ap.add_argument('-s', '--score', type=float, required=True, help='a float point indicating score threshold of bounding boxes')
	ap.add_argument('-n', '--number_of_blocks', type=int, required=True, help='an integer indicating number of blocks in the video')
	args = vars(ap.parse_args())

	img_dir = args['dir']
	score_threshold = args['score']
	nb_polygons = args['number_of_blocks']
	
	run(img_dir, score_threshold, nb_polygons, video_motion_detection=True)