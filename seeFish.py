import cv2
import csv
import json
import glob
import os
import argparse
import numpy as np



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

	'''
	print 'Fish Minibox:', fish_box
	cv2.drawContours(img, [fish_box], 0, (0,0,255), 2)
	# cv2.drawContours(thresh_3D, cnts, 0, (0,0,255), 2)
	cv2.imwrite( "Fish_1_0804_Converted/sampledFrames_MOG/fish_%05d.png" % 1, img)
	with open('Fish_1_0804_Converted/sampledFrames_MOG/fishCentroids.csv', 'wb') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['cx', 'cy'])
		for line in centroids:
			if None not in line:
				writer.writerow(line)

	# cv2.imshow("Show",img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	'''

def create_mask(image, bounding_boxes):
	"""
	input::
	image: as a numpy ndarray, in format (height, widht, channel)
	bounding_boxes: an ndarray with each row containing a list of corners of a bounding box
	output::
	mask_image: as a binary mask with elements inside of bounding boxes equal to 1, others are 0.
	"""
	mask_image = np.zeros(np.shape(image)[:2])

	# if there is no bounding_boxes found in the image, let the motion detection take care of fish detection
	bounding_boxes_coor_list = [x for sublist in bounding_boxes for x in sublist]
	if sum(bounding_boxes_coor_list) == 0:
		mask_image = np.ones(np.shape(image)[:2])

	else:
		for i in range(len(bounding_boxes)):
			bounding_boxes[i] = [np.int(x) for x in bounding_boxes[i]]
			if bounding_boxes[i] == [0, 0, 0, 0]:
				mask_image = mask_image
			else:
				# keep the bounding box enlargements
				mask_image[bounding_boxes[i][0]:bounding_boxes[i][2], bounding_boxes[i][1]:bounding_boxes[i][3]] = 1
				
				# remove the enlargements of bounding boxes
				enlargment = 0 # was set in 'object_detection_output_bndbox_coordinates.py' on 12/13/2017
				mask_image[bounding_boxes[i][0]+enlargment:bounding_boxes[i][2]-enlargment, bounding_boxes[i][1]+enlargment:bounding_boxes[i][3]-enlargment] = 1
	return mask_image

# Block ploygons in this 11_20_20 videos
polys = np.array([[[700, 346], [416, 347], [215, 210], [893, 208]],
                  [[1294, 189],[1065, 65], [1795, 60], [1565, 187]],
                  [[0, 107], [37, 91], [0, 66],[0, 66]]], np.int32)

def blocks_mask(block_vortices):
	
	polys_ = block_vortices.copy()
	for i in range(3):
		for j in range(4):
			polys_[i][j][1] = 1070-polys[i][j][1]

	mask = np.ones((1070, 1914), dtype=np.int8)
	for i in polys_.tolist():
		cv2.fillConvexPoly(mask, np.array(i), 0)
	return mask


#### import all images
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', required=True, help='dirctory that contains UndistortedPreprocessed')
args = vars(ap.parse_args())

img_dir = args['dir']

PWD = os.getcwd()
DATA_DIR = os.path.join(PWD, img_dir)
MOTION_DETECTION = 'motion_detection_MOG/fgbg_MOG_results/sampledFrames'
FISH_DETECTION = 'UndistortedPreprocessed/fishDetection_emeraldShiner_inference_graph_faster_rcnn_resnet101_coco_12_13_2017'

COMBINED_IMG_DIR = os.path.join(DATA_DIR, 'combined_imgs')
if not os.path.isdir(COMBINED_IMG_DIR):
	os.mkdir(COMBINED_IMG_DIR)

# the path to all imgs contained all moving objects in each frame (results from motion detection)
IMAGES_PATH = glob.glob(os.path.join(DATA_DIR, MOTION_DETECTION) + '/*.png')
print "There are %d images" %(len(IMAGES_PATH))
#### load coordinates of bounding boxes
with open(os.path.join(DATA_DIR, FISH_DETECTION, 'bounding_box_coordinates.json'), 'r') as f:
    box_coordinates = json.load(f)

centroids = []
for i, img_path in enumerate(IMAGES_PATH):
	
	if i > len(IMAGES_PATH)-1:
		print 'Done!'
		break

	else:

		print i
		im = cv2.imread(img_path)
		img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		#### removing noise, optional
		kernel_1 = np.ones((1,1), np.uint8)
		erosion  = cv2.erode(img, kernel_1)
		kernel_2 = np.ones((1,1), np.uint8)
		dilation = cv2.dilate(erosion, kernel_1)
		
		# binarize the images for input of create_mask
		ret,thresh = cv2.threshold(dilation,200,255,2)
		bounding_boxes = box_coordinates[i].values()[0]
		mask_image = create_mask(thresh, bounding_boxes)
		bloc_mask = blocks_mask(polys)
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

		with open(os.path.join(DATA_DIR, 'fishCentroids.csv'), 'wb') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow(['cx', 'cy'])
			for line in centroids:
				if None not in line:
					writer.writerow(line)
