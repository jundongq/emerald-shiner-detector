import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import glob

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

from utils_ import divideImg, tileImg, create_mask, coordinates_extract_transform, removeGlare

# Determin what model to load
MODEL_NAME = 'emeraldShiner_SSD_InferenceGraph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'emeraldShiner_detection.pbtxt')

NUM_CLASSES = 1


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# type(label_map): <class 'object_detection.protos.string_int_label_map_pb2.StringIntLabelMap'>

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# type(categories): list of dicts

category_index = label_map_util.create_category_index(categories)
# type(category_index): dict of dicts


def load_image_into_numpy_array(image):
	""" this function is same as  cv2.imread(image)"""
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/Volumes/Transcend/FishPassage/PTVexperiments/Analysis/Fish_1_0804_Converted/sampledFrames/Undistorted'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Undistorted_%05d.png') %i for i in range(324)]


DETECTED_FISH = 'test_sliced_imgs_{}'.format(MODEL_NAME)
SAMPLED_FRAME_DETECTED_FISH = os.path.join(PATH_TO_TEST_IMAGES_DIR, DETECTED_FISH)

if not os.path.exists(SAMPLED_FRAME_DETECTED_FISH):
	os.makedirs(SAMPLED_FRAME_DETECTED_FISH)

nb_slices = 6

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		count = 0
        
		# build a list to store the arrays that contain coordinates of all bounding boxes in all images in the group
		bndbox_coordinates_group = []

		for image_path in TEST_IMAGE_PATHS:
			# print image_path
			image = Image.open(image_path)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.

			patch_bndbox_coordinates_dict  = {}

			# obtain image name as key of coordinates_dict
			image_name = image_path.split('/')[-1]

			image_np = load_image_into_numpy_array(image)

			image_np = removeGlare.removeGlare(image_np)
			# plt.figure(figsize=(12,8))
			# plt.imshow(image_np)

			img_dict, nb_rows, nb_cols = divideImg.divideImg(image_np, nb_slices)
			# plt.figure(figsize=(12,8))
			# plt.imshow(img_dict['slice_1_1'])

			all_bndbox_coordinates_single_img = []
			for k in img_dict.keys():
				# print 'k is: ', k
				# Extract each slice of the original image at a time
				img_np_slice = img_dict[k]
				# print np.shape(img_np_slice)
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_slice_expanded = np.expand_dims(img_np_slice, axis=0)
				# print np.shape(image_np_expanded)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_slice_expanded})


				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					img_np_slice,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=2)
                
				# patch_bndbox_coordinates as an array containing all bounding boxes in an image slice
				patch_bndbox_coordinates = coordinates_extract_transform.coordinates_extract_transform(img_np_slice, k, boxes, scores, \
					score_criterion=0.1, enlarge_bndbox=True, enlarge_extent=5)

				# all_bndbox_coordinates_single_img as a list of arrays, containing all bounding boxes in all image slices
				all_bndbox_coordinates_single_img.append(patch_bndbox_coordinates)


			# after runing all image slices of a single full image, put the coordinates of all bounding boxes in one nd array.
			all_bndbox_coordinates_single_img_ = np.concatenate([all_bndbox_coordinates_single_img[i] for i in range(len(all_bndbox_coordinates_single_img))])
			patch_bndbox_coordinates_dict[image_name] = all_bndbox_coordinates_single_img_.tolist()
			bndbox_coordinates_group.append(patch_bndbox_coordinates_dict)
            
            # tile all image slices together after running object detection
			tiled_img = tileImg.tile_images(img_dict, nb_rows, nb_cols).astype(np.uint8)
			# print np.shape(tiled_img)
			count += 1
			print count
			tiled_img = cv2.cvtColor(tiled_img, cv2.COLOR_BGR2RGB)
			cv2.imwrite(os.path.join(SAMPLED_FRAME_DETECTED_FISH, "draw_%05d.png" % count), tiled_img)
			plt.figure(figsize=(12,8))
			# plt.imshow(tiled_img)


# save all the bndbox_coordinates_group into a json file
import json

with open(os.path.join(SAMPLED_FRAME_DETECTED_FISH, 'bounding_box_coordinates.json'), 'w') as f:
	json.dump(bndbox_coordinates_group, f)

print 'Done!'




