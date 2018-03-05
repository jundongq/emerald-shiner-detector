import cv2
import os
import sys
import time
import argparse
import glob
import re
import csv
import numpy as np


from animation_forPTV import forPTV_clip
from motion_detection import batch_identify_fish
from motion_detection_img import batch_identify_fish_imgs
from frameExtractor import frameExtractor
from smoothOut import smooth_fishCentroid

import seeFish_update as sf

"""
Usage example: python img_2_fishCentroid.py -d FishBehavior_11_09_5 -s 0.8 -n 2 -t 150 -w 15
"""


def imgs_2_fishCentroids(data_dir, score_threshold, nb_polygons, smooth_threshold, smooth_window, video_motion_detection=True):
	
	"""
	Input::
	dir: a directory with name of a video clip, within which images are preprocessed
	
	Output::
	fish centroids in csv form
	"""

	### Step 1: take preprocessed/forPTV images to form a video clip 'forPTV.mp4'
	forPTV_file = os.path.join(data_dir, "forPTV.mp4")
	print forPTV_file
	if not os.path.isfile(forPTV_file):
		print "Generating 'forPTV.mp4' with preprocessed images ......"
		forPTV_clip(data_dir)
		print "'forPTV.mp4' is generated!"
	else:
		print "'forPTV.mp4' exits!"

	### Step 2: take video clip 'forPTV.mp4', to do motion detection, return binary images containing moving objects
	motion_detection_dir = os.path.join(data_dir, 'motion_detection_MOG')
	if not os.path.exists(motion_detection_dir):
		os.makedirs(motion_detection_dir)
		if video_motion_detection:
			batch_identify_fish(data_dir, motion_detection_dir)
			# extracting each frame out of motion detection video
			frameExtractor(os.path.join(motion_detection_dir, 'fgbg_MOG_results.avi'), 1)	
		else:
			print "Detecting moving objects from a sequence of forPTV images ......"
			batch_identify_fish_imgs(data_dir)
	
	else:
		if not os.path.isfile(os.path.join(motion_detection_dir, 'fgbg_MOG_results.avi')):
			if video_motion_detection:
				print "Detecting moving objects based on 'forPTV.mp4' ......"
				batch_identify_fish(data_dir, motion_detection_dir)
				# extracting each frame out of motion detection video
				frameExtractor(os.path.join(motion_detection_dir, 'fgbg_MOG_results.avi'), 1)	
			else:
				print "Detecting moving objects from a sequence of forPTV images ......"
				batch_identify_fish_imgs(data_dir)
				print "Motion detection completes!"
		else:
			if not os.path.exists(os.path.join(motion_detection_dir, 'fgbg_MOG_results/sampledFrames')):
				# extracting each frame out of motion detection video
				frameExtractor(os.path.join(motion_detection_dir, 'fgbg_MOG_results.avi'), 1)
				print "Motion detection completes!"	
			else:
				print "Motion detection completes!"
	
	### Step 3: take binary images and bounding boxes as input, return fishCentroids.csv
	combined_dir = os.path.join(data_dir, "combined_imgs")
	if not os.path.exists(combined_dir):
		print "Identifying fish centroids ......"
		sf.run(data_dir, score_threshold, nb_polygons,video_motion_detection)
	else:
		# if the subdir exists, check if it is empty, if so, it still needs running on 'seeFish'
		if os.listdir(combined_dir) == []:
			sf.run(data_dir, score_threshold, nb_polygons, video_motion_detection)
		
	### Step 4: take 'fishCentroids.csv' as input, smooth out outliers.
	print "Smoothing outliers of fish centroids ......"	
	smooth_fishCentroid(data_dir, score_threshold, smooth_threshold, smooth_window)
	
	print "Fish centroids are retrieved and smoothed!"
	
if __name__ == '__main__':
	
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--data_dir', required=True, help='dirctory that contains fishCentroids.csv')
	ap.add_argument('-s', '--score_threshold', type=float, required=True, help='threshold to draw interrogation masks')
	ap.add_argument('-n', '--number_of_blocks', type=int, required=True, help='an integer indicating number of blocks in the video')
	ap.add_argument('-t', '--smooth_threshold', type=int, required=True, help='threshold to smooth out outliers')
	ap.add_argument('-w', '--window', type=int, required=True, help='window size to compute rolling median')
	
	args = vars(ap.parse_args())

	data_dir         = args['data_dir']
	score_threshold  = args['score_threshold']
	smooth_threshold = args['smooth_threshold']
	smooth_window    = args['window']
	nb_polygons      = args['number_of_blocks']
	
	imgs_2_fishCentroids(data_dir, score_threshold, nb_polygons, smooth_threshold, smooth_window, video_motion_detection=True)
