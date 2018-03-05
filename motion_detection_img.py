"""
Take in images, instead of videos to conduct motion detection. 02/27/2018
The results do not look good.
"""

# import the necessary packages
import numpy as np
import cv2
import sys
import os
import csv
import time
import argparse
import glob


def batch_identify_fish_imgs(videofile):
	"""Input as video, return processed video with identified fish"""

	fps    = 20

	# creating background subtraction object
	fgbg   = cv2.BackgroundSubtractorMOG() # It works for opencv 2.4

	data_dir = os.path.join(videofile.split('/')[0], 'motion_detection_MOG')
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	print data_dir
	
	#### starting the loop
	frames  = glob.glob(os.path.join(videofile, 'UndistortedPreprocessed/forPTV', 'forPTV*.png'))
	
	for i, img in enumerate(frames):
		
		print i
		
		for j in range(i, i+16):
   			bgImageFile = frames[j]
    		print "Opening background", bgImageFile
    		bg = cv2.imread(bgImageFile)
    		fgbg.apply(bg, learningRate=0.5)
		frame = cv2.imread(img)


		#### bluring before binarizing can help remove small noise
		# blurred  = cv2.bilateralFilter(clean_frame, 5, 15,15)
		# blurred_1  = cv2.GaussianBlur(blurred, (3, 3), 0)
		
		# return <type 'numpy.ndarray'>, 1-channel pics
		# fgmask   = fgbg.apply(frame,  learningRate=0.01)
		fgmask   = fgbg.apply(frame)
		# print np.shape(fgmask)
		fgmask_3d = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
		cv2.imwrite(os.path.join(videofile, 'motion_detection_MOG', 'motion_detecion_{}.png'.format(i)),fgmask_3d)

	print "Done!"


###### Run ######

if __name__ == "__main__":

	ag = argparse.ArgumentParser()
	ag.add_argument('-d', '--dir', required=True, help='a directory with name of a video clip, within which images are preprocessed ')
	
	args = vars(ag.parse_args())
	print args
	videofile = args['dir']
	

	t0 = time.time()
	batch_identify_fish_imgs(videofile)
	t1 = time.time()

	t = t1 - t0
	print "The process took %.2f mins." %(round(t/60., 2))

