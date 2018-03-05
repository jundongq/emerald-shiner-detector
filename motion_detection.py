# It is a pipeline to preprocess the images of fish moving
# Updated, 02/28, 2018

# import the necessary packages
import numpy as np
import cv2
import sys
import os
import csv
import time
import argparse


def batch_identify_fish(dir, motion_detection_dir):
	"""Input as video, return processed video with identified fish"""
	
	videofile = os.path.join(dir, 'forPTV.mp4')

	camera = cv2.VideoCapture(videofile)
	
	if camera.isOpened():
		pass
	else: 
		print "Converting MP4 file..."
		os.system('ffmpeg -i %s -c:v copy %s_Converted.mp4' %(videofile, videofile[:-4]))
		camera = cv2.VideoCapture('%s_Converted.MP4' % videofile[:-4])	

	fps    = camera.get(cv2.cv.CV_CAP_PROP_FPS)
	history= np.int(fps/2)

	# creating background subtraction object
	fgbg   = cv2.BackgroundSubtractorMOG() # It works for opencv 2.4

	# creating video writing object, outside of the loop
	size = (int(camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
			int(camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))*2)
	fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')     #works, large

	# output = cv2.VideoWriter(os.path.join(data_dir, 'rawVideo_identifiedFish.avi'),  fourcc, fps, size, True)

	# size for individual video
	size_1 = (int(camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
			int(camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
	# out_fish_only = cv2.VideoWriter(os.path.join(data_dir,'fishOnly.avi'), fourcc, fps, size_1, True)

	# size for individual video, background subtraction results
	fgbg_results = cv2.VideoWriter(os.path.join(motion_detection_dir,'fgbg_MOG_results.avi'), fourcc, fps, size_1, True)

	print "~~~ identifying fish from raw video..."

	#### starting the loop
	frame_nb  = 0

	while True:
		
		print frame_nb

		grabbed, frame = camera.read()

		if not grabbed:
			break

		frame_nb += 1

		#### bluring before binarizing can help remove small noise
		# blurred  = cv2.bilateralFilter(clean_frame, 5, 15,15)
		# blurred_1  = cv2.GaussianBlur(blurred, (3, 3), 0)
		
		# return <type 'numpy.ndarray'>, 1-channel pics
		# fgmask   = fgbg.apply(frame,  learningRate=0.01)
		fgmask   = fgbg.apply(frame)
		# print np.shape(fgmask)
		fgmask_3d = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
		# print np.shape(fgmask_3d)
		
		'''
		#### removing noise, optional
		kernel   = np.ones((3,3), np.uint8)
		erosion  = cv2.erode(fgmask, kernel)
		dilation = cv2.dilate(erosion, kernel)


		ret, thresh = cv2.threshold(dilation, 200, 255, 0)


		contours, hierarchy = cv2.findContours(thresh,1,2)

		c_x, c_y = find_fish(dilation, contours, remove_noise=True)
		centroids.append([c_x, c_y])
		# centroids_deque.appendleft((c_x, c_y))


		# print "the chanel of fgmask is: " + str(np.shape(fgmask)[-1])
		#### converting 1 channel binary images to 3 channel images, for stacking them together
		identified_fish   = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)

		# print np.shape(fgmask)
		cv2.imshow('frame', dilation)
		

		#### using cv2.inpaint to blend the fish into the background
		# blended = cv2.inpaint(frame, dilation, 3, cv2.INPAINT_TELEA)
		# print "the chanel of blended is: " + str(np.shape(blended)[-1])
	
		#### showing
		twoV  = np.vstack((frame, identified_fish))

		#### adding name on different frames, Bottom-left corner of the text string in the image
		font  = cv2.FONT_HERSHEY_SIMPLEX
		width = np.shape(frame)[0]
		hight = np.shape(frame)[1]
		cv2.putText(twoV, 'Original Video',               (int(width/2-80),int(hight/15)), font, 0.7,(0,0,255),2)
		cv2.putText(twoV, 'Identified Fish',              (int(width/2-80),int(hight/9)+int(hight/2)), font, 0.7,(0,0,255),2)
		'''
		# cv2.putText(threeV, 'Fish blended with Background', (int(width/2-80),int(hight/6)+int(hight)), font, 0.7,(0,0,255),2)
		# cv2.putText(blended, 'Fish blended with Background',(int(width/2-80),int(hight/15)), font, 0.7,(0,0,255),2)
		# cv2.putText(blended, '(Preprocessing for Streams)', (int(width/2-80),int(hight/15)+20), font, 0.7,(0,0,255),1)
		
		#### saving videos to the video recorder

# 		output.write(twoV)
# 		out_fish_only.write(identified_fish)
		fgbg_results.write(fgmask_3d)
# 		cv2.imshow('Feedin-Tracking', twoV)
# 		cv2.imshow('Identified Fish)', identified_fish)
# 		cv2.imshow('BgFg', fgmask_3d)

	#### release the video capture and video output after work.
	camera.release()
# 	output.release()
# 	out_fish_only.release()
	fgbg_results.release()
	cv2.destroyAllWindows()

	print "Done!"


###### Run ######

if __name__ == "__main__":

	ag = argparse.ArgumentParser()
	ag.add_argument('-d', '--dir', required=True, help='a directory containing forPTV.mp4')
	args = vars(ag.parse_args())
	dir = args['dir']
	motion_detection_dir = os.path.join(dir, 'motion_detection_MOG')
	if os.path.exists(motion_detection_dir):
		print 'Motion detection on forPTV.mp4 has been completed!'
	else:
		os.makedirs(motion_detection_dir)
		batch_identify_fish(dir, motion_detection_dir)



