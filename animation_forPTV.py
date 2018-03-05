import cv2
import glob
import os
import argparse
'''
# building an video object
vvw  =   cv2.VideoWriter('emeraldShiner_detection_ssd_inception_tiled_imgs.mp4',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),20,(1920,1080))

for i, img in enumerate(glob.glob('sampledFrames/Undistorted/splited_pics_detected_fish_ssd_inception_v2_0910_inference_graph/Tiled_Imgs/*.png')):
	cv_img = cv2.imread(img)
	print i
	# writing numpy array into the video
	vvw.write(cv_img)
'''


def forPTV_clip(DIR):
	# building an video object
	vvw = cv2.VideoWriter(os.path.join(DIR, 'forPTV.mp4'),cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),20,(1914,1070))
	img_handles = glob.glob(os.path.join(DIR,'UndistortedPreprocessed/forPTV/forPTV_*.png'))

	# vvw = cv2.VideoWriter('vel_fish.mp4',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),20,(2000,1200))
	# img_handles = glob.glob('vel_fish_imgs_smooth/*.png')

	print len(img_handles)

	# sorting imgs
	def getint(name):
		basename = name.partition('.')
		return int(float(basename[0].split('_')[-1]))

	img_handles.sort(key=getint)

	for i, img in enumerate(img_handles):
		cv_img = cv2.imread(img)
		print i
	
		# writing numpy array into the video
		vvw.write(cv_img)

if __name__ == "__main__":
	
	print 'Generating animation using forPTV images...'
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dir', required=True, help='a directory containing preprocessed images for a video clip')
	
	args = vars(ap.parse_args())
	
	DIR = args['dir']
	forPTV_clip(DIR)
	