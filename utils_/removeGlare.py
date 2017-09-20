import cv2
import numpy as np


def removeGlare(img, satThreshold = 180, glareThreshold=240):
	"""This function is used to remove light reflection on water surface during experiments,
	it return a np.ndarray() in shape of (height, width, channel).
	credit: http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
	input: image 
	output: clean image"""

	image     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	h, s, v   = cv2.split(hsv_image)

	# return 1/0, 1 indicates satuation <180, 0 indicates satuation >180, which do not need to be inpainted.
	# so nonSat serves as a mask, find all pixels that are not very saturated
	# all non saturated pixels need to be set as 1 (non-zero), which will be inpainted
	nonSat = s < satThreshold

	# Slightly decrease the area of the non-satuared pixels by a erosion operation.
	# return a kernel with designated shape, using erosion to minorly change the mask
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	nonSat = cv2.erode(nonSat.astype(np.uint8), kernel)

	# Set all brightness values, where the pixels are still saturated to 0.
	# Only non-zero pixels in the mask need to be inpainted. So the saturated pixels set as 0 (no need to inpaint)
	v2 = v.copy()

	# set all saturated pixels, which do not need to be inpainted as 0
	v2[nonSat == 0] = 0

	#  glare as a mask filter out very bright pixels.
	glare = v2 > glareThreshold
	# Slightly increase the area for each pixel
	glare = cv2.dilate(glare.astype(np.uint8), kernel)
	glare = cv2.dilate(glare.astype(np.uint8), kernel)

	clean_image = cv2.inpaint(image, glare, 3, cv2.INPAINT_TELEA)
	clean_image =  cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
	return clean_image