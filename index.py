# import the necessary packages
#from project.colordescript 
#import ColorDescriptor
import argparse
import glob
import cv2
import numpy as np
import imutils
class ColorDescriptor:
	def __init__(self, bins):
	
		self.bins = bins
	def describe(self, image):
		
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		seg = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]

		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        
		# loop over the segments
		for (startX, endX, startY, endY) in seg:
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			hist = self.histogram(image, cornerMask)
			features.extend(hist)

		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features

	def histogram(self, image, mask):
	
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		else:
			hist = cv2.normalize(hist, hist).flatten()
		return hist
		
# construct the argument parser and parse the arguments
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
argp.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(argp.parse_args())
# initialize the color descriptor
cdvar = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open(args["index"], "w")
# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	# describe the image
	features = cdvar.describe(image)
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
# close the index file
output.close()
