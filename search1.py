# import the necessary packages
#import ColorDescriptor
#import searcher
import argparse
import csv
import cv2
import numpy as np
import imutils
# construct the argument parser and parse the arguments
class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			my_reader = csv.reader(f)
			# loop over the rows in the index
			for row in my_reader:
				
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
				
				results[row[0]] = d
			# close the reader
			f.close()
		
		results = sorted([(v, k) for (k, v) in results.items()])
		
		return results[:limit]

	def chi2_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d
    
class ColorDescriptor:
  def __init__(self, bins):
	
   self.bins = bins
  def describe(self, image):
   image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   vectorized = image.reshape((-1,3))
   vectorized = np.float32(vectorized)
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)     
   K=5
   attempts=10
   ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
   center = np.uint8(center)
   res = center[label.flatten()]
   result_image = res.reshape((image.shape))
   features = []
   (h, w) = image.shape[:2]

   hist = self.histogram(result_image, 0)
   features.extend(hist)
		# return the feature vector
   return features

  def histogram(self, image, mask):
   hist = cv2.calcHist([image], [0, 1, 2], None, self.bins,		
                       [0, 180, 0, 256, 0, 256])
   if imutils.is_cv2():
    hist = cv2.normalize(hist).flatten()
   else:
    hist = cv2.normalize(hist, hist).flatten()
   return hist
		

argp = argparse.ArgumentParser()
argp.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
argp.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
argp.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(argp.parse_args())
# initialize the image descriptor
cdvar = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cdvar.describe(query)
# perform the search
searcher = Searcher(args["index"])
results = searcher.search(features)
# display the query
cv2.imshow("Query", query)
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(args["result_path"] + "/" + resultID)
	cv2.imshow("Result", result)
	cv2.waitKey(0)
