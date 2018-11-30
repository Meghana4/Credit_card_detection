import numpy as np 
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from os.path import *
from glob import glob

def get_args():
    parser = argparse.ArgumentParser(description="This script detects credit card contour, ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, default=" ",
                        help="test image")
    args = parser.parse_args()
    return args

def main():
	args = get_args()

	img = cv2.imread(args.image)
	colour_img = img.copy()
	img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	H,W = img.shape
	
	#extract_credit_card
	median = cv2.medianBlur(img,21)
	ret, mask = cv2.threshold(median,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	_,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

	largest_contour = cnts[0]
	screenCnt = None
	# # loop over our contours
	for c in cnts:
		peri = cv2.arcLength(c, True)
		# x,y,w,h = cv2.boundingRect(c)
		approx = cv2.approxPolyDP(c, 0.02* peri, True)
		if len(approx) == 4:
			screenCnt = c
			break
	if (screenCnt is None):
		rect = cv2.minAreaRect(largest_contour)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		out = cv2.drawContours(colour_img.copy(),[box],-1,(0,255,0),3)
	elif (cv2.arcLength(largest_contour,True) > cv2.arcLength(screenCnt,True)):
		rect = cv2.minAreaRect(largest_contour)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		out = cv2.drawContours(colour_img.copy(),[box],-1,(0,255,0),3)
	else:
		approx = cv2.approxPolyDP(screenCnt, 0.02* peri, True)
		out = cv2.drawContours(colour_img.copy(), [approx], -1, (0, 255, 0), 3)
	
	cv2.imshow("output",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	filename = directory  +  ((file).split("/")[8]).split(".jpg")[0] + "_output.png"
	cv2.imwrite(filename,out)


if __name__ == '__main__':
	main()