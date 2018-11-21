import numpy as np 
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
	# ret, mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	mask = cv2.Canny(img, 30, 200)

	_,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None
	# # loop over our contours
	for c in cnts:
		peri = cv2.arcLength(c, True)
		# x,y,w,h = cv2.boundingRect(c)
		approx = cv2.approxPolyDP(c, 0.02* peri, True)
		if len(approx) == 4:
			screenCnt = c
			break

	out = cv2.drawContours(colour_img.copy(), [screenCnt], -1, (0, 255, 0), 3)
	final = np.concatenate((colour_img,out),axis=1)
	cv2.imshow("output",final)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	filename = (args.image).split(".")[0] + "_output.png"
	cv2.imwrite(filename,final)


if __name__ == '__main__':
	main()