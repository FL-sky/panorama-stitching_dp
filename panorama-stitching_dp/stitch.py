
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2

import os
print(os.getcwd())

str1 = "./images/sample1-1.jpg"
str2 = "./images/sample2-1.jpg"

imageA = cv2.imread(str1)
imageB = cv2.imread(str2)

# imageA = cv2.imread("./images/3-left.JPG")
# imageB = cv2.imread("./images/3-right.JPG")




# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.imwrite('saved_image.jpg', result)

cv2.waitKey(0)
print("end")

### https://blog.csdn.net/qq_63510237/article/details/138043910