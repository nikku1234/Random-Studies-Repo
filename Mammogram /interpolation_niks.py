import cv2
import numpy as np

img = cv2.imread('test.png')

near_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
bilinear_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
bicubic_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)

cv2.imwrite('Nearest Neighbour.png', near_img)
cv2.imwrite('Bilinear.png', bilinear_img)
cv2.imwrite('Bicubic.png', bicubic_img)