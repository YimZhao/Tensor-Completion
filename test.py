import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('./output/rec.jpg')
h = np.zeros((256,256,3))
bins = np.arange(256).reshape(256,1)
color = [(255,0,0),(0,255,0),(0,0,255)]
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    cv.normalize(histr,histr,0,255*.9,cv.NORM_MINMAX)
    hist = np.int32(np.around(histr))
    pts = np.column_stack((bins,hist))
    cv.polylines(h,[pts],False,col)
h = np.flipud(h)
cv.imshow("colorhist",h)
cv.waitKey(0)
cv.destryAllWindows()