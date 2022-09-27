import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('seaside1.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[255],[0,255])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()