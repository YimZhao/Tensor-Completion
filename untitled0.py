# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 19:33:24 2022

@author: Administrator
"""
import cv2
import tensorly as tl
import numpy as np
from numba import jit
from line_profiler import LineProfiler

def ReplaceInd(X, known, Image):

    imSize = Image.shape

    for i in range(len(known)):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1)
        in2 = int(imSize[0] - known[i] % imSize[1] - 1)
        X[in1, in2, :] = Image[in1, in2, :]
    return X

KownPercentage = 0.5
Image = cv2.imread("seaside.jpg")
# cv2.namedWindow('Corrupting Image', cv2.WINDOW_NORMAL)
# cv2.imshow("Corrupting Image", Image.astype(np.uint8))
# cv2.waitKey(0)
imSize = Image.shape
known = np.arange(np.prod(imSize) / imSize[2])
np.random.shuffle(known)
known = known[:int(KownPercentage * (np.prod(imSize) / imSize[2]))]
print(known.shape)
# Corrupting Image
X = np.zeros(imSize)
X = ReplaceInd(X, known, Image)
#cv2.namedWindow('Corrupting Image', cv2.WINDOW_NORMAL)
#cv2.imshow("Corrupting Image", X.astype(np.uint8))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
a = abs(np.random.rand(3, 1))
a = a / np.sum(a)
p = 1e-6
K = 50
ArrSize = np.array(imSize)
ArrSize = np.append(ArrSize, 3)
Mi = np.zeros(ArrSize)
Yi = np.zeros(ArrSize)