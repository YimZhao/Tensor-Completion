import cv2
import numpy as np
ori = cv2.imread("./jpg/seaside1.jpg")
interval2x = cv2.imread("./jpg/interval2x.png")
def recoverScore(X,img):
    imgSize = img.shape
    same = 0
    for i in range(imgSize[0]):
        for j in range(imgSize[1]):
            if (abs(X[i,j,:] - img[i,j,:])<3).all():
                same += 1
    score = float(same/(imgSize[0]*imgSize[1]))
    return score
def histo(img):
    h = np.zeros((256,256,3))
    bins = np.arange(256).reshape(256,1)
    color = [(255,0,0),(0,255,0),(0,0,255)]
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        cv2.normalize(histr,histr,0,255*.9,cv2.NORM_MINMAX)
        hist = np.int32(np.around(histr))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)
    h = np.flipud(h)
    return h
print(recoverScore(ori,interval2x))
histo_ori = histo(ori.astype(np.uint8))
cv2.imshow("0",histo_ori)
cv2.imwrite("histo_ori.jpg",histo_ori)
histo_int = histo(interval2x.astype(np.uint8))
cv2.imshow("1",histo_int)
cv2.imwrite("histo_2x.jpg",histo_int)
cv2.waitKey(0)
cv2.destroyAllWindows()