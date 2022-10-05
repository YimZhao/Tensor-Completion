import numpy as np
import cv2 
img = cv2.imread("./jpg/seaside1.jpg")
imsize = img.shape
X_old = np.zeros(imsize)

for i in range(0,imsize[0],2):
    for j in range(0,imsize[1],2):
        X_old[i,j,:] = img[i,j,:]
X = np.zeros([int(imsize[0]/2),int(imsize[1]/2),3])
Xsize = X.shape
for i in range(Xsize[0]):
    for j in range(Xsize[1]):
        X[i,j,:] = X_old[2*i,2*j,:]
print(Xsize)
cv2.namedWindow('Corrupting Image', cv2.WINDOW_NORMAL)
cv2.imshow("Corrupting Image", X.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("interval.jpg", X.astype(np.uint8))
