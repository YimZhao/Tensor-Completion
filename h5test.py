import h5py as h5
import numpy as np
import cv2
import matplotlib.pyplot as plt
def Normalization(data):
    _range = np.max(data)-np.min(data)
    return (255*(data-np.min(data))/_range).astype(np.uint8)
def gray2rgb(a,b,c):
    img = np.array([a,b,c])
    img = np.swapaxes(img, 2, 0)
    img = np.swapaxes(img, 1,0)
    return img

h5file = h5.File("E:/Thesis/BNL/Tensor-Completion-for-Estimating-Missing-Values-in-Visual-Data-master/h5/scan_70438.h5","r")
'''
for key in h5file.keys():
    print(h5file[key].name)
    print(h5file[key].shape)
    #print(h5file[key][()])
'''
co=np.array(h5file["Co_xrf"])
mn = np.array(h5file["Mn_xrf"])
ni = np.array(h5file["Ni_xrf"])
points = np.array(h5file["points"])
co = Normalization(co.reshape((105,120)))
mn = Normalization(mn.reshape((105,120)))
ni = Normalization(ni.reshape((105,120)))

img = gray2rgb(co,mn,ni)
print(img.shape)
'''
cv2.imshow('test1',co)
cv2.imshow('test2',mn)
cv2.imshow('test3',ni)
'''
cv2.imshow('test',img)
cv2.waitKey(0)
h5file.close()
saveimgid = "./h5/co.jpg"
cv2.imwrite(saveimgid, co)
saveimgid = "./h5/mn.jpg"
cv2.imwrite(saveimgid, co)
saveimgid = "./h5/ni.jpg"
cv2.imwrite(saveimgid, co)
'''
iimg = np.abs(iimg)
iimg0 = Normalization(iimg[0])

'''