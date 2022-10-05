from asyncio.windows_events import NULL
import cv2
import tensorly as tl
import numpy as np
from numba import jit
from line_profiler import LineProfiler
import math
import random
from matplotlib import pyplot as plt

def shrinkage(X, t):
    U, Sig, VT = np.linalg.svd(X,full_matrices=False)

    Temp = np.zeros((U.shape[1], VT.shape[0]))
    for i in range(len(Sig)):
        Temp[i, i] = Sig[i]  
    Sig = Temp

    Sigt = Sig
    imSize = Sigt.shape

    for i in range(imSize[0]):
        Sigt[i, i] = np.max(Sigt[i, i] - t, 0)

    temp = np.dot(U, Sigt)
    T = np.dot(temp, VT)
    return T

@jit(nopython=True)
def ReplaceInd(X, known, Image):
    
    imSize = Image.shape
    
    for i in range(len(known)):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1)
        
        in2 = int(imSize[0] - known[i] % imSize[1] - 1)
        
        X[in1, in2, :] = Image[in1, in2, :]
        
    #print("Rank after replacement", np.linalg.matrix_rank(cv2.cvtColor(X.astype(np.uint8), cv2.COLOR_BGR2GRAY)))
    return X

@jit(nopython=True)
def ReplaceIndInterval(X, known, Image, step):
    imSize = Image.shape
    for i in range(imSize[0]):
        startj = 0
        if i%2 == 1:
            startj = 1
        for j in range(startj,imSize[1]):
            if known[i,j] == 1:
                X[i,j,:] = Image[i,j]

    
    '''
    for i in range(0,len(known),step):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1)
        in2 = int(imSize[0] - known[i] % imSize[1] - 1)
        X[in1, in2, :] = Image[in1, in2, :]
    '''
    '''
    for i in range(1,len(known),step):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1)
        in2 = int(imSize[0] - known[i] % imSize[1] - 1)
        in3 = int(np.ceil(known[i+1] / imSize[1]) - 1)
        in4 = int(imSize[0] - known[i+1] % imSize[1] - 1)
        in5 = int(np.ceil(known[i-1] / imSize[1]) - 1)
        in6 = int(imSize[0] - known[i-1] % imSize[1] - 1)
        X[in1, in2, :] = (Image[in3, in4, :]+Image[in5, in6, :])/2
    '''
    #print("Rank after replacement", np.linalg.matrix_rank(cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)))
    return X

@jit(nopython=True)
def ReplaceIndRndSq(X, map, Image, n, sqsize):
    #n: How many pixel is choosen in a square
    #sqsize: side length of a square
    imSize = Image.shape
    for i in range(0,imSize[0]):
        for j in range(0, imSize[1]):
            if map[i][j] == 1:
                X[i,j,:] = Image[i,j,:]
    return X

@jit(nopython=True)
def ReplaceIndRays(X, known, Image):
    imSize = Image.shape
    for i in range(0,imSize[0]):
        for j in range(0, imSize[1]):
            if known[i][j] == 1:
                X[i,j,:] = Image[i,j,:]
    '''
    #slices represents the number in pi/4
    imsize = Image.shape
    i_anchor = np.zeros(slices+1)
    j_anchor = np.zeros(slices+1)
    
    #print("1")
    
    for s in range(slices+1):
        i_anchor[s] = (imsize[1]/2)*math.tan(s*math.pi/(4*slices))
        j_anchor[s] = (imsize[0]/2)*math.tan(s*math.pi/(4*slices))
        #print(i_anchor[s])
        #print(j_anchor[s])
    i_point = np.zeros(2*slices+1)
    j_point = np.zeros(2*slices+1)
    
    #print("2")
    for i in range(2*slices+1):
        if i < slices:
            i_point[i] = -i_anchor[slices-i] + imsize[1]/2
            j_point[i] = -j_anchor[slices-i] + imsize[0]/2
        else:
            i_point[i] = imsize[1]/2 + i_anchor[-slices+i]
            j_point[i] = imsize[0]/2 + j_anchor[-slices+i]
        #print(i_point[i])
        #print(j_point[i])
    
    #print("3")
    
    for i in range(2*slices+1):
        p1i = 0
        p1j = i_point[i]
        p2i = imsize[0]-1
        p2j = i_point[-i]
        k = (p1j-p2j)/(p1i-p2i)
        m = (p1i*p2j-p1j*p2i)/(p1i-p2i)
        for row in range(imsize[0]):
            j = int(k*row+m)
            X[row,j,:] = Image[row,j,:]
            p1j = 0
            p1i = i_point[i]
            p2j = imsize[1]-1
            p2i = i_point[-i]
            k = (p1i-p2i)/(p1j-p2j)
            m = (p1j*p2i-p1i*p2j)/(p1j-p2j)
            for col in range(imsize[1]):
                j = int(k*col+m)-1
                if j >=0 and j < imsize[0]:
                    X[j,col,:] = Image[j,col,:]
    #print("Rank after replacement", np.linalg.matrix_rank(cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)))
    '''
    return X

def init():
    KownPercentage = 0.25
    Image = cv2.imread("./jpg/seaside1.jpg")
    imSize = Image.shape
    '''
    Image2 = cv2.imread("6.jpg")
    Image3 = cv2.imread("7.jpg")
    
    for i in range(imSize[0]):
        for j in range(imSize[1]):
            Image[i,j,1] = Image2[i,j,1]
            Image[i,j,2] = Image3[i,j,2]
    '''
    print("Rank of image", np.linalg.matrix_rank(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)))
    print("Choose replace method")
    print("1: random")
    print("2: interval")
    print("3: rays")
    print("4: random in square")
    
    Image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    Image_b, Image_g, Image_r = cv2.split(Image)
    Image_gray = np.resize(Image_gray,(imSize[0],imSize[1],1))
    Image_b = np.resize(Image_b,(imSize[0],imSize[1],1))
    Image_g = np.resize(Image_g,(imSize[0],imSize[1],1))
    Image_r = np.resize(Image_r,(imSize[0],imSize[1],1))
    
    print(imSize)
    Image = Image
    if len(imSize) == 3:
        known = np.arange(np.prod(imSize) / imSize[2])
    else:
         known = np.arange(np.prod(imSize))
         np.resize(Image,(imSize[0],imSize[1],1))
    m=input("Method:")
    X = np.zeros(imSize)
    n = 0
    sqsize = 0
    if m == '1':
        np.random.shuffle(known)
        known = known[:int(KownPercentage * (np.prod(imSize) / imSize[2]))]
        print(known.shape)
        
        X = np.zeros(imSize)
        X = ReplaceInd(X, known, Image)
    
    # Corrupting Image

    if m == '2':
        X = np.zeros(imSize)
        known = np.zeros((imSize[0],imSize[1]))
        for i in range(0,imSize[0],1):
            jstart = 0
            if i%2 == 1:
                jstart = 1
            for j in range(jstart,imSize[1],2):
                #if random.randint(0,9) < 8:
                    known[i,j] = 1
        X = ReplaceIndInterval(X,known,Image,2)
        for i in range(0,imSize[0]):
            for j in range(0, imSize[1]):
                if known[i][j] != 1:
                    if i < imSize[0]-3 and i > 2 and j < imSize[1]-3 > 2:
                        for c in range(3):
                            pixelsum = 0
                            pixelnum = 0
                            for x in range(i-2,i+2):
                                for y in range(j-2,j+2):
                                    if known[x,y]:
                                        pixelsum += Image[x,y,c]
                                        pixelnum += 1
                            X[i,j,c] = pixelsum/pixelnum

    if m == '3':
        X = np.zeros(imSize)
        known = np.zeros((imSize[0],imSize[1]))
        for i in range(imSize[0]):
            for j in range(imSize[1]):
                if Image[i,j,:].any() != 0:
                    known[i,j] = 1
        X = ReplaceIndRays(X,known,Image)
    
    if m == '4':
        known = np.zeros((imSize[0],imSize[1]))
        X = np.zeros(imSize)
        print("input n and sqsize")
        n = int(input("How many pixels in a square: "))
        sqsize = int(input("Sqaure size: "))
        for i in range(0,imSize[0],sqsize):
            for j in range(0, imSize[1], sqsize):
                subknown = np.arange(sqsize*sqsize)
                #print(subknown)
                np.random.shuffle(subknown)
                #print(subknown)
                subknown = subknown[:n]
                #print(subknown)
                for x in range(n):
                    xi = int(subknown[x]/sqsize)
                    xj = subknown[x]-xi*sqsize
                    if i+xi<imSize[0] and j+xj < imSize[1]:
                        known[i+xi,j+xj] = 1
        X = ReplaceIndRndSq(X,known,Image,n,sqsize)
        for i in range(0,imSize[0]):
            for j in range(0, imSize[1]):
                if known[i][j] != 1:
                    if i < imSize[0]-3 and i > 2 and j < imSize[1]-3 > 2:
                        for c in range(3):
                            pixelsum = 0
                            pixelnum = 0
                            for x in range(i-2,i+2):
                                for y in range(j-2,j+2):
                                    if known[x,y]:
                                        pixelsum += Image[x,y,c]
                                        pixelnum += 1
                            X[i,j,c] = pixelsum/pixelnum

    
    cv2.namedWindow('Corrupting Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Corrupting Image", X.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    corrname = "./output/corr.jpg"
    cv2.imwrite(corrname, X.astype(np.uint8))
    a = abs(np.random.rand(3, 1))
    a = a / np.sum(a)
    p = 1e-7
    K = 100
    ArrSize = np.array(imSize)
    ArrSize = np.append(ArrSize, 3)
    Mi = np.zeros(ArrSize)
    Yi = np.zeros(ArrSize)
    histo_ori = histo(Image.astype(np.uint8))
    cv2.imshow("0",histo_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    recname = "./output/histogram_ori.jpg"
    cv2.imwrite(recname, histo_ori.astype(np.uint8))
    return Image, X, known, a, Mi, Yi, imSize, ArrSize, p, K, m, n, sqsize

def fuc():
    Image, X, known, a, Mi, Yi, imSize, ArrSize, p, K, m, n, sqsize = init()
    
    for k in range(K):
        print(k)
        # compute Mi tensors(Step1)
        for i in range(ArrSize[3]):
            temp1 = shrinkage(tl.unfold(X, mode=i) + tl.unfold(np.squeeze(Yi[:, :, :, i]), mode=i) / p, a[i] / p)
            temp = tl.fold(temp1, i, imSize)
            Mi[:, :, :, i] = temp
        # Update X(Step2)
        X = np.sum(Mi - Yi / p, ArrSize[3]) / ArrSize[3]
        if m == '1':
            X = ReplaceInd(X, known, Image)
        if m == '2':
            X = ReplaceIndInterval(X,known,Image,2)
        if m == '3':
            X = ReplaceIndRays(X,known,Image)
        if m == '4':
            X = ReplaceIndRndSq(X,known,Image,n,sqsize)
        # Update Yi tensors (Step 3)
        for i in range(ArrSize[3]):
            Yi[:, :, :, i] = np.squeeze(Yi[:, :, :, i]) - p * (np.squeeze(Mi[:, :, :, i]) - X)
        # Modify rho to help convergence(Step 4)
        p = 1.2 * p
       
        print("Recover score: ", recoverScore(X,Image), X.max(),X.min())

    return X, Image

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

def removeOutBoundsPixels(X):
    imSize = X.shape
    for i in range(imSize[0]):
        for j in range(imSize[1]):
            for c in range(3):
                if (X[i,j,c]<0):
                    X[i,j,c] = 0
                if (X[i,j,c]>255):
                    X[i,j,c] = 255
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return X

lp = LineProfiler()
lp_wrapper = lp(fuc)
X, Image = lp_wrapper()
lp.print_stats()
X = removeOutBoundsPixels(X)
histo_rec = histo(X.astype(np.uint8))
cv2.imshow("1",histo_rec)
recname = "./output/histogram_rec.jpg"
cv2.imwrite(recname, histo_rec.astype(np.uint8))
print("Recover score after remove pixels out of bounds: ", recoverScore(X, Image))
cv2.namedWindow('HaLRTC', cv2.WINDOW_NORMAL)
cv2.imshow("HaLRTC", X.astype(np.uint8))
print("Rank after recover", np.linalg.matrix_rank(cv2.cvtColor(X.astype(np.uint8), cv2.COLOR_BGR2GRAY)))
cv2.waitKey(0)
cv2.destroyAllWindows()
recname = "./output/rec.jpg"
cv2.imwrite(recname, X.astype(np.uint8))