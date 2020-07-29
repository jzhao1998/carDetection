import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pytesseract
import numpy as np
import pytesseract
from skimage import morphology
import os
import math

def array2png(img):
    return (255.0 / img.max() * (img - img.min())).astype(np.uint8)

def delpoints(img,blackground):
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            list=[]
            for m in range(3):
                for n in range(3):
                    list.append(img[i-1+m][j-1+n].tolist())
            if(list.count(list[4])<=2):
                img[i][j]=blackground
    return img

def calHist(img):
    img=img.tolist()
    histDict={}
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixel=tuple(img[i][j])
            if(pixel not in histDict):
                histDict[pixel]=1
            else:
                histDict[pixel]+=1
    histList=[[i,histDict[i]] for i in histDict]
    histList.sort(key=takeSecond)
    return histList,histDict

def takeSecond(elem):
    return elem[1]    

def blackthreshold(img,black):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if((img[i][j]==black).all()):
                img[i][j]=255
            else:
                img[i][j]=0
    return img

def specialErosion(img):
    histList,_=calHist(img)
    homoColor=homoImage(histList)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(histList[-2][1]<112):
                if(tuple(img[i][j].tolist())!=histList[-1][0]):
                    img[i][j]=np.array(histList[-2][0])
            else:
                if(tuple(img[i][j].tolist())!=histList[-2][0]):
                    img[i][j]=np.array(histList[-1][0])
    return img

def image2booleanImage(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j]==255):
                img[i][j]=0
            else:
                img[i][j]=1
    return img.astype(bool)

def booleanImage2Image(img):
    img=img.astype(int)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j]==1):
                img[i][j]=255
    return img

def delsawtooth(img,blackground,index):
    for m in range(index):
        for i in range(1,len(img)-1):
            for j in range(1,len(img[0])-1):
                list=[img[i-1][j],img[i+1][j],img[i][j-1],img[i][j+1]]
                if(list.count(blackground)>=3):
                    img[i][j]=blackground
    return img

def rgba2rgb(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if((img[i][j]==np.array([0,0,0,255])).all()):
                img[i][j]=np.array([255,255,255,255])
    return img[:,:,:3]

def homoImage(histDict):
    colors=[]
    for i in histDict:
        colors.append(i[0])
    baseColor=colors[-2]
    homoColor=[]
    for i in colors[:-2]:
        for j in range(3):
            if(abs(baseColor[j]-i[j])<10):
                if(i[j] not in homoColor):
                    homoColor.append(i)
    return homoColor





def transformImage(path,savepath):
    size=28
    for filename in os.listdir(path):
        img=cv2.imread(path+filename,cv2.IMREAD_UNCHANGED)
        img=rgba2rgb(img)
        imgNew=np.zeros([len(img),len(img[0])])
        for i in range(4):
            newImage=img[:,8+size*i:8+size*i+size]
            newImage=specialErosion(newImage)
            newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
            newImage = blackthreshold(newImage,0)
            newImage=delpoints(newImage,255)
            imgNew[:,8+size*i:8+size*i+size]=newImage
        imgNew=image2booleanImage(imgNew)
        imgNew = morphology.remove_small_objects(imgNew, min_size=20, connectivity=1,in_place=False)

        imgNew=delsawtooth(imgNew,False,3)
        imgNew=booleanImage2Image(imgNew)
        for i in range(45):
            for j in range(8):
                imgNew[i][j]=0
        cv2.imwrite(savepath+filename,imgNew)

if __name__ == "__main__":
    path="./a/"
    savepath="./t/"
    transformImage(path,savepath)
