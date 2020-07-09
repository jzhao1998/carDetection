import cv2
import numpy as np
from matplotlib import pyplot as plt
 
#由轮廓的顶点得到轮廓的中心坐标
def getCenter(list):
    center=[0,0]
    for i in range(len(list)):
        for j in range(2):
            center[j]+=list[i][0][j]
    for i in range(2):
        center[i]=int(center[i]/len(list))
    return center

#读取视频
cap = cv2.VideoCapture("DJI_0005.MP4")
#查看是否打开文件
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
frameNum = 0 
# 读取视频
while(cap.isOpened()):
  #一帧帧读取
  ret, frame = cap.read()
  frameNum += 1
  if ret == True:   
    tempframe = frame    
    if(frameNum==1):
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
    if(frameNum>=2):
        currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)        
        currentframe = cv2.absdiff(currentframe,previousframe) 
        #median即是二帧之间的区别
        median = cv2.medianBlur(currentframe,3)
        #设置阀值，主要是筛去一些因为光线变换导致的细微变换
        #对于其他情况，从这里开始，需要一步步调整参数
        ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
        #设置膨胀核
        kernel = np.ones((5, 5), np.uint8)
        #通过膨胀操作扩大车辆面积并抹去一些细小黑色部分
        dilation=cv2.dilate(median,kernel,iterations=5)
        #设置阀值，主要是筛去一些因为光线变换导致的细微变换
        _, binary = cv2.threshold(dilation,20,255,cv2.THRESH_BINARY)
        #通过cv2.RETR_EXTERNAL的mode得到最外部的轮廓
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        index=[]
        for i in range(len(contours)):
            if(cv2.contourArea(contours[i])<2000):
                index.append(i)
        contours=np.delete(contours,index)
        #把中心写到txt中以便保存，之后直接读取txt就行了。不然的话可以省略这一步
        file=open("centers.txt",'a')
        for i in contours:
            c=getCenter(i)
            file.write(str(c[0])+",")
            file.write(str(c[1])+"|")
        file.write("\n")
        file.close()
        cv2.drawContours(median,contours,-1,100,3)  
        #plt可以一张一张的播放图片
        #plt.imshow(median)
        #plt.show()
        cv2.namedWindow("median", 0)
        cv2.resizeWindow('median', 960, 540)
        cv2.imshow('median',median) 
 
        # 按Q退出
        if cv2.waitKey(33) & 0xFF == ord('q'):
          break    
    previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
  else: 
    break

cap.release()
cv2.destroyAllWindows()
