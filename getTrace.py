import math
from matplotlib import pyplot as plt

#有些视频经过剪辑软件之后会变小，可能需要改
videoSize=[3840,2160]

classes=["东","南","西","北"]

#读取文件中的内容
def getCarCenters(filename):
    file=open(filename,'r')
    list=[]
    for i in file.readlines():
        list.append([])
        list[-1]=i.split("|")
        list[-1].remove('\n')
        for j in range(len(list[-1])):
            list[-1][j]=list[-1][j].split(',')
            list[-1][j][0]=int(list[-1][j][0])
            list[-1][j][1]=int(list[-1][j][1])
    return list

def calDirection(p1,p2):
    if(p2[1]<850):
        deg=math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))
        if(deg<0.3 and deg>-0.3):
            return [classes[3],classes[2]]
        elif(deg<-0.3 or deg >-0.9):
            return [classes[3],classes[1]]
        else:
            return [classes[3],classes[0]]
    if(p2[1]>1480):
        deg=math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))
        if(deg<0.3 and deg>-0.3):
            return [classes[1],classes[0]]
        elif(deg<-0.3 or deg >-0.9):
            return [classes[1],classes[3]]
        else:
            return [classes[1],classes[2]]
    if(p2[0]<1170):
        deg=math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))
        if(deg<0.3 and deg>-0.3):
            return [classes[2],classes[3]]
        elif(deg<-0.3 or deg >-0.9):
            return [classes[2],classes[0]]
        else:
            return [classes[2],classes[1]]
    if(p2[0]>2010):
        deg=math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))
        if(deg<0.3 and deg>-0.3):
            return [classes[0],classes[1]]
        elif(deg<-0.3 or deg >-0.9):
            return [classes[0],classes[2]]
        else:
            return [classes[0],classes[3]]

#只是方便用来计算点与点之间的距离
def distance(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

#画线，主要是用来显示算出来的路径
def drawTrace(list):
    x_val,y_val=[],[]
    for i in list[:-1]:
        x_val.append(i[0])
        y_val.append(videoSize[1]-i[1])
    plt.xlim(0,videoSize[0])
    plt.ylim(0,videoSize[1])
    plt.plot(x_val,y_val)
    plt.show()

def drawPoint(i,list):
    plt.title(str(i))
    plt.xlim(0,videoSize[0])
    plt.ylim(0,videoSize[1])
    for i in list:
        plt.plot(i[0],i[1],'g^')
    plt.show()

#计算路径，参数是之前算出来的中心矩阵
def getTrace(centerList):
    trace=[]
    finaltrace=[]
    #从第一张图片中的车辆中心开始计算
    for i in centerList[0]:
        trace.append([])
        trace[-1].append(i)
    #一张一张来匹配之前的图片中的中心
    for i in range(1,len(centerList)):
        needRemove=[]
        #一个个尝试匹配中心
        for j in trace:
            #这只是一个极大数
            d=300
            #下一个点坐标
            next=[0,0]
            #一个个点匹配，找到最近的点
            for m in centerList[i]:
                if(distance(j[-1],m)<d):
                    next=m
                    d=distance(j[-1],m)
            #把最近的点加入路径
            j.append(next)     
        #如果最后一个点是[0,0]，就是指在范围内找不到合适的点，也即这条路径断了  
        for j in trace:
            if(j[-1]!=[0,0] and j[-1] in centerList[i]):
                centerList[i].remove(j[-1])
            else:
                finaltrace.append(j)
                needRemove.append(j)
        #不能在同一个for循环中remove，不然会出错
        for j in needRemove:
            trace.remove(j)
        #如果有点没有被匹配，那就视为一个新的起点，短的路径之后会筛出来的
        for j in centerList[i]:
            trace.append([j])
    #只是我用来统计一下的工具
    j=0
    print(len(finaltrace))
    for i in finaltrace:
        if(len(i)>=3):
            vector=[i[-2][0]-i[0][0],i[-2][1]-i[0][1]]
        else:
            vector=[i[-2][0]-i[0][0],i[-2][1]-i[0][1]]
        #路径总长度太短，基本都是一些干扰因素，需要筛掉
        if(vector[0]*vector[0]+vector[1]*vector[1]>100000):
            #drawTrace(i)
            print("*********************")
            print(i)
            print(vector[0]*vector[0]+vector[1]*vector[1])
            j+=1
    print(j)

centers=getCarCenters("b.txt")
getTrace(centers)
