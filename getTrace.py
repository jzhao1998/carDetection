import math
from matplotlib import pyplot as plt

#有些视频经过剪辑软件之后会变小，可能需要改
videoSize=[3840,2160]

classes=["东","南","西","北","无"]

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
            list[-1][j][2]=int(list[-1][j][2])
            list[-1][j][3]=int(list[-1][j][3])
    return list

def getLocation(p):
    if(p[1]<850):
        return classes[3]
    if(p[1]>1480):
        return classes[1]
    if(p[0]<1170):
        return classes[2]
    if(p[0]>2010):
        return classes[0]
    return classes[4]




def calDirection(rect1,rect2):
    p1=[int((rect1[0]+rect1[2])/2),int((rect1[1]+rect1[3])/2)]
    p2=[int((rect2[0]+rect2[2])/2),int((rect2[1]+rect2[3])/2)]
    location1=getLocation(p1)
    location2=getLocation(p2)
    if(location1!=location2 and location1!=classes[4] and location2!=classes[4]):
        return location1,location2
    else:
        return None,None
    

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

def pltRect(list):
    plt.xlim(0,videoSize[0])
    plt.ylim(0,videoSize[1])
    for i in list:
        plt.gca().add_patch(plt.Rectangle((i[0],i[1]),i[2]-i[0],i[3]-i[1]))
    plt.show()

def calAreaCover(start1,end1,start2,end2):
    start=[max(start1[0],start2[0]),max(start1[1],start2[1])]
    end=[min(end1[0],end2[0]),min(end1[1],end2[1])]
    if(start[0]<=end[0] and start[1]<=end[1]):
        return (end[1]-start[1])*(end[0]-start[0])
    else:
        return 0

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
            d=0
            #下一个点坐标
            next=j[-1]
            #一个个点匹配，找到最近的点
            for m in centerList[i]:
                newD=calAreaCover((j[-1][0],j[-1][1]),(j[-1][2],j[-1][3]),(m[0],m[1]),(m[2],m[3]))
                if(newD>d):
                    next=m
                    d=newD
            #把最近的点加入路径
            if(next==j[-1]):
                finaltrace.append(j)
                needRemove.append(j)
            else:
                j.append(next)
                centerList[i].remove(next)
        #不能在同一个for循环中remove，不然会出错
        for j in needRemove:
            trace.remove(j)
        #如果有点没有被匹配，那就视为一个新的起点，短的路径之后会筛出来的
        for j in centerList[i]:
            trace.append([j])
    for i in trace:
        finaltrace.append(i)
    #只是我用来统计一下的工具
    j=0
    print(len(finaltrace))
    for i in finaltrace:
        if(len(i)>=50):
            vector=[i[-1][0]-i[0][0],i[-1][1]-i[0][1]]
            #print(i[0])
            #drawTrace(i)
            #print("*********************")
            #print(math.atan(vector[1]/vector[0]))
            #print(vector[0]*vector[0]+vector[1]*vector[1])
            direction=calDirection(i[-1],i[0])
            if(direction!=(None,None)):
                print(direction)
            j+=1
    print(j)

centers=getCarCenters("centers.txt")
getTrace(centers)
