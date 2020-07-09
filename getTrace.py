import math
from matplotlib import pyplot as plt

videoSize=[1920,1080]


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

def distance(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

def drawTrace(list):
    x_val,y_val=[],[]
    for i in list:
        x_val.append(i[0])
        y_val.append(i[1])
    plt.xlim(0,videoSize[0])
    plt.ylim(0,videoSize[1])
    plt.plot(x_val,y_val)
    plt.show()

def getTrace(centerList):
    trace=[]
    finaltrace=[]
    for i in centerList[0]:
        trace.append([])
        trace[-1].append(i)
    for i in range(1,len(centerList)):
        for j in trace:
            d=1000000
            next=[0,0]
            for m in centerList[i]:
                if(distance(j[-1],m)<d):
                    next=m
                    d=distance(j[-1],m)
            if(next!=[0,0] and d<200):
                j.append(next)
                centerList[i].remove(next)
            elif(next==[0,0]):
                j.append(next)
                finaltrace.append(j)
                trace.remove(j)
        for j in centerList[i]:
            trace.append([j])
    j=0
    for i in trace:
        vector=[i[-1][0]-i[0][0],i[-1][1]-i[0][1]]
        if(vector[0]*vector[0]+vector[1]*vector[1]>100000):
            drawTrace(i)
            print("*********************")
            print(vector)
            j+=1
    print(j)

getTrace(getCarCenters("centers.txt"))