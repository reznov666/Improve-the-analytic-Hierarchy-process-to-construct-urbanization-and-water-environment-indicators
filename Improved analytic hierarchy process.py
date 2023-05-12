import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
hrb1=pd.read_csv(r'D:\ck_data\hrbin.csv',index_col=0)
zy1=pd.read_csv(r'D:\ck_data\zunyi.csv',index_col=0) #Raw data import
hrb2=hrb1.values
zy2=zy1.values
hrb3=(hrb2-hrb2.min(1)[:,np.newaxis])/((hrb2.max(1)-hrb2.min(1))[:,np.newaxis])
zy3=(zy2-zy2.min(1)[:,np.newaxis])/((zy2.max(1)-zy2.min(1))[:,np.newaxis])
def ina():  #Judgment matrix input
    A1=np.array([[1,1,1/3,1/7],[1,1,1/2,1/7],[3,2,1,1/2],[7,7,2,1]])
    B1=np.array([[1,1/3,1],[3,1,2],[1,1/2,1]])
    B2=np.array([[1,1/2],[2,1]])
    B3=np.array([[1,1/3],[3,1]])
    B4=np.array([[1,1],[1,1]])
    A2=np.array([[1,1/3,1/5,1/9],[3,1,1,1/4],[5,1,1,1/3],[9,4,3,1]])
    B5=np.array([[1]])
    B6=np.array([[1,1/3],[3,1]])
    B7=np.array([[1,2],[1/2,1]])
    B8=np.array([[1,2,3,7],[1/2,1,2,5],[1/3,1/2,1,4],[1/7,1/5,1/4,1]])
    return A1,B1,B2,B3,B4,A2,B5,B6,B7,B8
def cha(x):  #Each column of the matrix is normalized
    m=np.sum(x,axis=0)
    f=np.ones((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x[i])):
            f[i][j]=x[i][j]/(m[j])
    return f
def chb(x):  #Vector normalization
    w1=(np.prod(x,axis=1))**(1/len(x))
    w=np.sum(w1)
    for i in range(len(w1)):
        w1[i]=w1[i]/w
    return w1
def chc(x,y,n):  #Solve for the largest characteristic root
    b=np.dot(y,x.T)
    r=0
    for i in range(len(b)):
        r=r+b[i]/(n*x[i])
    return r
def chd(x,n):  #Conduct consistency check
    if n == 1:
        cr=0
    else:
        c=(round(x,2)-n)/(n-1)
        ri=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45]
        if ri[int(n-1)]==0:
            cr=0
        else:
            cr=c/ri[int(n-1)]
    return cr 
def che(x_a):   #Calculate the weight of each underlying parameter and output consistency
    n=len(x_a)
    W1=cha(x_a)
    W2=chb(W1)
    W3=chc(W2,x_a,n)
    W4=chd(W3,n)
    print('一致性:{:.2f}'.format(W4))
    return W2

def Improve(X):     #Matrix self-consistency correction
    Xb=np.log10(X)
    Xc=np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xc[i,j]=np.sum(Xb[i]-Xb[j])/X.shape[1]
    XA=10**(Xc)
    return XA
a1,b1,b2,b3,b4,a2,b5,b6,b7,b8=ina() #Import judgment matrix
a1=Improve(a1)      #Consistency correction
a2=Improve(a2)
b1=Improve(b1)
b2=Improve(b2)
b3=Improve(b3)
b4=Improve(b4)
b5=Improve(b5)
b6=Improve(b6)
b7=Improve(b7)
b8=Improve(b8)

W11=che(a1)     #Calculate the weight of each level
W21=che(b1)
W22=che(b2)
W23=che(b3)
W24=che(b4)
W12=che(a2)
W25=che(b5)
W26=che(b6)
W27=che(b7)
W28=che(b8)

P31=np.array([zy3[0,:],zy3[1,:],zy3[2,:]]).T      #The weight is multiplied by the underlying index
P32=np.array([zy3[3,:],zy3[4,:]]).T
P33=np.array([zy3[5,:],zy3[6,:]]).T
P34=np.array([zy3[7,:],zy3[8,:]]).T
P35=np.array([zy3[9,:]]).T
P36=np.array([zy3[10,:],zy3[11,:]]).T
P37=np.array([zy3[12,:],zy3[13,:]]).T
P38=np.array([zy3[14,:],zy3[15,:],zy3[16,:],zy3[17,:]]).T

P21=np.dot(P31,W21)
P22=np.dot(P32,W22)
P23=np.dot(P33,W23)
P24=np.dot(P34,W24)
P25=np.dot(P35,W25)
P26=np.dot(P36,W26)
P27=np.dot(P37,W27)
P28=np.dot(P38,W28)
P24=1/(1+2.68*np.exp(-11.42*P24))   #Nonlinear correction
P28=0.5*np.exp(0.85*P28)

P11=np.array([P21,P22,P23,P24]).T
P12=np.array([P25,P26,P27,P28]).T
S1=np.dot(P11,W11)
S2=np.dot(P12,W12)
datazy=np.array([8.5,4.6,4.6,10.1,4.6,4.6,21.7,35.1,18.3,29.8,18,19.3,13.6,19.4,13,13,0])   #Import the measured value
datazy=datazy/100
# datazy=(datazy-datazy.min())/(datazy.max()-datazy.min())
year=np.arange(2003,2020,1).astype(dtype=np.str)
year=pd.Series(year)
fig = plt.figure(figsize=(10, 6))       #Draw the curve
ax = fig.add_subplot(1,1,1)
ax.plot(year,S1-S2,'-ob',linewidth=2,label='the water quality pressure index')
ax.plot(year,datazy,'-or',linewidth=2,label='non-compliance rate of urban water quality')
ax.set_ylabel('Predicted and True Values',fontsize=16)
ax.set_xlabel('year',fontsize=18)
ax.set_ylim(-0.6,0.5)
ax.set_title('Comparison in Zunyi',fontsize=20)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.text('2003',0.4,'(a)',va='top',ha='left',fontsize=18)
plt.legend(frameon=False)
ax.annotate('Wujiang River water pollution incident in 2009',xy=('2009',0.2),xycoords='data',
            xytext=(30,-90),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-0.1'))
