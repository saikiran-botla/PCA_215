import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.linalg import norm

np.random.seed(23+142)
def matrixsum(A,B) :
    r = np.array([[A[i][j]+B[i][j] for j in range(len(A[0]))]for i in range(len(A))])
    return r  
    #for matrix addition

def matrixsub(A,B) :
    r = np.array([[A[i][j]-B[i][j] for j in range(len(A[0]))]for i in range(len(A))])
    return r
    # for subtraction of matrix

def column(A,i) :
    v = np.array([[row[i]] for row in A])
    return v
  # for taking elements columnwise
   

mean = np.array([[1],[2]]) #µ
cov = np.array([[1.6250, -1.9486],[-1.9486, 3.8750]]) #AA'
evalue, evect = np.linalg.eig(cov)
r = np.sqrt(evalue)
A = r*(evect)
# X = A*W + µ

e_m0 = [[0.0]*1]*100
e_m1 = [[0.0]*1]*100
e_m2 = [[0.0]*1]*100
e_m3 = [[0.0]*1]*100
e_m4 = [[0.0]*1]*100
e_c0 = [[0.0]*1]*100
e_c1 = [[0.0]*1]*100
e_c2 = [[0.0]*1]*100
e_c3 = [[0.0]*1]*100
e_c4 = [[0.0]*1]*100
 #these are to store error in mean and covariance.

N = pow(10,1)
for j in range(100) :
    x1 = [0.0]*N
    x2 = [0.0]*N
    cov = [[0.0]*2]*2
    for k in range(N) :
        p = np.random.normal()
        q = np.random.normal()
        W = np.array([[p],[q]])
        X = matrixsum(np.dot(A,W) , mean)
        cov = matrixsum(cov , np.dot(X,np.transpose(X)))
        x1[k] = X[0][0]
        x2[k] = X[1][0]
    sumx1=sum(x1)    #sum of all the x1 
    sumx2=sum(x2)    #sum of all the x2 
    avgx1= sumx1/N       #average of X1 
    avgx2= sumx2/N       #average of X2 
    ML_m=np.array([[avgx1],[avgx2]])   #ML estimate
    ML_cov = cov/N                      
    ML_cov = matrixsub(ML_cov , np.dot(ML_m,np.transpose(ML_m)))  #ML_estimate
    err1 = np.dot(np.transpose(matrixsub(ML_m , mean)),matrixsub(ML_m , mean))
    err2 = np.transpose(mean)*mean
    e_m0[j] = math.sqrt(abs(err1[0][0]/err2[0][0]))
    e_c0[j]=norm(matrixsub(ML_cov,cov))/norm(cov)
    if j == 1:
        [evalues_mlcov,evect_mlcov] = np.linalg.eig(ML_cov)
        plt.scatter(x1,x2)
        pt1 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking1 = np.concatenate((np.transpose(ML_m),np.transpose(pt1)))
        pt2 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking2 = np.concatenate((np.transpose(ML_m),np.transpose(pt2)))
        pt3 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1)) 
        marking3 = np.concatenate((np.transpose(ML_m),np.transpose(pt3)))         
        pt4 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1))          
        marking4 = np.concatenate((np.transpose(ML_m),np.transpose(pt4)))
    #  these points are for plotting a line starting at the empirical mean and going a distance equal to the empirical eigen value along a direction given by the empirical eigen-vector.
        plt.title("scatter plot for N points x2 v/s x1 , N = 10" )
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.plot( column(marking1,0),column(marking1,1),'k-')
        plt.plot( column(marking2,0),column(marking2,1),'k-')
        plt.plot( column(marking3,0),column(marking3,1),'k-')
        plt.plot( column(marking4,0),column(marking4,1),'k-')
        plt.show()

    
N = pow(10,2)
for j in range(100) :
    x1 = [0.0]*N
    x2 = [0.0]*N
    cov = [[0.0]*2]*2
    for k in range(N) :
        p = np.random.normal()
        q = np.random.normal()
        W = np.array([[p],[q]])
        X = matrixsum(np.dot(A,W) , mean)
        cov = matrixsum(cov , np.dot(X,np.transpose(X)))
        x1[k] = X[0][0]
        x2[k] = X[1][0]
    sumx1=sum(x1)    #sum of all the x1 
    sumx2=sum(x2)    #sum of all the x2 
    avgx1= sumx1/N       #average of X1 
    avgx2= sumx2/N       #average of X2 
    ML_m=np.array([[avgx1],[avgx2]])  #ML_estimate
    ML_cov = cov/N                       
    ML_cov = matrixsub(ML_cov , np.dot(ML_m,np.transpose(ML_m)))
    err1 = np.dot(np.transpose(matrixsub(ML_m , mean)),matrixsub(ML_m , mean)) #ML_estimate
    err2 = np.transpose(mean)*mean
    e_m1[j] = math.sqrt(abs(err1[0][0]/err2[0][0]))
    e_c1[j]=norm(matrixsub(ML_cov,cov))/norm(cov)
    if j == 1:
        [evalues_mlcov,evect_mlcov] = np.linalg.eig(ML_cov)
        plt.scatter(x1,x2)
        pt1 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking1 = np.concatenate((np.transpose(ML_m),np.transpose(pt1)))
        pt2 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking2 = np.concatenate((np.transpose(ML_m),np.transpose(pt2)))
        pt3 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1)) 
        marking3 = np.concatenate((np.transpose(ML_m),np.transpose(pt3)))         
        pt4 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1))          
        marking4 = np.concatenate((np.transpose(ML_m),np.transpose(pt4)))
#  these points are for plotting a line starting at the empirical mean and going a distance equal to the empirical eigen value along a direction given by the empirical eigen-vector.
        
        plt.title("scatter plot for N points x2 v/s x1 , N = 100" )
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.plot( column(marking1,0),column(marking1,1),'k-')
        plt.plot( column(marking2,0),column(marking2,1),'k-')
        plt.plot( column(marking3,0),column(marking3,1),'k-')
        plt.plot( column(marking4,0),column(marking4,1),'k-')
        plt.show()


N = pow(10,3)
for j in range(100) :
    x1 = [0.0]*N
    x2 = [0.0]*N
    cov = [[0.0]*2]*2
    for k in range(N) :
        p = np.random.normal()
        q = np.random.normal()
        W = np.array([[p],[q]])
        X = matrixsum(np.dot(A,W) , mean)
        cov = matrixsum(cov , np.dot(X,np.transpose(X)))
        x1[k] = X[0][0]
        x2[k] = X[1][0]
    sumx1=sum(x1)    #sum of all the x1 
    sumx2=sum(x2)    #sum of all the x2 
    avgx1= sumx1/N       #average of X1 
    avgx2= sumx2/N       #average of X2 
    ML_m=np.array([[avgx1],[avgx2]])  #ML_estimate
    ML_cov = cov/N                       
    ML_cov = matrixsub(ML_cov , np.dot(ML_m,np.transpose(ML_m)))
    err1 = np.dot(np.transpose(matrixsub(ML_m , mean)),matrixsub(ML_m , mean)) #ML_estimate
    err2 = np.transpose(mean)*mean
    e_m2[j] = math.sqrt(abs(err1[0][0]/err2[0][0]))
    e_c2[j]=norm(matrixsub(ML_cov,cov))/norm(cov)
    if j == 1:
        [evalues_mlcov,evect_mlcov] = np.linalg.eig(ML_cov)
        plt.scatter(x1,x2)
        pt1 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking1 = np.concatenate((np.transpose(ML_m),np.transpose(pt1)))
        pt2 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking2 = np.concatenate((np.transpose(ML_m),np.transpose(pt2)))
        pt3 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1)) 
        marking3 = np.concatenate((np.transpose(ML_m),np.transpose(pt3)))         
        pt4 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1))          
        marking4 = np.concatenate((np.transpose(ML_m),np.transpose(pt4)))
#  these points are for plotting a line starting at the empirical mean and going a distance equal to the empirical eigen value along a direction given by the empirical eigen-vector.
         
        plt.title("scatter plot for N points x2 v/s x1 , N = 1000" )
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.plot( column(marking1,0),column(marking1,1),'k-')
        plt.plot( column(marking2,0),column(marking2,1),'k-')
        plt.plot( column(marking3,0),column(marking3,1),'k-')
        plt.plot( column(marking4,0),column(marking4,1),'k-')
        plt.show()


N = pow(10,4)
for j in range(100) :
    x1 = [0.0]*N
    x2 = [0.0]*N
    cov = [[0.0]*2]*2
    for k in range(N) :
        p = np.random.normal()
        q = np.random.normal()
        W = np.array([[p],[q]])
        X = matrixsum(np.dot(A,W) , mean)
        cov = matrixsum(cov , np.dot(X,np.transpose(X)))
        x1[k] = X[0][0]
        x2[k] = X[1][0]
    sumx1=sum(x1)    #sum of all the x1 
    sumx2=sum(x2)    #sum of all the x2 
    avgx1= sumx1/N       #average of X1 
    avgx2= sumx2/N       #average of X2 
    ML_m=np.array([[avgx1],[avgx2]])  #ML_estimate
    ML_cov = cov/N                       
    ML_cov = matrixsub(ML_cov , np.dot(ML_m,np.transpose(ML_m)))
    err1 = np.dot(np.transpose(matrixsub(ML_m , mean)),matrixsub(ML_m , mean)) #ML_estimate
    err2 = np.transpose(mean)*mean
    e_m3[j] = math.sqrt(abs(err1[0][0]/err2[0][0]))
    e_c3[j]=norm(matrixsub(ML_cov,cov))/norm(cov)
    if j == 1:
        [evalues_mlcov,evect_mlcov] = np.linalg.eig(ML_cov)
        plt.scatter(x1,x2)
        pt1 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking1 = np.concatenate((np.transpose(ML_m),np.transpose(pt1)))
        pt2 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking2 = np.concatenate((np.transpose(ML_m),np.transpose(pt2)))
        pt3 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1)) 
        marking3 = np.concatenate((np.transpose(ML_m),np.transpose(pt3)))         
        pt4 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1))          
        marking4 = np.concatenate((np.transpose(ML_m),np.transpose(pt4)))
#  these points are for plotting a line starting at the empirical mean and going a distance equal to the empirical eigen value along a direction given by the empirical eigen-vector.
         
        plt.title("scatter plot for N points x2 v/s x1 , N = 10000" )
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.plot( column(marking1,0),column(marking1,1),'k-')
        plt.plot( column(marking2,0),column(marking2,1),'k-')
        plt.plot( column(marking3,0),column(marking3,1),'k-')
        plt.plot( column(marking4,0),column(marking4,1),'k-')
        plt.show()


N = pow(10,5)
for j in range(100) :
    x1 = [0.0]*N
    x2 = [0.0]*N
    cov = [[0.0]*2]*2
    for k in range(N) :
        p = np.random.normal()
        q = np.random.normal()
        W = np.array([[p],[q]])
        X = matrixsum(np.dot(A,W) , mean)
        cov = matrixsum(cov , np.dot(X,np.transpose(X)))
        x1[k] = X[0][0]
        x2[k] = X[1][0]
    sumx1=sum(x1)    #sum of all the x1 
    sumx2=sum(x2)    #sum of all the x2 
    avgx1= sumx1/N       #average of X1 
    avgx2= sumx2/N       #average of X2 
    ML_m=np.array([[avgx1],[avgx2]])  #ML_estimate
    ML_cov = cov/N                       
    ML_cov = matrixsub(ML_cov , np.dot(ML_m,np.transpose(ML_m)))
    err1 = np.dot(np.transpose(matrixsub(ML_m , mean)),matrixsub(ML_m , mean)) #ML_estimate
    err2 = np.transpose(mean)*mean
    e_m4[j] = math.sqrt(abs(err1[0][0]/err2[0][0]))
    e_c4[j]=norm(matrixsub(ML_cov,cov))/norm(cov)
    if j == 1:
        [evalues_mlcov,evect_mlcov] = np.linalg.eig(ML_cov)
        plt.scatter(x1,x2)
        pt1 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking1 = np.concatenate((np.transpose(ML_m),np.transpose(pt1)))
        pt2 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[0]))*column(evect_mlcov,0))
        marking2 = np.concatenate((np.transpose(ML_m),np.transpose(pt2)))
        pt3 = matrixsum(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1)) 
        marking3 = np.concatenate((np.transpose(ML_m),np.transpose(pt3)))         
        pt4 = matrixsub(ML_m , math.sqrt(abs(evalues_mlcov[1]))*column(evect_mlcov,1))          
        marking4 = np.concatenate((np.transpose(ML_m),np.transpose(pt4)))
#  these points are for plotting a line starting at the empirical mean and going a distance equal to the empirical eigen value along a direction given by the empirical eigen-vector.
         
        plt.title("scatter plot for N points x2 v/s x1 , N = 100000" )
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.plot( column(marking1,0),column(marking1,1),'k-')
        plt.plot( column(marking2,0),column(marking2,1),'k-')
        plt.plot( column(marking3,0),column(marking3,1),'k-')
        plt.plot( column(marking4,0),column(marking4,1),'k-')
        plt.show()


data = [e_m0,e_m1,e_m2,e_m3,e_m4]
fig, ax = plt.subplots()
ax.set_title('Error measure: $||µ-$µ_N$||_2$/ $||µ||_2$ / v/s $log_{10}$(N)')
ax.set_ylabel(' $log_{10}$(N)')
ax.set_xlabel('error measure between the true mean µ and $µ_{est}$')
plt.boxplot(data)
plt.show()

data =[e_c0,e_c1,e_c2,e_c3,e_c4]
fig, ax = plt.subplots()
ax.set_title('Error measure: $||C-$C_N$||_{Fro}$/$||C||_{Fro}$ vs $log_{10}$(N)')
ax.set_ylabel(' $log_{10}$(N)')
ax.set_xlabel('error measure between  C and $C_{est}$')
plt.boxplot(data)
plt.show() 
