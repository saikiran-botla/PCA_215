import matplotlib.pyplot as plt
import numpy as np
import h5py

f=h5py.File('./code/points2D_Set1.mat','r')  # h5py package in python is used to read the data from the binary dat file

l=list(f.keys())
x=np.array(f['x'])
x=x.flatten()
#print(x[1])
y=np.array(f['y'])
y=y[0]
#print(y[1])

x_mean=0
y_mean=0

n=np.size(x)

for i in range(0,n):
	x_mean+=x[i]
	y_mean+=y[i]

x_mean=x_mean/np.size(x)
y_mean=y_mean/np.size(y)

slope_mean=((np.sum(y*x)-n*x_mean*y_mean)/(np.sum(x*x)-n*x_mean*x_mean))
cons_mean=y_mean-slope_mean*x_mean

#print(x*y)

plt.scatter(x,y,c='cornflowerblue')
plt.plot(x,x*slope_mean+cons_mean,'r')
plt.legend(f'y={slope_mean}x+{cons_mean}')

plt.xlabel("x",c='red')
plt.ylabel("y",c='red')
plt.savefig('./results/q3_set1.png') 