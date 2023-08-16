import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import linalg

f=h5py.File('./code/mnist.mat')

digits_train=np.array(f['digits_train'],dtype=float)
labels_train=np.array(f['labels_train'][0])

count=np.zeros(10)
mean=np.zeros((10,784,1))
cov_matrix=np.zeros((10,784,784))

for i in range(60000):
	j=labels_train[i]
	count[j]+=1
	a=np.reshape(digits_train[i],(784,1))
	mean[j]+=a
	cov_matrix[j]+=a*np.transpose(a)

for i in range(10):
	mean[i]=mean[i]/count[i]
	cov_matrix[i]=cov_matrix[i]-mean[i]*np.transpose(mean[i])

####################################################################################################
#the functions reduction and reconstruction are used to convert (784,1) to (84,1) and (84,1) to (784,1) respectively

def reduction(a,cov):
	x=np.zeros((84,1))
	eigen_values,eigen_vectors=linalg.eigh(cov)
	sorted_index=np.argsort(eigen_values)
	eigen_vectors=np.transpose(eigen_vectors)
	newvariance=np.zeros((84,784))

	for i in range(84):
		index=sorted_index[i]
		newvariance[i]=eigen_vectors[index]

	x=np.matmul(newvariance,a)
	return x

def reconstruction(a,cov):
	x=np.zeros((784,1))
	eigen_values,eigen_vectors=linalg.eigh(cov)
	sorted_index=np.argsort(eigen_values)
	eigen_vectors=np.transpose(eigen_vectors)
	newvariance=np.zeros((84,784))

	for i in range(84):
		index=sorted_index[i]
		newvariance[i]=eigen_vectors[index]

	newvariance=np.transpose(newvariance)
	x=np.matmul(newvariance,a)
	return x

######################################################################################################

for i in range(10):
	index=np.where(labels_train==i)
	m=index[0][0]
	#print(m)
	
	a=np.reshape(digits_train[m],(784,1))
	plt.imshow(digits_train[m])
	plt.savefig(f'./results/original_label{i}.png')
	a=a-mean[i]
	convert=reduction(a,cov_matrix[i])
	construct=reconstruction(convert,cov_matrix[i])
	newa=construct+mean[i]
	newa=np.reshape(newa,(28,28))
	plt.imshow(newa)
	plt.savefig(f'./results/reconstructed_label{i}.png')
