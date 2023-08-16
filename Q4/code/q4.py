import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import h5py
from math import sqrt

f=h5py.File('./code/mnist.mat')

digits_train=np.array(f['digits_train'],dtype=float)
labels_train=np.array(f['labels_train'])

count=np.zeros(10)
mean=np.zeros((10,28*28,1))
cov_matrix=np.zeros((10,784,784))
eigen_values=np.zeros((10,784))
eigen_vectors=np.zeros((10,784,784))

for i in range(60000):
	j=labels_train[0][i]
	count[j]+=1
	a=np.reshape(digits_train[i],(784,1))
	#print(a.shape)
	mean[j]+=a
	#print((a*np.transpose(a))[300][300])
	cov_matrix[j]+=np.matmul(a,np.transpose(a))
#print(cov_matrix[1][300][300])

for i in range(10):
	mean[i]=mean[i]/count[i]
	cov_matrix[i]=cov_matrix[i]/count[i]
	print(f'printing the mean of label {i} ')
	print(mean[i])
	print('')
	m=np.reshape(mean[i],(28,28))
	plt.imshow(m)
	plt.savefig(f'./results/mean_label{i}.png')
	plt.close()

sorted_eigenvalues=np.zeros((10,784))

for i in range(10):
	cov_matrix[i]=cov_matrix[i]-np.matmul(mean[i],np.transpose(mean[i]))
	#print(f'printing the covariance matrix of label {i} .......')
	#print(cov_matrix[i])
	#print('')
	eigen_values[i],eigen_vectors[i]=linalg.eigh(cov_matrix[i])
	sorted_eigenvalues[i]=np.sort(eigen_values[i])

	print(f'printing the lamda1 of label {i}',sorted_eigenvalues[i][-1])

	lamda1=sorted_eigenvalues[i][-1]
	index=np.where(eigen_values[i]==lamda1)
	v1=eigen_vectors[i][:][index]
	#print(f'printing the eigen_vector v1 of label {i}.........')
	#print(eigen_vectors[i][index])
	x=np.arange(784)
	plt.xlabel('count')
	plt.ylabel('eigen_values')
	plt.plot(x,sorted_eigenvalues[i])
	plt.savefig(f'./results/eigenvalues_label{i}.png')
	plt.close()
	v1=np.reshape(v1,(784,1))
	d1=np.zeros((784,1))
	d1=mean[i]-sqrt(lamda1)*v1
	data1=np.reshape(d1,(28,28))
	plt.imshow(data1)
	plt.savefig(f'./results/minus_lamda1_label{i}.png')
	plt.close()

	d2=np.zeros((784,1))
	d2=mean[i]+sqrt(lamda1)*v1
	data2=np.reshape(d2,(28,28))
	plt.imshow(data2)
	plt.savefig(f'./results/plus_lamda1_label{i}.png')
	plt.close()





#print(cov_matrix[1][300][300])

