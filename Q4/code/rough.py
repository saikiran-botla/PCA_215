import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import h5py
from math import sqrt

f=h5py.File('./code/mnist.mat')

digits_train=np.array(f['digits_train'],dtype=float)
labels_train=np.array(f['labels_train'])

plt.imshow(np.transpose(digits_train[4]))
plt.show()