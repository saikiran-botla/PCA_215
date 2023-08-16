import numpy
import matplotlib.pyplot as plt
import random

numpy.random.seed(23+142)
# Returns theta in [-pi]
def generate_theta(a, b):
    theta = random.random()*0.5*numpy.pi 
    v = random.random() # solving theta according to v values
    if v < 0.25:
        return theta
    elif v < 0.5:
        return numpy.pi - theta
    elif v < 0.75:
        return numpy.pi + theta
    else:
        return -theta 
def random_point(a, b): #creating random location within the ellipse
    random_theta = generate_theta(a, b)
    k = numpy.sqrt(random.random())
    return numpy.array([
        a* numpy.cos(random_theta)*k ,
        b* numpy.sin(random_theta)*k
    ]) #sorting all such points into array

a = 1
b = 0.5
N= 10000000
points = [[0]*2]*N
points = numpy.array([random_point(a, b) for _ in range(N)])
x_coor = numpy.array(points[:,0])
y_coor = numpy.array(points[:,1])
plt.hist2d(x_coor,y_coor,bins =100,range =[[-2,2],[-1,1]])
#plt.show()
plt.savefig('./results/q1a.png')