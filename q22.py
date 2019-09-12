import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from coeffs import *


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100

y1 = np.linspace(-4,4,len)
y2 = np.power(y1,2)
y2=(1/4)*y2

y = np.vstack((y2,y1))

#Given parabola parameters
V = np.array(([0,0],[0,1]))
u = -0.5*np.array([4,0])
F = 0

#Affine transformation
g = -0.5*np.array([4,0])
vcm = g-u
vcp = g+u
A = np.vstack((V,vcp.T))
b = np.append(vcm,-F)
c = LA.lstsq(A,b,rcond=None)[0]
c = c.reshape(2,1)

#Generating the parabola
x_par = y + c

#tangent is (-2 2)x=2

#Generating points on the tangent T

x1=np.array([3,4])
x2=np.array([-3,-2])

T=line_gen(x1,x2)

#defining centre and radius of Circle 
c=[0.0]
r=np.sqrt(5)

#Generating points on the circle 
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)

#plotting the parabola
plt.plot(x_par[0,:],x_par[1,:],label='Parabola')
#plotting tangent T
plt.plot(T[0,:],T[1,:],label='Tangent')
#plotting circle 
plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')

#Points
OptC=np.array([3/4,7/4])

#Plotting point
plt.plot(OptC[0],OptC[1], 'o')
plt.text( OptC[0], OptC[1]-.5, 'point')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()

plt.show()
