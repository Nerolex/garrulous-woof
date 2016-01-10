import matplotlib.pyplot as plt
import numpy as np

'''
phi = np.linspace(0,2,100)
phi = phi * np.pi
x = np.cos(phi)
y = np.sin(phi)

x2 = 1.1 * x
y2 = 1.1 * y

z = x**2 + y**2
z2 = x2**2 + y2 **2

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.scatter(x, y, c="r")
ax1.scatter(x2, y2, c="b")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x, y, z, c="b")
ax2.scatter(x2, y2, z2, c="r")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_zticks([])
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

plt.show()
'''

# <w,x> + b = 0
# w = (1,2,3)  x= 1,1,1 b = -6
w = [1, 2, 3]
x = [1, 1, 1]
b = -6
print np.inner(w, x) + b

point = np.array(x)
normal = np.array(w)

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt.show()
