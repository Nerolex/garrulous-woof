import time

import sklearn.svm as SVC

import DataLoader as dl

x, x_test, y, y_test = dl.load_libsvm_file("ijcnn")

linSvm = SVC.SVC(kernel="linear", C=10).fit(x, y)
gaussSvm = SVC.SVC(kernel="rbf", C=10, gamma=1).fit(x, y)

print("Linear SVM Predict Bulk:")
timeStartLin = time.time()
linSvm.predict(x_test)
print(str(time.time() - timeStartLin) + "s")

print("Single")
timeStartLin = time.time()
for elem in x_test:
    linSvm.predict(elem)
print(str(time.time() - timeStartLin) + "s")

print("Gauss Svm Predict Bulk:")
timeStartGauss = time.time()
gaussSvm.predict(x_test)
print(str(time.time() - timeStartGauss) + "s")

print("Single")
timeStartGauss = time.time()
for elem in x_test:
    gaussSvm.predict(elem)
print(str(time.time() - timeStartGauss) + "s")




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
'''
