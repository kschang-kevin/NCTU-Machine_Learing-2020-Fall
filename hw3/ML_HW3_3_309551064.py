import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
b = int(sys.argv[1])
n = int(sys.argv[2])
initial_a = float(sys.argv[3])
w = []
for i in range(n):
    w.append(float(sys.argv[i+4]))

def func(cofficient, n, imx):
    if n == 0:
        return cofficient[n] * (imx ** n)
    else:
        return cofficient[n] * (imx ** n) + func(cofficient, n-1, imx)

def polynomial_basis_linear_model_data_generator():
    x = np.random.uniform(-1, 1)
    U = -6
    for _ in range(12):
        U += np.random.uniform(0, 1)
    e = math.sqrt(initial_a)* U
    y = 0
    for i in range(n):
        y += w[i] * (x ** i)
    y += e
    return x, [[y]]

k = 0
mean = 0
variance = 0
new_variance = 0
new_mean = 0
data_x = []
data_y = []
previous_mu = np.full((n), -100)
while True:
    x, y = polynomial_basis_linear_model_data_generator()
    k += 1
    data_x.append(x)
    data_y.append(y[0][0])
    A = np.zeros((1, n), dtype=np.float)
    for i in range(n):
        A[0][i] = (x ** i)
    
    new_mean = (((k - 1) * new_mean) + y[0][0]) / k
    new_variance = new_variance + (mean ** 2) - (new_mean ** 2) + (((y[0][0] ** 2) - new_variance - (mean ** 2)) / (k + 1))
    
    if new_variance == 0:
        a = 1e-5
    else:
        a = new_variance
    
    if k == 1:
        sigma = np.linalg.inv(a * A.transpose().dot(A) + b * np.identity(n))
        mu = a * sigma.dot(A.transpose()).dot(y)
    else:
        previous_precision =  np.linalg.inv(sigma)
        sigma = np.linalg.inv(a * A.transpose().dot(A) + previous_precision)
        mu = sigma.dot(a * A.transpose().dot(y) + previous_precision.dot(mu))
    
    print('Add data point (' + str(x) + ', ' + str(y[0][0]) + '):')
    print()
    print('Posterior mean:')
    for i in range(n):
        print(mu[i][0])
    print()
    print('Posterior variance:')
    for i in range(n):
        for j in range(n):
            print(sigma[i][j], end = '')
            if j != n-1:
                print(', ', end = '')
        print()
    predictive_distribution_mean = A.dot(mu)
    predictive_distribution_variance = (1 / a) + A.dot(sigma).dot(A.transpose())
    print('Predictive distribution ~ N(' + str(predictive_distribution_mean[0][0]) + ', ' + str(predictive_distribution_variance[0][0]) + '):')
    print('--------------------------------------------------')
    error = 0
    for i in range(len(mu)):
        error += (mu[i] - previous_mu[i]) ** 2
    if k > 10000 or error < 1e-10:
        break

    mean = new_mean
    variance = new_variance
    previous_mu = mu
    if k == 10:
        k_10_data_x = data_x.copy()
        k_10_data_y = data_y.copy()
        k_10_mu = mu.copy()
        k_10_sigma = sigma.copy()
        k_10_a = a
    if k == 50:
        k_50_data_x = data_x.copy()
        k_50_data_y = data_y.copy()
        k_50_mu = mu.copy()
        k_50_sigma = sigma.copy()
        k_50_a = a

gs = gridspec.GridSpec(9, 9)
imx = np.arange(-2, 2, 1e-5)
#Ground Truth
ax1 = plt.subplot(gs[:4, :4])
ax1.set_title('Ground truth')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-20, 20])

imy = func(w, n-1, imx)
ax1.plot(imx, imy, color = 'black')

imy = func(w, n-1, imx) + initial_a
ax1.plot(imx, imy, color = 'red')

imy = func(w, n-1, imx) - initial_a
ax1.plot(imx, imy, color = 'red')

#Predict Result
ax2 = plt.subplot(gs[:4, 5:9])
ax2.set_title('Predict result')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-20, 20])

for i in range(len(data_x)):
    ax2.plot(data_x[i], data_y[i], 'b.')

predict_mean = func(mu, n-1, imx)
ax2.plot(imx, predict_mean, color = 'black')

predict_add_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_add_variance.append(predict_mean[i] + (1 / a) + A.dot(sigma).dot(A.transpose())[0][0])
ax2.plot(imx, predict_add_variance, color = 'red')

predict_minus_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_minus_variance.append(predict_mean[i] - ((1 / a) + A.dot(sigma).dot(A.transpose())[0][0]))
ax2.plot(imx, predict_minus_variance, color = 'red')

#10
ax3 = plt.subplot(gs[5:9, :4])
ax3.set_title('After 10 incomes')
ax3.set_xlim([-2, 2])
ax3.set_ylim([-20, 20])

for i in range(len(k_10_data_x)):
    ax3.plot(k_10_data_x[i], k_10_data_y[i], 'b.')

predict_mean = func(k_10_mu, n-1, imx)
ax3.plot(imx, predict_mean, color = 'black')

predict_add_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_add_variance.append(predict_mean[i] + (1 / k_10_a) + A.dot(k_10_sigma).dot(A.transpose())[0][0])
ax3.plot(imx, predict_add_variance, color = 'red')

predict_minus_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_minus_variance.append(predict_mean[i] - ((1 / k_10_a) + A.dot(k_10_sigma).dot(A.transpose())[0][0]))
ax3.plot(imx, predict_minus_variance, color = 'red')

#50
ax4 = plt.subplot(gs[5:9, 5:9])
ax4.set_title('After 50 incomes')
ax4.set_xlim([-2, 2])
ax4.set_ylim([-20, 20])

for i in range(len(k_50_data_x)):
    ax4.plot(k_50_data_x[i], k_50_data_y[i], 'b.')
  
predict_mean = func(k_50_mu, n-1, imx)
ax4.plot(imx, predict_mean, color = 'black')

predict_add_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_add_variance.append(predict_mean[i] + (1 / k_50_a) + A.dot(k_50_sigma).dot(A.transpose())[0][0])
ax4.plot(imx, predict_add_variance, color = 'red')

predict_minus_variance = []
for i in range(len(imx)):
    A = np.zeros((1, n), dtype=np.float)
    for j in range(n):
        A[0][j] = (imx[i] ** j)
    predict_minus_variance.append(predict_mean[i] - ((1 / k_50_a) + A.dot(k_50_sigma).dot(A.transpose())[0][0]))
ax4.plot(imx, predict_minus_variance, color = 'red')

plt.show()
