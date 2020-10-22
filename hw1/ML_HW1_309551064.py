import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def func(cofficient, n, imx):
    if n == 0:
        return cofficient[n] * (imx ** n)
    else:
        return cofficient[n] * (imx ** n) + func(cofficient, n-1, imx)
        
filename = sys.argv[1]
n = int(sys.argv[2])
LSE_lambda = float(sys.argv[3])
data_point = pd.read_csv(filename, header = None)

#LSE
design_matrix = np.zeros((len(data_point), n), dtype=np.float)
for i in range(len(data_point)):
    for j in range(n):
        design_matrix[i][j] = data_point[0][i] ** (n-j-1)
design_matrix_T = design_matrix.transpose()
I = np.identity(n)
ATA_lamdaI = design_matrix_T.dot(design_matrix) + (LSE_lambda * I)

L = np.identity(n)
U = ATA_lamdaI.copy()

for i in range(1, n):
    for j in range(i):
        scalar = U[i][j] / U[j][j]
        for k in range(n):
            U[i][k] = U[i][k] - (scalar * U[j][k])
        L[i][j] = scalar
inverse_L = L.copy()
for i in range(1, n):
    for j in range(i):
        inverse_L[i][j] = -inverse_L[i][j]
inverse_U = np.identity(n)

for i in range(n-1, -1, -1):
    for j in range(n-1, -1, -1):
        if j > i:
            continue
        if i == j:
            scalar = 1 / U[i][i]
            for k in range(n):
                inverse_U[j][k] = scalar * inverse_U[j][k]
            U[j][i] = U[j][i] * scalar
        else:
            scalar = U[j][i] / U[i][i]
            for k in range(n):
                inverse_U[j][k] = inverse_U[j][k] - (scalar * inverse_U[i][k])
            U[j][i] = U[j][i] - (scalar * U[i][i])
    
inverse_ATA_lamdaI = inverse_U.dot(inverse_L)
b = np.zeros((len(data_point), 1), dtype=np.float)
for i in range(len(data_point)):
    b[i] = data_point[1][i]
x = inverse_ATA_lamdaI.dot(design_matrix_T).dot(b)
error = 0
for i in range(len(data_point)):
    ans = 0
    for j in range(n):
        ans += x[j] * (data_point[0][i] ** (n-j-1))
    error += (ans - data_point[1][i]) ** 2
print('LSE:')
print('Fitting line: ', end='')
for i in range(len(x)):
    print(abs(x[i][0]), end='')
    if i != len(x)-1:
        print('X^', end='')
        print(len(x)-i-1, end='')
        if x[i+1][0] < 0:
            print(' - ', end = '')
        if x[i+1][0] >= 0:
            print(' + ', end = '')
print()
print('Total error: ', end = '')
print(error[0])

"""
Newton's method
"""
result = np.full((n, 1), 100)
design_matrix = np.zeros((len(data_point), n), dtype=np.float)
for i in range(len(data_point)):
    for j in range(n):
        design_matrix[i][j] = data_point[0][i] ** (n-j-1)
design_matrix_T = design_matrix.transpose()
twoATAx = 2 * design_matrix_T.dot(design_matrix).dot(result)
b = np.zeros((len(data_point), 1), dtype=np.float)
for i in range(len(data_point)):
    b[i] = data_point[1][i] 
twoATb = 2 * design_matrix_T.dot(b)
gradient = twoATAx - twoATb
hessian_matrix = 2 * design_matrix_T.dot(design_matrix)
L = np.identity(n)
U = hessian_matrix.copy()
for i in range(1, n):
    for j in range(i):
        scalar = U[i][j] / U[j][j]
        for k in range(n):
            U[i][k] = U[i][k] - (scalar * U[j][k])
        L[i][j] = scalar
inverse_L = L.copy()
for i in range(1, n):
    for j in range(i):
        inverse_L[i][j] = -inverse_L[i][j]
inverse_U = np.identity(n)

for i in range(n-1, -1, -1):
    for j in range(n-1, -1, -1):
        if j > i:
            continue
        if i == j:
            scalar = 1 / U[i][i]
            for k in range(n):
                inverse_U[j][k] = scalar * inverse_U[j][k]
            U[j][i] = U[j][i] * scalar
        else:
            scalar = U[j][i] / U[i][i]
            for k in range(n):
                inverse_U[j][k] = inverse_U[j][k] - (scalar * inverse_U[i][k])
            U[j][i] = U[j][i] - (scalar * U[i][i])
    
inverse_hessian_matrix = inverse_U.dot(inverse_L)
update = gradient.transpose().dot(inverse_hessian_matrix)
result = result - update
error = 0
for i in range(len(data_point)):
    ans = 0
    for j in range(n):
        ans += result[0][j] * (data_point[0][i] ** (n-j-1))
    error += (ans - data_point[1][i]) ** 2
print('Newton\'s Method:')
print('Fitting line: ', end='')
for i in range(len(result)):
    print(abs(result[0][i]), end='')
    if i != len(result)-1:
        print('X^', end='')
        print(len(result)-i-1, end='')
        if result[0][i+1] < 0:
            print(' - ', end = '')
        if result[0][i+1] >= 0:
            print(' + ', end = '')
print()
print('Total error: ', end = '')
print(error)

"""
Draw
"""
fig, axs = plt.subplots(2)
for i in range(len(data_point)):
    axs[0].plot(data_point[0][i], data_point[1][i], 'ro')
    axs[1].plot(data_point[0][i], data_point[1][i], 'ro')

max = np.max(data_point.iloc[:, 0].values)
min = np.min(data_point.iloc[:, 0].values)

imx = np.array(range(int(math.floor(min))-3, int(math.ceil(max))+3))
imy = func(x[::-1], n-1, imx)
axs[0].plot(imx, imy)
answer = []
for i in range(n-1, -1, -1):
    answer.append(result[0][i])
imx = np.array(range(int(math.floor(min))-3, int(math.ceil(max))+3))
imy = func(answer, n-1, imx)
axs[1].plot(imx, imy)
plt.show()