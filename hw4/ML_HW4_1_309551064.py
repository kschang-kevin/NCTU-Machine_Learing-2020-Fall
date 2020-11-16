import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def gaussian_data_generator(mean, variance): 
    U = -6
    for _ in range(12):
        U += np.random.uniform(0, 1)
    return mean + math.sqrt(variance) * U

n = int(sys.argv[1])

mx1 = float(sys.argv[2])
vx1 = float(sys.argv[3])
my1 = float(sys.argv[4])
vy1 = float(sys.argv[5])

mx2 = float(sys.argv[6])
vx2 = float(sys.argv[7])
my2 = float(sys.argv[8])
vy2 = float(sys.argv[9])

all_data_point = []
data_point_1 = []
data_point_2 = []

all_y = []
for _ in range(n):
    x = gaussian_data_generator(mx1, vx1)
    y = gaussian_data_generator(my1, vy1)
    all_data_point.append([x, y, 1])
    all_y.append([0])
    data_point_1.append([x, y])
    x = gaussian_data_generator(mx2, vx2)
    y = gaussian_data_generator(my2, vy2)
    all_data_point.append([x, y, 1])
    all_y.append([1])
    data_point_2.append([x, y])

all_data_point = np.array(all_data_point)
all_y = np.array(all_y)

gradient_descent_result = [[0], [0], [0]]
old_gradient_descent_result = [[0], [0], [0]]
while True:
    J = all_y - (1 / (1 + np.exp(-(all_data_point.dot(gradient_descent_result)))))
    gradient = all_data_point.transpose().dot(J)
    gradient_descent_result = gradient_descent_result + 0.01 * gradient
    if math.sqrt(abs(gradient[0][0] ** 2) + abs(gradient[1][0] ** 2) + abs(gradient[2][0]) ** 2) < 1e-2:
        break
    old_gradient_descent_result = gradient_descent_result

predict = 1 / (1 + np.exp(-(all_data_point.dot(gradient_descent_result))))

predict_1_real_1 = 0
predict_1_real_0 = 0
predict_0_real_1 = 0
predict_0_real_0 = 0

gradient_descent_data_point_1 = []
gradient_descent_data_point_2 = []
for i in range(len(predict)):
    if predict[i] < 0.5:
        gradient_descent_data_point_1.append(all_data_point[i])
        if all_y[i] == 0:
            predict_0_real_0 += 1
        else:
            predict_0_real_1 += 1
    else:
        gradient_descent_data_point_2.append(all_data_point[i])
        if all_y[i] == 1:
            predict_1_real_1 += 1
        else:
            predict_1_real_0 += 1

print('Gradient descent')
print('w:')
print(gradient_descent_result[0][0])
print(gradient_descent_result[1][0])
print(gradient_descent_result[2][0])
print('Confusion Matrix:')
print('               Predict cluster 1 Predict cluster 2')
print('Is Cluster 1         {:0>2d}                 {:0>2d}'.format(predict_0_real_0, predict_1_real_0))
print('Is Cluster 2         {:0>2d}                 {:0>2d}'.format(predict_0_real_1, predict_1_real_1))
print('Sensitivity (Successfully predict cluster 1): {:5f}'.format(predict_0_real_0 / (predict_0_real_0 + predict_1_real_0)))
print('Specificity (Successfully predict cluster 2): {:5f}'.format(predict_1_real_1 / (predict_0_real_1 + predict_1_real_1)))
# ==========================================================================================================

newton_method_result = [[0], [0], [0]]
old_newton_method_result = [[0], [0], [0]]

while True:
    D = np.identity(len(all_data_point))
    for i in range(len(all_data_point)):
        if -(all_data_point[i].dot(newton_method_result)) > 700:
            expo = 700
        else:
            expo = -(all_data_point[i].dot(newton_method_result))
        D[i][i] = math.exp(expo) / ((1 + math.exp(expo)) ** 2)
    H = all_data_point.transpose().dot(D).dot(all_data_point)
    J = all_y - (1 / (1 + np.exp(-(all_data_point.dot(newton_method_result)))))
    gradient = all_data_point.transpose().dot(J)
    if H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - H[0][2] * H[1][1] * H[2][0] - H[0][1] * H[1][0] * H[2][2] - H[0][0] * H[1][2] * H[2][1] == 0:
        newton_method_result = newton_method_result + 0.01 * gradient
    else:
        newton_method_result = newton_method_result + 0.01 * np.linalg.inv(H).dot(gradient)
    if math.sqrt(abs(newton_method_result[0][0] - old_newton_method_result[0][0]) ** 2 + abs(newton_method_result[1][0] - newton_method_result[1][0]) ** 2 + abs(newton_method_result[2][0] - old_newton_method_result[2][0]) ** 2) < 1e-2:
        break
    old_newton_method_result = newton_method_result

predict = 1 / (1 + np.exp(-(all_data_point.dot(newton_method_result))))

predict_1_real_1 = 0
predict_1_real_0 = 0
predict_0_real_1 = 0
predict_0_real_0 = 0

newton_method_data_point_1 = []
newton_method_data_point_2 = []
for i in range(len(all_data_point)):
    if predict[i] < 0.5:
        newton_method_data_point_1.append(all_data_point[i])
        if all_y[i] == 0:
            predict_0_real_0 += 1
        else:
            predict_0_real_1 += 1
    else:
        newton_method_data_point_2.append(all_data_point[i])
        if all_y[i] == 1:
            predict_1_real_1 += 1
        else:
            predict_1_real_0 += 1

print('Newton\'s method')

print('w:')
print(newton_method_result[0][0])
print(newton_method_result[1][0])
print(newton_method_result[2][0])
print('Confusion Matrix:')
print('               Predict cluster 1 Predict cluster 2')
print('Is Cluster 1         {:0>2d}                 {:0>2d}'.format(predict_0_real_0, predict_1_real_0))
print('Is Cluster 2         {:0>2d}                 {:0>2d}'.format(predict_0_real_1, predict_1_real_1))

print('Sensitivity (Successfully predict cluster 1): {:5f}'.format(predict_0_real_0 / (predict_0_real_0 + predict_1_real_0)))
print('Specificity (Successfully predict cluster 2): {:5f}'.format(predict_1_real_1 / (predict_0_real_1 + predict_1_real_1)))
#draw

gs = gridspec.GridSpec(9, 11)
ax1 = plt.subplot(gs[:9, :3])
ax1.set_title('Ground truth')
for i in range(n):
    ax1.plot(data_point_1[i][0], data_point_1[i][1], 'ro')
    ax1.plot(data_point_2[i][0], data_point_2[i][1], 'bo')

ax2 = plt.subplot(gs[:9, 4:7])
ax2.set_title('Grandient descent')
for i in range(len(gradient_descent_data_point_1)):
    ax2.plot(gradient_descent_data_point_1[i][0], gradient_descent_data_point_1[i][1], 'ro')
for i in range(len(gradient_descent_data_point_2)):    
    ax2.plot(gradient_descent_data_point_2[i][0], gradient_descent_data_point_2[i][1], 'bo')

ax3 = plt.subplot(gs[:9, 8:11])
ax3.set_title('Newton\'s method')
for i in range(len(newton_method_data_point_1)):
    ax3.plot(newton_method_data_point_1[i][0], newton_method_data_point_1[i][1], 'ro')
for i in range(len(newton_method_data_point_2)):
    ax3.plot(newton_method_data_point_2[i][0], newton_method_data_point_2[i][1], 'bo')


plt.show()