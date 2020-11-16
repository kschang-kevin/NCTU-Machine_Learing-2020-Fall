import sys
import random
import math
import numpy as np

def gaussian_data_generator(mean, variance):
    U = -6
    for _ in range(12):
        U += np.random.uniform(0, 1)
    x = mean + math.sqrt(variance) * U
    return x

initial_mean = float(sys.argv[1])
initial_variance = float(sys.argv[2])

n = 0
mean = 0
variance = 0
new_variance = 0
new_mean = 0

print('Data point source function: N(' + str(initial_mean) + ', ' + str(initial_variance) + ')')
print()

while True:
    x = gaussian_data_generator(initial_mean, initial_variance)
    n += 1
    new_mean = (((n - 1) * new_mean) + x) / n
    new_variance = new_variance + (mean ** 2) - (new_mean ** 2) + (((x ** 2) - new_variance - (mean ** 2)) / n) 
    
    print('Add data point: ' + str(x))
    print('Mean = ' + str(new_mean) + ' Variance = '+ str(new_variance))
    if abs(new_mean - mean) < 1e-4 and abs(new_variance - variance) < 1e-4:
        break
    else:
        mean = new_mean
        variance = new_variance