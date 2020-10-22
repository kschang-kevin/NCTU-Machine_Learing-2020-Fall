import sys
import random
import math
import numpy as np

mean = float(sys.argv[1])
variance = float(sys.argv[2])

def gaussian_data_generator():
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)

    x = ((-2 * math.log(U)) ** (1 / 2)) * math.cos(math.pi * 2 * V)
    y = ((-2 * math.log(U)) ** (1 / 2)) * math.sin(math.pi * 2 * V)

    return x, y


print('Data point source function: N(' + str(mean) + ', ' + str(variance) + ')')
print()
while new_mean - mean < 1e-6 and new_variance - variance < 1e-6:
    x, y = gaussian_data_generator(mean, variance)