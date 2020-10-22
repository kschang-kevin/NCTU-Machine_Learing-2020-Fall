import sys
import random
import math
import numpy as np

mean = float(sys.argv[1])
variance = float(sys.argv[2])

U = np.random.uniform(0, 1)
V = np.random.uniform(0, 1)

x = ((-2 * math.log(U)) ** (1 / 2)) * math.cos(math.pi * 2 * V)
y = ((-2 * math.log(U)) ** (1 / 2)) * math.sin(math.pi * 2 * V)

print(x, y)