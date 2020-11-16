import sys
import random
import math
import numpy as np

mean = float(sys.argv[1])
variance = float(sys.argv[2])

U = -6
for _ in range(12):
    U += np.random.uniform(0, 1)

x = mean + math.sqrt(variance) * U
print(x)