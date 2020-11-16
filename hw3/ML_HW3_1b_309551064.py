import sys
import random
import math
import numpy as np

n = int(sys.argv[1])
a = float(sys.argv[2])
w = []
for i in range(n):
    w.append(float(sys.argv[i+3]))

x = np.random.uniform(-1, 1)

U = -6
for _ in range(12):
    U += np.random.uniform(0, 1)
e = math.sqrt(a)* U

y = 0
for i in range(n):
    y += w[i] * (x ** i)

y += e

print(y)