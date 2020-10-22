import sys
import pandas as pd
import math 

filename = sys.argv[1]
a = int(sys.argv[2])
b = int(sys.argv[3])

df = pd.read_csv(filename, header = None)

for i in range(len(df)):
    N = len(df[0][i])
    M = df[0][i].count('1')
    likelihood = ((M / N) ** M) * ((1 - M / N) ** (N - M)) * math.factorial(N) / math.factorial(N - M) / math.factorial(M)
    print('case ' + str(i + 1) + ': ' + df[0][i])
    print('Likelihood: ', likelihood)
    print('Beta prior: a = ' + str(a) + ' b = ' + str(b))
    a = a + M
    b = N - M +b
    print('Beta posterior: a = ' + str(a) + ' b = ' + str(b))
    print()