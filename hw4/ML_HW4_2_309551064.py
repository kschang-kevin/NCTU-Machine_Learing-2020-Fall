import gzip
import struct
import sys
import math
import numpy as np
import numba as nb

def read_files():
  #read training image
  with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()
  file_content = file_content[16:]
  training_image = []
  for i in range(60000):
    training_image.append(file_content[784*i:784*(i+1)])

  training_image_bin = np.full((60000, 784), 0)
  for i in range(60000):
    for j in range(784):
      if training_image[i][j] < 128:
        training_image_bin[i][j] = 0
      else:
        training_image_bin[i][j] = 1

  #read training label
  with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    file_content = f.read()
  file_content = file_content[8:]
  training_label = []
  for i in range(60000):
    training_label.append(file_content[i:i+1])
  for i in range(len(training_label)):
    training_label[i] = struct.unpack('>b',training_label[i])[0]
  training_label = np.array(training_label)
  return training_image_bin, training_label

@nb.jit
def E_step(training_image_bin, IM, L, W):
  for i in range(60000):
    prob = []
    prob_sum = 0
    for j in range(10):
      tmp = 1
      for k in range(784):
        if training_image_bin[i][k] == 1:
          tmp *= IM[j][k]
        else:
          tmp *= (1-IM[j][k])
      tmp *= L[j][0]
      prob_sum += tmp
      prob.append(tmp)
    if prob_sum == 0:
      prob_sum = 1
    for l in range(10):
      W[l][i] = prob[l] / prob_sum
  return W

@nb.jit
def M_Step(training_image_bin, IM, L, W):
  for i in range(10):
    w = 0
    for j in range(60000):
      w += W[i][j]
    L[i][0] = w / 60000
  
  for i in range(784):
    for j in range(10):
      w = 0
      wx = 0
      for k in range(60000):
        w += W[j][k]
        wx += training_image_bin[k][i] * W[j][k]
      if w == 0:
        w = 1
      IM[j][i] = wx / w
  return IM, L

@nb.jit
def decide_label(training_image_bin, training_label, IM, L):
  table = np.full((10, 10), 0)
  relation = np.full((10), -1)
  for i in range(60000):
    prob = np.zeros(shape=10)
    for j in range(10):
      tmp = 1
      for k in range(784):
        if training_image_bin[i][k] == 1:
          tmp *= IM[j][k]
        else:
          tmp *= (1-IM[j][k])
      tmp *= L[j][0]
      prob[j] = tmp
    table[training_label[i]][np.argmax(prob)] += 1
  for _ in range(10):
    max_num = -1
    max_indexi = 10
    max_indexj = 10
    for i in range(10):
      for j in range(10):
        if table[i][j] > max_num:
          max_num = table[i][j]
          max_indexi = i
          max_indexj = j
    # ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
    relation[max_indexi] = max_indexj
    for l in range(10):
      table[max_indexi][l] = -1 
      table[l][max_indexj] = -1
  return relation

@nb.jit 
def confusion_matrix_f(training_image_bin, training_label, IM, L, relation):
  confusion_matrix = np.zeros(shape=(10,2,2), dtype=np.int64)
  for j in range(60000):
    prob = np.zeros(shape=10, dtype=np.float64)
    for k in range(10):
      tmp = 0
      for l in range(784):
        if training_image_bin[j][l] == 1:
          tmp += IM[k][l]
        else:
          tmp += (1-IM[k][l])
      tmp += L[k][0]
      prob[k] = tmp
    predict = np.argmax(prob)
    for m in range (10):
      if relation[m] == predict:
        predict = m
        break
    for i in range(10):
      if training_label[j] == i:
        if predict == i:
          confusion_matrix[i][0][0] += 1
        else:
          confusion_matrix[i][0][1] += 1
      else:
        if predict == i:
          confusion_matrix[i][1][0] += 1
        else:
          confusion_matrix[i][1][1] += 1
  return confusion_matrix

if __name__ == "__main__":
  training_image_bin, training_label = read_files()
  # intial parameter 
  L = np.full((10, 1), 0.1) # select each area
  IM = np.random.rand(10, 28 * 28).astype(np.float64) #each coin chance of head
  old_IM = np.zeros((10, 28 * 28), dtype=np.float64) #last time each coin chance of head
  W = np.full((10, 60000), 0.1, dtype=np.float64) #likelihood of 60000 images

  iteration = 0
  while True:
    iteration += 1
    W = E_step(training_image_bin, IM, L, W)
    IM, L = M_Step(training_image_bin, IM, L, W)
    
    all_difference = 0
    for i in range(10):
      difference = 0
      for j in range(784):
        difference += abs(IM[i][j] - old_IM[i][j])
      all_difference += difference
    old_IM = IM
    #print
    for i in range(10):
      print('class {:d}:'.format(i), end = '')
      for j in range(784):
        if j % 28 == 0:
          print()
        if IM[i][j] > 0.5:
          print('1', end=' ')
        else:
          print('0', end=' ')
      print()
      print()
    print('No. of Iteration: {:d}, Difference: {:f}'.format(iteration, all_difference))
    print()
    print('------------------------------------------------------------')
    print()

    tmp = 0
    for i in range(10):
      if L[i][0] == 0:
        tmp = 1
        L = np.full((10, 1), 0.1)
        IM = np.random.rand(10, 28 * 28).astype(np.float64)
        W = np.full((10, 60000), 0.1, dtype=np.float64)
        old_IM = np.zeros((10, 28 * 28), dtype=np.float64)
        break
    
    if all_difference < 10 and tmp == 0 and ((L[np.argmax(L)] - L[np.argmin(L)]) < 1e-1 or iteration > 20):
      break

relation = decide_label(training_image_bin, training_label, IM, L)

for i in range(10):
  print('labeled class {:d}:'.format(i), end = '')
  for j in range(784):
    if j % 28 == 0:
      print()
    if IM[relation[i]][j] > 0.5:
      print('1', end=' ')
    else:
      print('0', end=' ')
  print()
  print()

confusion_matrix = confusion_matrix_f(training_image_bin, training_label, IM, L, relation)
error = 60000
for i in range(10):
    print('Confusion Matrix {:d}:'.format(i))
    print('            Predict number {:d}  Predict not number {:d}'.format(i, i))
    print('Is number {:d}         {:0>5d}               {:0>5d}'.format(i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
    print('Isn\'t number {:d}      {:0>5d}               {:0>5d}'.format(i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
    print()
    print('Sensitivity (Successfully predict number {:d}):     {:5f}'.format(i, confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])))
    print('Specificity (Successfully predict not number {:d}): {:5f}'.format(i, confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])))
    print()
    print('------------------------------------------------------------')
    print()
    error -= confusion_matrix[i][0][0]

print('Total iteration to converge: {:d}'.format(iteration))
print('Total error rate: {:5f}'.format(error/60000))
