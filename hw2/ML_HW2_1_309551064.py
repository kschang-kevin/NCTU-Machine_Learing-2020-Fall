import gzip
import struct
import sys
import math

training_image_file = sys.argv[1]
training_label_file = sys.argv[2]
testing_image_file = sys.argv[3]
testing_label_file = sys.argv[4]
toggle = int(sys.argv[5])
#read training image
with gzip.open(training_image_file, 'rb') as f:
  file_content = f.read()
file_content = file_content[16:]
training_image = []
for i in range(60000):
  training_image.append(file_content[784*i:784*(i+1)])
#read training label
with gzip.open(training_label_file, 'rb') as f:
  file_content = f.read()
file_content = file_content[8:]
training_label = []
for i in range(60000):
  training_label.append(file_content[i:i+1])
for i in range(len(training_label)):
  training_label[i] = struct.unpack('>b',training_label[i])[0]
#read testing image
with gzip.open(testing_image_file, 'rb') as f:
  file_content = f.read()
file_content = file_content[16:]
testing_image = []
for i in range(10000):
  testing_image.append(file_content[784*i:784*(i+1)])
#read testing label
with gzip.open(testing_label_file, 'rb') as f:
  file_content = f.read()
file_content = file_content[8:]
testing_label = []
for i in range(10000):
  testing_label.append(file_content[i:i+1])
for i in range(len(testing_label)):
  testing_label[i] = struct.unpack('>b',testing_label[i])[0]
if toggle == 0:
  #preprocess
  image_bin = [[[0 for _ in range(32)] for _ in range(784)] for _ in range(10)] 
  for i in range(len(training_image)):
    for j in range(784):
      image_bin[training_label[i]][j][int((training_image[i][j]) / 8)] += 1

  image_label = [0 for _ in range(10)]
  for i in range(len(training_label)):
    image_label[training_label[i]] += 1

  #posterior
  error = 0
  for i in range(len(testing_image)):
    posterior = [0 for _ in range(10)]
    for j in range(len(testing_image[i])):
      for k in range(10):
        posterior[k] += math.log((image_bin[k][j][int(testing_image[i][j] / 8)] + 1) / image_label[k])
    for k in range(10):
      posterior[k] += math.log(image_label[k] / 60000) 
    sum_ = 0
    for q in range(10):
      sum_ += posterior[q]
    for q in range(10):
      posterior[q] = posterior[q] / sum_
    print('Posterior (in log scale):')
    for q in range(10):
      print(str(q) + ': ' + str(posterior[q]))
    if posterior.index(min(posterior)) != testing_label[i]:
      error += 1
    print('Prediction: ' + str(posterior.index(min(posterior))) + ', Ans: ', str(testing_label[i]))
    print()

  #imagination of number
  print('Imagination of numbers in Bayesian classifier:')
  for i in range(10):
    print(str(i) + ':', end = '')
    for j in range(784):
      tmp = 0
      if j % 28 == 0:
        print()
      else:
        print(' ', end = '')
      count = 0
      for k in range(len(training_image)):
        if training_label[k] == i:
          count += 1
          tmp += training_image[k][j]
      if round(tmp / count) < 128:
        print('0', end = '')
      else:
        print('1', end = '')
    print()
    print()
  print('Error rate: ', error/10000)        

if toggle == 1:
  #preprocess
  image_gaussian = [[[0 for _ in range(2)] for _ in range(784)] for _ in range(10)]
  for i in range(10):
    total_sum = [0 for _ in range(784)]
    count = 0
    for j in range(len(training_image)):
      if training_label[j] == i:
        count += 1
        for k in range(784):
          total_sum[k] += training_image[j][k]
    for j in range(784):
      image_gaussian[i][j][0] = total_sum[j] / count
    for j in range(784):
      variance = 0
      for k in range(len(training_image)):
        if training_label[k] == i:
          variance += (training_image[k][j] - image_gaussian[i][j][0]) ** 2
      image_gaussian[i][j][1] = (variance + 1) / count
  
  image_label = [0 for _ in range(10)]
  for i in range(len(training_label)):
    image_label[training_label[i]] += 1
  #MLE
  error = 0
  for i in range(len(testing_image)):
    MLE = [0 for _ in range(10)]
    for j in range(len(testing_image[i])):
      for k in range(10):
        sigmasqrt2pi = math.sqrt(image_gaussian[k][j][1]) * math.sqrt((2 * math.pi))
        expo =  ((testing_image[i][j] - image_gaussian[k][j][0]) ** 2) / (2 * image_gaussian[k][j][1])
        MLE[k] -= (math.log(sigmasqrt2pi) + expo)
        # MLE[k] -= ((((testing_image[i][j] - image_gaussian[k][j][0]) ** 2) / (2 * image_gaussian[k][j][1])) + math.log((image_gaussian[k][j][1] ** (1 / 2)) * ((2 * math.pi) ** (1 / 2))))
    for k in range(10):
      MLE[k] += math.log(image_label[k] / 60000)
    sum_ = 0
    for q in range(10):
      sum_ += MLE[q]
    for q in range(10):
      MLE[q] = MLE[q] / sum_
    print('Posterior (in log scale):')
    for q in range(10):
      print(str(q) + ': ' + str(MLE[q]))
    if MLE.index(min(MLE)) != testing_label[i]:
      error += 1
    print('Prediction: ' + str(MLE.index(min(MLE))) + ', Ans: ', str(testing_label[i]))
    print()
  #imagination of number
  print('Imagination of numbers in Bayesian classifier:')
  for i in range(10):
    print(str(i) + ':', end = '')
    for j in range(784):
      tmp = 0
      if j % 28 == 0:
        print()
      else:
        print(' ', end = '')
      count = 0
      if image_gaussian[i][j][0] < 128:
        print('0', end = '')
      else:
        print('1', end = '')
    print()
    print()
  print('Error rate: ', error/10000)



      
        

    



