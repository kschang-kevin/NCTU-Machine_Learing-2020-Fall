import os
import re
import cv2
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from PIL import Image

def read_input(dir):
    training_file = os.listdir(dir)
    data = []
    subject = []
    for file in training_file:
        file_name = dir + file
        number = int(file_name.split('subject')[1].split('.')[0])
        subject.append(number)
        img = Image.open(file_name)
        img = img.resize((60,60), Image.ANTIALIAS)
        width, height = img.size
        pixel = np.array(img.getdata()).reshape((60*60))
        data.append(pixel)
    return np.array(data), np.array(subject)

def PCA(data):
    covariance = np.cov(data.T)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:,idx][:,:25]
    return eigen_vectors

def kernelPCA(gram_matrix):
    N = gram_matrix.shape[0]
    one_n = np.ones((N, N)) / N
    K = gram_matrix - one_n.dot(gram_matrix) - gram_matrix.dot(one_n) + one_n.dot(gram_matrix).dot(one_n)
    eigen_values, eigen_vectors = np.linalg.eigh(K)
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:,idx][:,:25]
    return eigen_vectors

def visualization(data):
    random_idx = np.random.randint(len(data), size=10)
    for idx in random_idx:
        cv2.imwrite('./reconstruction_face/' + str(idx) + '.png', data[idx].reshape(60, 60))

def draweigenface(eigen_vectors):
    eigen_vectors = eigen_vectors.T
    for i in range(25):
        img = (eigen_vectors[i]-min(eigen_vectors[i]))/(max(eigen_vectors[i])-min(eigen_vectors[i])) * 255
        cv2.imwrite('./eigen_face/' + str(i) + '.png',img.reshape(60, 60))

def KNN(training_data, testing_data, training_subject):
    result = np.zeros(testing_data.shape[0])
    for i in range(testing_data.shape[0]):
        distance = np.zeros(training_data.shape[0])
        for j in range(training_data.shape[0]):
            distance[j] = np.sqrt(np.sum((testing_data[i] - training_data[j]) ** 2))
        result[i] = training_subject[np.argmin(distance)]
    return result

def checkperformance(testing_subject, predict):
    correct = 0
    for i in range(len(testing_subject)):
        if testing_subject[i] == predict[i]:
            correct += 1
    print(correct / len(testing_subject))

if __name__ == '__main__':
    # PCA
    training_data, training_subject = read_input('./Yale_Face_Database/Training/')
    eigen_vectors = PCA(training_data)
    lower_dimension_training_data = training_data.dot(eigen_vectors)
    reconstruct_data = lower_dimension_training_data.dot(eigen_vectors.T)
    visualization(reconstruct_data)
    draweigenface(eigen_vectors)

    testing_data, testing_subject = read_input('./Yale_Face_Database/Testing/')
    lower_dimension_testing_data = testing_data.dot(eigen_vectors)

    predict = KNN(lower_dimension_training_data, lower_dimension_testing_data, training_subject)
    checkperformance(testing_subject, predict)
    
    # kernel PCA
    training_gram_matrix = training_data.dot(training_data.T)
    # training_gram_matrix = np.power(training_data.dot(training_data.T), 2)
    eigen_vectors = kernelPCA(training_gram_matrix)
    lower_dimension_training_data = training_gram_matrix.dot(eigen_vectors)

    testing_gram_matrix = testing_data.dot(training_data.T)
    # testing_gram_matrix = np.power(testing_data.dot(training_data.T), 2)
    lower_dimension_testing_data = testing_gram_matrix.dot(eigen_vectors)
    
    predict = KNN(lower_dimension_training_data, lower_dimension_testing_data, training_subject)
    checkperformance(testing_subject, predict)
