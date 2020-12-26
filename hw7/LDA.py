import os
import re
import cv2
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

gamma = 1e-3
SUBJECT = 11


def read_input(dir):
    training_file = os.listdir(dir)
    data = []
    subject = []
    for file in training_file:
        file_name = dir + file
        number = int(file_name.split('subject')[1].split('.')[0])
        subject.append(number)
        img = cv2.imread(file_name, 0)
        img = cv2.resize(img, (60, 60))
        data.append(img.reshape(60 * 60))
    return np.array(data), np.array(subject)

def computeWithinClass(data, subject):
    class_mean = np.zeros((15, data.shape[1]))
    for i in range(len(data)):
        class_mean[subject[i]-1] += data[i]
    class_mean = class_mean / 9

    within_class = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        tmp = (data[i] - class_mean[subject[i] - 1]).reshape(data.shape[1], 1)
        within_class += tmp.dot(tmp.T)    
    return within_class

def computeBetweenClass(data, subject):
    class_mean = np.zeros((15, data.shape[1]))
    for i in range(len(data)):
        class_mean[subject[i]-1] += data[i]
    class_mean = class_mean / 11
    all_mean = (np.sum(data, axis=0) / len(data)).reshape(-1, 1)

    between_class = np.zeros((data.shape[1], data.shape[1]))
    for i in range(15):
        tmp = 11 *  (class_mean[i] - all_mean[i]).reshape(data.shape[1], 1)
        between_class += tmp.dot(tmp.T)       
    return between_class 

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

def kernelLDA(data, target, method):
    gram_matrix = None
    if method == 'rbf':
        sq_dists = squareform(pdist(data), 'sqeuclidean')
        gram_matrix = np.exp(-gamma * sq_dists)
    elif method == 'linear':
        gram_matrix = np.matmul(data, data.T)

    M = np.zeros([data.shape[0], data.shape[0]])
    for i in range(CLASS):
        classM = gram_matrix[np.where(target == i+1)[0]].copy()
        classM = np.sum(classM, axis=0).reshape(-1, 1) / SUBJECT
        allM = gram_matrix[np.where(target == i+1)[0]].copy()
        allM = np.sum(allM, axis=0).reshape(-1, 1) / data.shape[0]
        dist = np.subtract(classM, allM)
        multiplydist = SUBJECT * np.matmul(dist, dist.T)
        M += multiplydist

    N = np.zeros([data.shape[0], data.shape[0]])
    I_minus_one = np.identity(SUBJECT) - (SUBJECT * np.ones((SUBJECT, SUBJECT)))
    for i in range(CLASS):
        Kj = gram_matrix[np.where(target == i+1)[0]].copy()
        multiply = np.matmul(Kj.T, np.matmul(I_minus_one, Kj))
        N += multiply

    eigenvectors = compute_eigen(np.matmul(np.linalg.pinv(N), M))
    lower_dimension_data = np.matmul(gram_matrix, eigenvectors)
    return lower_dimension_data

if __name__ == '__main__':
    training_data, training_subject = read_input('./Yale_Face_Database/Training/')
    # within_class_scatter = computeWithinClass(training_data, training_subject)
    # between_class_scatter = computeBetweenClass(training_data, training_subject)
    
    # eigen_values, eigen_vectors = np.linalg.eigh(np.linalg.pinv(within_class_scatter).dot(between_class_scatter))
    # idx = eigen_values.argsort()[::-1]
    # eigen_vectors = eigen_vectors[:,idx][:,:25]
    # lower_dimension_data = training_data.dot(eigen_vectors)
    # reconstruct_data = lower_dimension_data.dot(eigen_vectors.T)
    # visualization(reconstruct_data)
    # draweigenface(eigen_vectors)

    testing_data, testing_subject = read_input('./Yale_Face_Database/Testing/')
    data = np.concatenate((training_data, testing_data), axis=0)
    subject = np.concatenate((training_subject, testing_subject), axis=0)
    within_class_scatter = computeWithinClass(data, subject)
    between_class_scatter = computeBetweenClass(data, subject)
    
    eigen_values, eigen_vectors = np.linalg.eigh(np.linalg.pinv(within_class_scatter).dot(between_class_scatter))
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:,idx][:,:25]
    lower_dimension_data = data.dot(eigen_vectors)
    lower_dimension_testing_data = lower_dimension_data[training_data.shape[0]:].copy()
    lower_dimension_training_data = lower_dimension_data[:training_data.shape[0]].copy()

    predict = KNN(lower_dimension_training_data, lower_dimension_testing_data, training_subject)
    checkperformance(testing_subject, predict)


    # print("=======================================================================")

    # #Kernel LDA
    # dirtrain = './Training/'
    # dirtest = './Testing/'
    # storedir = './LDA_result/'
    # method = 'linear'
    # data, target, totalfile = read_input(dirtrain)      
    # datatest, targettest, totalfiletest = read_input(dirtest)
    # data = np.concatenate((data, datatest), axis=0)                 #data : 165 x 3600, 
    # target = np.concatenate((target, targettest), axis=0)           #target : 165 x 1
    # lower_dimension_data = kernelLDA(data, target, method)
    # lower_dimension_data_train = lower_dimension_data[:totalfile.shape[0]].copy()
    # lower_dimension_data_test = lower_dimension_data[totalfile.shape[0]:].copy()
    # targettrain = target[:totalfile.shape[0]].copy()
    # targettest = target[totalfile.shape[0]:].copy()
    # predict = KNN(lower_dimension_data_train, lower_dimension_data_test, targettrain)
    # checkperformance(targettest, predict)