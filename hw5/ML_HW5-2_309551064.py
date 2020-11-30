import numpy as np
from scipy.spatial.distance import cdist
from libsvm.svmutil import *

def read_file():
    X_train = np.loadtxt('./X_train.csv', dtype=np.float, delimiter=',')
    Y_train = np.loadtxt('./Y_train.csv', dtype=np.float, delimiter=',')
    X_test = np.loadtxt('./X_test.csv', dtype=np.float, delimiter=',')
    Y_test = np.loadtxt('./Y_test.csv', dtype=np.float, delimiter=',')
    return X_train, Y_train, X_test, Y_test

def part1(X_train, Y_train, X_test, Y_test):
    kernel = ['linear', 'polynomial', 'RBF']
    for i in range(3):
        print('Kernel Function: {}'.format(kernel[i]))
        parameter = '-q -t ' + str(i)
        model = svm_train(Y_train , X_train, parameter)
        svm_predict(Y_test, X_test, model)

def part2(X_train, Y_train, X_test, Y_test):
    cost = ['1', '2', '3']
    gamma = ['0.25', '0.5']
    degree = ['2', '3', '4']
    coef0 = ['0', '1', '2']
    best_parameter = ''
    best_accuracy = 0
    for i in range(3):
        parameter = '-v 10 -q -t 0 -c ' + cost[i]
        accuracy = svm_train(Y_train, X_train, parameter)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameter = parameter

    for i in range(3):
        for j in range(2):
            for k in range(3):
                for l in range(3):
                    parameter = '-v 10 -q -t 1 -c ' + cost[i] + ' -g ' + gamma[j] + ' -d ' + degree[k] + ' -r ' + coef0[l]
                    accuracy = svm_train(Y_train, X_train, parameter)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameter = parameter
    
    for i in range(3):
        for j in range(2):
            parameter = '-v 10 -q -t 2 -c ' + cost[i] + ' -g ' + gamma[j]
            accuracy = svm_train(Y_train, X_train, parameter)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameter = parameter
    
    print('Best Accuracy: {}'.format(best_accuracy))
    print('Corresponding Parameter: {}'.format(best_parameter))

def part3(X_train, Y_train, X_test, Y_test):
    negative_gamma = -1 / 784
    train_linear_kernel = X_train.dot(X_train.transpose())
    # train_poly_kernel = np.power(X_train.dot(X_train.transpose()), 3)
    train_rbf_kernel = np.exp(negative_gamma * cdist(X_train, X_train, 'sqeuclidean'))
    X_train_kernel = np.concatenate((np.arange(1, 5001).reshape((5000, 1)), train_linear_kernel + train_rbf_kernel), axis=1)
    # X_train_kernel = np.concatenate((np.arange(1, 5001).reshape((5000, 1)), train_linear_kernel + train_poly_kernel + train_rbf_kernel), axis=1)
    
    test_linear_kernel = X_test.dot(X_train.transpose())
    # test_poly_kernel = np.power(X_test.dot(X_train.transpose()), 3)
    test_rbf_kernel = np.exp(negative_gamma * cdist(X_test, X_train, 'sqeuclidean'))
    X_test_kernel = np.concatenate((np.arange(1, 2501).reshape((2500, 1)), test_linear_kernel + test_rbf_kernel), axis=1)
    # X_test_kernel = np.concatenate((np.arange(1, 2501).reshape((2500, 1)), test_linear_kernel + test_poly_kernel + test_rbf_kernel), axis=1)
    
    
    prob  = svm_problem(Y_train, X_train_kernel, isKernel=True)
    param = svm_parameter('-q -t 4')
    model = svm_train(prob, param)
    svm_predict(Y_test, X_test_kernel, model)
    
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = read_file()
    print('Part 1:')
    part1(X_train, Y_train, X_test, Y_test)
    print('===================================================================')
    print('Part 2:')
    part2(X_train, Y_train, X_test, Y_test)
    print('===================================================================')
    print('Part 3:')
    part3(X_train, Y_train, X_test, Y_test)