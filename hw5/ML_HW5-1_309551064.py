import numpy as np
import math
from scipy.spatial.distance import cdist 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_file():
    data = np.loadtxt('input.data')
    X = []
    Y = []
    for i in range(len(data)):
        X.append([data[i][0]])
        Y.append([data[i][1]])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def kernel(x1, x2, parameters):
    overall_variance = parameters[0]
    length_scale = parameters[1]
    scale_mixture = parameters[2]
    return (overall_variance ** 2) * (1 + ((cdist(x1, x2, 'sqeuclidean') ** 2) / (2 * scale_mixture * (length_scale ** 2)))) ** (-1 *  scale_mixture)

def gaussian_process_training(x, parameters):
    C = kernel(x, x, parameters) + (1 / 5) * np.identity(x.shape[0])
    return C

def gaussian_process_testing(training_x, training_y, C, parameters):
    testing_x = np.array([np.arange(-60, 60, 0.1)]).transpose()
    testing_kernel = kernel(testing_x, testing_x, parameters) + (1 / 5)
    training_testing_kernel = kernel(training_x, testing_x, parameters)
    mu = training_testing_kernel.transpose().dot(np.linalg.inv(C)).dot(training_y)
    sigma_2 = testing_kernel - (training_testing_kernel.transpose().dot(np.linalg.inv(C)).dot(training_testing_kernel))
    return mu, np.sqrt(sigma_2)

def negative_marginal_log_likelihood(parameters, training_x, training_y):
    k = kernel(training_x, training_x, parameters)
    negative_log_likelihood = -(-0.5 * np.log(np.linalg.det(k)) - 0.5 * training_y.transpose().dot(np.linalg.inv(k)).dot(training_y) - 0.5 * len(training_x) * np.log(2*np.pi))
    return negative_log_likelihood[0][0]

def draw(x, y, mu, sigma, optimized_mu, optimized_sigma):
    gs = gridspec.GridSpec(9, 9)
    imx = np.arange(-60, 60, 0.1)
    ax1 = plt.subplot(gs[:4, :9])
    ax1.set_title('Basic')
    ax1.set_xlim([-60, 60])
    for i in range(len(x)):
        ax1.plot(x[i], y[i], 'ro')
    ax1.plot(imx, mu, 'k')
    ax1.fill_between(imx, (mu[:, 0] - 2 * np.diag(sigma)),  (mu[:, 0] + 2 * np.diag(sigma)))
        
    ax2 = plt.subplot(gs[5:9, :9])
    ax2.set_title('Optimized')
    ax2.set_xlim([-60, 60])
    for i in range(len(x)):
        ax2.plot(x[i], y[i], 'ro')
    ax2.plot(imx, optimized_mu, 'k')
    ax2.fill_between(imx, (optimized_mu[:, 0] - 2 * np.diag(optimized_sigma)),  (optimized_mu[:, 0] + 2 * np.diag(optimized_sigma)))

    plt.show()


if __name__ == '__main__': 
    training_x, training_y = read_file()
    parameters = [1, 1, 1]
    C = gaussian_process_training(training_x, parameters)
    mu, sigma = gaussian_process_testing(training_x, training_y, C, parameters)
    
    optimized_parameters = minimize(fun=negative_marginal_log_likelihood, x0=parameters, args=(training_x, training_y))
    print(optimized_parameters.x)
    C = gaussian_process_training(training_x, optimized_parameters.x)
    optimized_mu, optimized_sigma = gaussian_process_testing(training_x, training_y, C, optimized_parameters.x)
    
    draw(training_x, training_y, mu, sigma, optimized_mu, optimized_sigma)