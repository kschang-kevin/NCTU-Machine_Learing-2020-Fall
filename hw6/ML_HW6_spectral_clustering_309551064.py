from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

K = 2

def read_data(file_name):
    img = Image.open(file_name)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width*height, 3))

    position = []
    for i in range(100):
        for j in range(100):
            position.append([i, j])
    position = np.array(position)
    return position, pixel

def initial(method, data):
    if method == 'random_partition':
        classification = np.random.randint(0, K, size=10000)
        mu = np.zeros((K, K), dtype=np.float)
        count = np.zeros(K, dtype=np.float)
        for i in range(len(classification)):
            mu[classification[i]] += data[i]
            count[classification[i]] += 1
        for i in range(K):
            mu[i] /= count[i]
    if method == 'kmeans++':
        initial_center = np.random.randint(0, 10000, size=1)
        mu = np.zeros((K, K), dtype=np.float) 
        mu[0] = data[initial_center]
        for p in range(1, K):
            distance = np.zeros(10000, dtype=np.int)
            for i in range(len(data)):
                dis = np.zeros(p, dtype=np.int)
                for j in range(p):
                    tmp = 0
                    for k in range(len(data[0])):
                        tmp += (data[i][k] - mu[j][k]) ** 2
                    dis[j] = tmp
                distance[i] = min(dis)
            mu[p] = data[np.argmin(distance)]
        classification = np.zeros(10000, dtype=np.int)
        for i in range(10000):
            distance = np.zeros(K, dtype=np.float32)
            for j in range(K):
                for k in range(K):
                    distance[j] += abs(data[i][k] - mu[j][k]) ** 2
            classification[i] = np.argmin(distance)
    return mu, classification

def kernel(spatial, color):
    gamma_c = 1/(255*255)
    gamma_s = 1/(100*100)
    spatial_sq_dists = squareform(pdist(spatial, 'sqeuclidean'))
    spatial_rbf = np.exp(-gamma_s * spatial_sq_dists)
    color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
    color_rbf = np.exp(-gamma_c * color_sq_dists)
    kernel = spatial_rbf * color_rbf
    return kernel

def visualization(classification, iteration, file_name, method):
    img = Image.open(file_name)
    pixel = img.load()
    color = [(0, 0, 0), (125, 0, 0), (0, 255, 0), (0, 255, 255)]
    for i in range(100):
        for j in range(100):
            pixel[j, i] = color[classification[i * 100 + j]]
    img.save(file_name.split('.')[0] + str(K) + '_' + str(method) + '_' + str(iteration) + '.png')

def classify(data, mu):
    classification = np.zeros(10000, dtype=np.int)
    for i in range(10000):
        distance = np.zeros(K, dtype=np.float32)
        for j in range(K):
            for k in range(K):
                distance[j] += abs(data[i][k] - mu[j][k]) ** 2
        classification[i] = np.argmin(distance)
    return classification

def difference(classification, old_classification):
    diff = 0
    for i in range(len(classification)):
        diff += abs(classification[i] - old_classification[i])
    return diff

def update(data, mu, classification):
    new_mu = np.zeros(mu.shape, dtype=np.float32)
    count = np.zeros(K, dtype=np.int)
    for i in range(len(classification)):
        new_mu[classification[i]] += data[i]
        count[classification[i]] += 1
    for i in range(len(new_mu)):
        if count[i] == 0:
            count[i] = 1
        new_mu[i] = new_mu[i] / count[i]
    return new_mu

def draw(classification, data):
    color = [(0, 0, 0), (0.5, 0, 0), (0, 1, 0), (0, 1, 1)]
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], s=8, c=[color[classification[i]]])
    plt.show()

def spectral_clustering(file_name, data):
    initial_methods = ['random_partition', 'kmeans++']
    for method in initial_methods:
        mu, classification = initial(method, data)
        iteration = 0
        old_diff = 1e9
        visualization(classification, iteration, file_name, method)
        while iteration < 20:
            iteration += 1
            old_classification = classification
            classification = classify(data, mu)
            diff = difference(classification, old_classification)
            visualization(classification, iteration, file_name, method)
            if diff == old_diff:
                break
            old_diff = diff
            mu = update(data, mu, classification)
        draw(classification, data)

def normalized_cut(spatial, color):
    W = kernel(spatial, color)
    D = np.diag(np.sum(W, axis=1))
    D_inverse_sqrt = np.diag(np.power(np.diag(D), -0.5))
    L_sym = np.identity(len(spatial)) - D_inverse_sqrt @ W @ D_inverse_sqrt

    eigen_values, eigen_vectors = np.linalg.eig(L_sym)
    idx = np.argsort(eigen_values)[1: K+1]
    U = eigen_vectors[:, idx].real.astype(np.float32)

    T = np.zeros((U.shape[0], U.shape[1]))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            sum_tmp = 0
            for k in range(T.shape[1]):
                sum_tmp += U[i][k] ** 2
            T[i][j] = U[i][j] / (sum_tmp ** 0.5)
    
    return T

def ratio_cut(spatial, color):    
    W = kernel(spatial, color)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    eigen_values, eigen_vectors = np.linalg.eig(L)
    idx = np.argsort(eigen_values)[1: K+1]
    U = eigen_vectors[:, idx].real.astype(np.float32)

    return U
    
if __name__ == '__main__':
    spatial, color = read_data('image1.png')
    data = normalized_cut(spatial, color)
    spectral_clustering('image1.png', data)