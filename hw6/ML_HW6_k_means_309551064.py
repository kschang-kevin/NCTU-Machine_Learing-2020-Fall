from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform
import numba as nb
K = 4

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

def initial(method):
    if method == 'random_partition':
        classification = np.random.randint(0, K, size=10000)
    if method == 'Forgy':
        classification = np.zeros(10000, dtype=np.int)
        center = np.random.randint(0, 10000, size=K)
        for i in range(100):
            for j in range(100):
                near = 1e9
                for k in range(len(center)):
                    if (((i - (center[k] / 100)) ** 2) + (j - (center[k] % 100)) ** 2) < near:
                        classification[100 * i + j] = k
                        near = (((i - (center[k] / 100)) ** 2) + (j - (center[k] % 100)) ** 2)    
    return classification

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
    color = [(0,0,0), (125, 0, 0), (0, 255, 0), (255, 255, 255)]
    for i in range(100):
        for j in range(100):
            pixel[j, i] = color[classification[i * 100 + j]]
    img.save(file_name.split('.')[0] + '_' + str(K) + '_' + str(method) + '_' + str(iteration) + '.png')

@nb.jit
def classify(gram_matrix, classification):
    new_classification = np.zeros(10000, dtype=np.int)
    third_term = np.zeros(K, dtype=np.int)
    cluster_num = np.zeros(K, dtype=np.int)
    for i in range(len(classification)):
        cluster_num[classification[i]] += 1
    for i in range(10000):
        for j in range(10000):
            if classification[i] == classification[j]:
                third_term[classification[i]] += gram_matrix[i][j]
    for i in range(len(third_term)):
        third_term[i] /= cluster_num[i] ** 2
    
    for i in range(10000):
        distance = np.zeros(K, dtype=np.float32)
        for j in range(K):
            second_term = 0
            count = 0
            for k in range(10000):
                if classification[k] == j:
                    second_term += gram_matrix[i][k]
                    count += 1
            second_term = second_term * 2 / count 
            distance[j] = gram_matrix[i][i] - second_term + third_term[j]
        new_classification[i] = np.argmin(distance)
    
    return new_classification

def difference(classification, old_classification):
    diff = 0
    for i in range(len(classification)):
        diff += abs(classification[i] - old_classification[i])
    return diff

def K_means(file_name, spatial, color):
    gram_matrix = kernel(spatial, color)
    initial_methods = ['random_partition', 'Forgy']
    for method in initial_methods:
        classification = initial(method)         
        iteration = 0
        old_diff = 0
        visualization(classification, iteration, file_name, method)
        while iteration < 20:
            iteration += 1
            old_classification = classification
            classification = classify(gram_matrix, classification)
            diff = difference(classification, old_classification)
            visualization(classification, iteration, file_name, method)
            if diff == old_diff:
                break
            old_diff = diff 

if __name__ == '__main__':
    spatial, color = read_data('image2.png')
    K_means('image2.png', spatial, color)