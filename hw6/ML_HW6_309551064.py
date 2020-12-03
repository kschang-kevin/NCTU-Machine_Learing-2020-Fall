from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform

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

def initial():
    classification = np.random.randint(0, 2, size=10000)
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

def visualization(classification, iteration, file_name):
    img = Image.open(file_name)
    width, height = img.size
    pixel = img.load()
    color = [(0,0,0), (125, 0, 0), (0, 255, 0), (255, 255, 255)]
    for i in range(100):
        for j in range(100):
            pixel[j, i] = color[classification[i * 100 + j]]
    img.save(file_name + '_' + str(iteration) + '.png')

def calculate_third_term(kernel_data, classification):
    cluster_sum = np.zeros(2,dtype=np.int)
    kernel_sum = np.zeros(2,dtype=np.float)
    for i in range(classification.shape[0]):
        cluster_sum[classification[i]] += 1
    for cluster in range(2):
        for p in range(kernel_data.shape[0]):
            for q in range(kernel_data.shape[0]):
                if classification[p] == cluster and classification[q] == cluster:
                    kernel_sum[cluster] += kernel_data[p][q]
    for cluster in range(2):
        if cluster_sum[cluster] == 0:
            cluster_sum[cluster] = 1
        kernel_sum[cluster] /= (cluster_sum[cluster] ** 2)
    
    return kernel_sum

def calculate_second_term(kernel_data, classification, dataidx, cluster):
    cluster_sum = 0
    kernel_sum = 0
    for i in range(classification.shape[0]):
        if classification[i] == cluster:
            cluster_sum += 1
    if cluster_sum == 0:
        cluster_sum = 1
    for i in range(kernel_data.shape[0]):
        if classification[i] == cluster:
            kernel_sum += kernel_data[dataidx][i]

    return (-2) * kernel_sum / cluster_sum

def classify(kernel_data, classification):
    this_classification = np.zeros(10000, dtype=np.int)
    third_term = calculate_third_term(kernel_data, classification)
    for dataidx in range(10000):
        distance = np.zeros(2, dtype=np.float32)
        for cluster in range(2):
            distance[cluster] = calculate_second_term(kernel_data, classification, dataidx, cluster) + third_term[cluster]
        this_classification[dataidx] = np.argmin(distance)
    
    return this_classification

def difference(classification, old_classification):
    diff = 0
    for i in range(len(classification)):
        diff += abs(classification[i] - old_classification[i])
    return diff

def K_means(file_name, spatial, color):
    classification = initial()
    gram_matrix = kernel(spatial, color) 
    iteration = 0
    old_diff = 0
    visualization(classification, iteration, file_name)
    while iteration < 20:
        iteration += 1
        old_classification = classification
        classification = classify(gram_matrix, classification)
        diff = difference(classification, old_classification)
        visualization(classification, iteration, file_name)
        if diff == old_diff:
            break
        old_diff = diff 

if __name__ == '__main__':
    spatial, color = read_data('image1.png')
    K_means('image1.png', spatial, color)