import cv2
import numpy as np
import matplotlib.pyplot as plt

# important note: all inputs must be numpy arrays dtype float32
# note: since pixels distanced from center of an image don't offer much in object detection, the padding process is not
# implemented in filtering

# 2.1 simplification of the image
############################################################################################################

# Since we can get the best results from a symmetric kernel, these functions just creates kernels with odd values as
# kernel size
def create_gaussian_kernel(kernel_size, variance):
    gaussian_kernel = np.zeros((kernel_size, kernel_size), 'float32')
    loop_start = -(kernel_size//2)
    loop_stop = kernel_size // 2 + 1
    for i in range(loop_start, loop_stop):
        for j in range(loop_start, loop_stop):
            kernel_co_i = i + (loop_start-1)
            kernel_co_j = j + (loop_start-1)
            gaussian_kernel[kernel_co_i, kernel_co_j] = (1 / (2 * np.pi * np.power(variance, 2))) * \
                                    np.exp(-((np.power(i, 2) + np.power(j, 2)) / (2 * np.power(variance, 2))))

    print("gaussian kernel created")
    return gaussian_kernel

################

def create_emboss_kernel(kernel_size):
    emboss_kernel = np.ones((kernel_size, kernel_size), 'float32')
    loop_start = -(kernel_size // 2)
    loop_stop = kernel_size // 2 + 1
    for i in range(loop_start, loop_stop):
        for j in range(loop_start, loop_stop):
            kernel_co_i = i + (loop_start - 1)
            kernel_co_j = j + (loop_start - 1)
            if i == -j:
                emboss_kernel[kernel_co_i, kernel_co_j] = 0
            elif i + j > 0:
                emboss_kernel[kernel_co_i, kernel_co_j] = -emboss_kernel[kernel_co_i, kernel_co_j]

    print("emboss kernel created")
    return emboss_kernel

################

# Using elementwise multiplication to merge kernels
# This function gets an array of kernels, kernels must be of the same size, and results in a merged kernel(kernels must)
# be linear)
def merge_kernels(kernel_size, kernels):
    merged_kernel = np.zeros((kernel_size, kernel_size), 'float32')
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            new_value = 1
            for kernel in kernels:
                new_value *= kernel[i, j]

            merged_kernel[i, j] = new_value

    print("kernels merged")
    return merged_kernel

################

def filter_image(img, kernel):
    print("filtering image, please be patient...")

    img = np.array(img, 'float32')
    kernel = np.array(kernel, 'float32')
    new_img = np.zeros(img.shape, 'int32')
    kernel_loop_start = -(kernel.shape[0] // 2)
    kernel_loop_stop = kernel.shape[1] // 2 + 1

    for i in range(3, img.shape[0]-3):
        for j in range(3, img.shape[1]-3):
            new_value_b = 0
            new_value_g = 0
            new_value_r = 0
            f = 0
            for k in range(kernel_loop_start, kernel_loop_stop):
                for l in range(kernel_loop_start, kernel_loop_stop):
                    f += 1
                    new_value_b += (img[i + k, j + l, 0] * kernel[k+1, l+1])
                    new_value_g += (img[i + k, j + l, 1] * kernel[k+1, l+1])
                    new_value_r += (img[i + k, j + l, 2] * kernel[k+1, l+1])

            # print(new_value_b)
            # new_value_b /= kernel.shape[0] * kernel.shape[1]
            # new_value_g /= kernel.shape[0] * kernel.shape[1]
            # new_value_r /= kernel.shape[0] * kernel.shape[1]
            # print(new_value_b)
            # print(new_value_g)
            # print(new_value_r)
            new_img[i, j, 0] = new_value_b//1
            new_img[i, j, 1] = new_value_g//1
            new_img[i, j, 2] = new_value_r//1
            # print(new_img[10,10])

    print("image filtered successfully")
    return new_img

############################################################################################################



# 2.2 Analysis of the cluster of points
############################################################################################################

# dividing image to 100 clusters
def divide_image_to_clusters(img):
    img_height, img_width = np.shape(img)[0], np.shape(img)[1]
    cluster_width = img_width // 10
    cluster_height = img_height // 10
    clusters = np.zeros([100, cluster_width, cluster_height, 3], 'float32')
    s = 0
    for i in range(0, 10):
        for j in range(0, 10):
            new_cluster = img[i * cluster_height: (i+1) * cluster_height, j * cluster_width: (j+1) * cluster_width]
            new_cluster = cv2.normalize(new_cluster, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            clusters[s] = new_cluster
            s += 1

    return clusters

################

# determining the quality of a cluster
def determining_quality_of_cluster(cluster, fuzzy_decision):
    number_of_cluster_elements = cluster.shape[0] * cluster.shape[1]
    # first_part_of_eq = np.power(number_of_cluster_elements, -1) * fuzzy_decision
    first_part_of_eq = 1 / number_of_cluster_elements * fuzzy_decision
    result_of_second_part_of_eq = 0
    for i in range(1, number_of_cluster_elements):
        sum = 0
        for j in range(1, number_of_cluster_elements):
            sum += j
        result_of_second_part_of_eq += i / number_of_cluster_elements * sum

    final_value = first_part_of_eq * result_of_second_part_of_eq
    return final_value


################

# in the following functions I create a fuzzy controller which has to be experimentally tested to modify the values used
# in it's trapezoidal membership function

# this function processes the linguistic descriptions and results in a tuple with 4 values which has to be used later
# P.S also known as "rule output level"
def rule_output_level_processing(cluster_size, clusterr_big, cluster_small, fuzzy_decision_for_1):
    u_1 = np.power(cluster_size / fuzzy_decision_for_1, -(1 / 2))

    u_2 = 0
    if cluster_size > fuzzy_decision_for_1:
        u_2 = cluster_size / fuzzy_decision_for_1
    else:
        u_2 = fuzzy_decision_for_1 / cluster_size

    u_3 = np.power(cluster_size * fuzzy_decision_for_1, -1)

    u_4 = np.power(fuzzy_decision_for_1 / cluster_size, (1 / 2))

    final_result = np.array([u_1, u_2, u_3, u_4])

    return final_result

################

# calculate Hue–Saturation–Lightness component
# P.S since the paper uses a specific kind of convertion to hsv, we can'y use openCV's predefined function
def calculate_hsv(cluster):
    hsl_cluster = np.zeros(cluster.shape)
    for i in range(0, cluster.shape[0]):
        for j in range(0, cluster.shape[1]):
            b, g, r = cluster[i, j, 0], cluster[i, j, 1], cluster[i, j, 2]
            max = np.amax(cluster[i, j])
            min = np.amin(cluster[i, j])

            b = (min + max) / 2

            h = 0
            if max == b:
                h = 60 * (((r - g) / (max - min)) + 4)
            elif max == r:
                h = 60 * (((g - b) / (max - min)) % 6)
            elif max == g:
                h = 60 * (((b - r) / (max - min)) + 2)

            s = 0
            if min != max:
                s = (max - min) / (1 - np.abs(max + min - 1))

            hsl_cluster[i, j] = [h, s, b]
            # if b != 0:
            #     print([h, s, b])
    print(hsl_cluster)
    return hsl_cluster

################

# calculate quality of cluster for membership function
def quality_for_membership_func(cluster):
    # cluster = calculate_hsv(cluster)
    cluster = cv2.cvtColor(cluster, cv2.COLOR_BGR2HSV)
    sum = 0
    for i in range(0, cluster.shape[0]):
        for j in range(0, cluster.shape[1]):
            sum += cluster[i, j, 0]

    final_value = sum / (cluster.shape[0] * cluster.shape[1])
    return final_value

################

# cluster membership function
# gets 4 values as input(a <= b <= c <= d) and response of quality_for_membership_func quality
def calculate_membership_function(quality, parameters):
    a, b, c, d = parameters[0], parameters[1], parameters[2], parameters[3]
    val_1 = (quality - a) / (b - a)
    val_2 = (d - quality) / (d - c)
    degree_of_membership = np.min([val_1, val_2, 1])
    degree_of_membership = np.max([degree_of_membership, 0])

    return degree_of_membership

################

# calculating the weight of Rule output level, also known as Rule firing strength
# this function also gets 4 parameters to pass to preceding one
def calculate_weight(cluster, parameters, average_fuzzy_decision):
    quality = quality_for_membership_func(cluster)
    degree_of_membership = calculate_membership_function(quality, parameters)
    weight = np.min([degree_of_membership, average_fuzzy_decision])
    # return average_fuzzy_decision
    return weight

################

# calculating all 4 weights used in fuzzy controller(we need 2 * 4 parameters to calculate the degree of membership to)
# both small and big clusters. We also need the average of image significance of 1 like some other functions)
# parameters must be in shape of [2, 4], this function returns a tuple of all 4 weights
def calculate_all_4_weights(cluster, parameters, fuzzy_decision):
    weight_big = calculate_weight(cluster, parameters[0], fuzzy_decision)
    weight_small = calculate_weight(cluster, parameters[1], fuzzy_decision)

    result = np.array([weight_big, weight_big, weight_small, weight_small])
    return result

################

# calculating significance of a cluster, input 4 Rule output levels(outputs)  and 4 Rule firing strengths(weights)
# result is the weighted sum of all
def calculate_significance(weights, outputs):
    val_1 = 0
    for i in range(0, 4):
        val_1 += weights[i] * outputs[i]

    val_2 = 0
    for i in range(0, 4):
        val_2 += weights[i]

    significance = val_1 / val_2

    return significance

################

# calculating the domain of membership function, input the same parameters for just a, b, c, returns two values, e_left
# and e_right of each clusters later to be compared with each other, input cluster must be in HSL format
def calculate_domain(cluster, parameters):
    results = np.zeros([cluster.shape[0], cluster.shape[1]])
    a, b, c = parameters[0], parameters[1], parameters[2]
    sum_of_all_parameters = a + b + c

    for i in range(0, cluster.shape[0]):
        for j in range(0, cluster.shape[1]):
            value = (a * cluster[i, j, 2]) + (b * cluster[i, j, 0]) + (c * cluster[i, j, 1])
            results[i, j] = value / sum_of_all_parameters

    e_left, e_right = np.min(results), np.max(results)

    return e_left, e_right

################

# calculate domain of global trapezoidal membership function, gets an array of all cluster domains
def calculate_global_domain(local_domains):
    e_left, e_right = np.min(local_domains), np.max(local_domains)
    return e_left, e_right

################

# calculate if the clusters are good enough to be processed in CNN,gets two arrays one of significance of cluster and
# one of corresponding clusters and results in a new array of clusters
def final_decision_of_clusters(clusters, fuzzy_decision_on_clusters):
    label_array = np.zeros((clusters.shape[0]))
    best = max(fuzzy_decision_on_clusters)

    mid = np.average(fuzzy_decision_on_clusters)

    count = 0
    for i in range(0, clusters.shape[0]):
        if fuzzy_decision_on_clusters[i] >= mid:
            label_array[i] = 1
            count += 1

    final_clusters = np.zeros((count, clusters.shape[1], clusters.shape[2], clusters.shape[3]))
    k = 0
    for i in range(0, label_array.shape[0]):
        if label_array[i] == 1:
            final_clusters[k] = clusters[i]
            k += 1

    return final_clusters

################

def connect_clusters(clusters):
    shape = np.sqrt(clusters.shape[0]) + 1
    shape = shape.astype(int)
    img = np.zeros((shape * clusters.shape[1], shape * clusters.shape[2], 3))
    count = 0
    for i in range(0, shape):
        for j in range(0, shape):
            if count < clusters.shape[0]:
                img[i, j] = clusters[count, i, j]
                count += 1
            else:
                return img


