import cv2
import numpy as np
import matplotlib.pyplot as plt
import pre_processing_funcs as ppf
# Tensorflow only used for mnist
import tensorflow as tf
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

gaussian = ppf.create_gaussian_kernel(3, 1)
emboss = ppf.create_emboss_kernel(3)

kernel = ppf.merge_kernels(3, (gaussian, emboss))


img = cv2.imread("Lenna.png")
# img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# filtered_image = ppf.filter_image(img, kernel)
filtered_image = cv2.filter2D(img, -1, kernel)
# cv2.imshow('t', filtered_image)
# cv2.waitKey()

clusters = ppf.divide_image_to_clusters(filtered_image)

quality_decision_for_1 = np.zeros(100)
for i in range(0, 100):
    quality_decision_for_1[i] = ppf.determining_quality_of_cluster(clusters[i], 1)
average_for_decision_for_1 = np.average(quality_decision_for_1)


weights = np.zeros((100, 4))
parameters = np.array([[0, 500, 1000, 1500], [20, 700, 1200, 1700]])
for i in range(0, 100):
    weights[i] = ppf.calculate_all_4_weights(clusters[i], parameters, quality_decision_for_1[i])

rule_outputs = np.zeros((100, 4))
for i in range(0, 100):
    rule_outputs[i] = ppf.rule_output_level_processing(clusters[i].shape[0]*clusters[i].shape[1], 2000, 500, quality_decision_for_1[i])

significances = np.zeros((100))
for i in range(0, 100):
    significances[i] = ppf.calculate_significance(weights[i], rule_outputs[i])

final_clusters = ppf.final_decision_of_clusters(clusters, significances)

final_image = ppf.connect_clusters(final_clusters)

# print(final_clusters)
# print("::::::::::::::::::::::::::::::::::::::::::::::::")
# print(reshaped_clusters)

train_clusters = np.array([final_image], 'int')
train_labels = np.array([1], 'int')
print(train_clusters.shape)
print(train_labels.shape)

# mnist = tf.keras.datasets.mnist
# (train_clusters, train_labels), (x_test, y_test) = mnist.load_data()
# train_clusters, x_test = train_clusters / 255.0, x_test / 255.0

conv = Conv3x3(3)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # out = conv.forward(image)#(image / 255) - 0.5)
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

# Training

train_clusters = np.array([final_image], 'int')
print(train_clusters.shape)
train_labels = np.array([0], 'int')
print(train_labels.shape)

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_clusters))
  train_clusters = train_clusters[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_clusters, train_labels)):
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc
