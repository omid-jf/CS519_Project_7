# ======================================================================= 
# This file is part of the CS519_Project_7 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import numpy as np
import os
import struct
from ConvNN import ConvNN


# Mnist dataset loader
def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# Reading Mnist dataset
X_data, y_data = load_mnist("", kind="train")
X_test, y_test = load_mnist("", kind="t10k")

# Splitting training and validating sets
X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]

print("Training set:\t", X_train.shape, y_train.shape)
print("Validation set:\t", X_valid.shape, y_valid.shape)
print("Test set:\t", X_test.shape, y_test.shape)

# Standardizing data
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = (X_valid - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val

# CNN
cnn = ConvNN(random_seed=123)
cnn.train(training_set=(X_train_centered, y_train), validation_set=(X_valid_centered, y_valid), initialize=True)
cnn.save(epoch=20)
del cnn

cnn2 = ConvNN(random_seed=123)
cnn2.load(epoch=20, path="./tflayers-model/")
print(cnn2.predict(X_test_centered[:10, :]))

preds = cnn2.predict(X_test_centered)
print("Test Accuracy: %.2f%%" % (100 * np.sum(y_test == preds) / len(y_test)))

