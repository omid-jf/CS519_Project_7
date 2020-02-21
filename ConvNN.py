# ======================================================================= 
# This file is part of the CS519_Project_7 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import tensorflow as tf
import numpy as np
import os


class ConvNN(object):
    def __init__(self, batchsize=100, epochs=20, learning_rate=1e-4, dropout_rate=0.5, shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph()  # Create an empty graph
        with g.as_default():
            tf.set_random_seed(random_seed)  # Set random-seed
            self.build()  # Build the network
            self.init_op = tf.global_variables_initializer()  # Initializer
            self.saver = tf.train.Saver()  # Saver

        self.sess = tf.Session(graph=g)  # Create a session

    def build(self):
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
        tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")
        is_train = tf.placeholder(tf.bool, shape=(), name="is_train")

        # reshape x to a 4D tensor: [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name="input_x_2dimages")
        # One hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name="input_y_onehot")

        # 1st layer: Conv_1
        h1 = tf.layers.conv2d(tf_x_image, kernel_size=(3, 3), strides=(1, 1), padding="valid", filters=4, activation=tf.nn.relu)
        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2))  # MaxPooling

        # 2nd layer: Conv_2
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(3, 3), strides=(3, 3), padding="valid", filters=2, activation=tf.nn.relu)
        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(4, 4), strides=(4, 4))  # MaxPooling

        # 3rd layer: Fully Connected
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 10, activation=tf.nn.relu)

        # Prediction
        predictions = {"probabilities": tf.nn.softmax(h3, name="probabilities"),
                       "labels": tf.cast(tf.argmax(h3, axis=1), tf.int32, name="labels")}

        # Loss Function and Optimization
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=tf_y_onehot),
                                            name="cross_entropy_loss")

        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name="train_op")

        # Finding accuracy
        correct_predictions = tf.equal(predictions["labels"], tf_y, name="correct_preds")
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    def batch_generator(self, X, y, batch_size=100, shuffle=False, random_seed=None):
        idx = np.arange(y.shape[0])
        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(idx)
            X = X[idx]
            y = y[idx]

        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size, :], y[i:i + batch_size])

    def train(self, training_set, validation_set=None, initialize=True):
        # Initialize variables
        if initialize:
            self.sess.run(self.init_op)

        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1, self.epochs + 1):
            batch_gen = self.batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {"tf_x:0": batch_x, "tf_y:0": batch_y, "is_train:0": True}
                loss, _ = self.sess.run(["cross_entropy_loss:0", "train_op"], feed_dict=feed)
                avg_loss += loss

            print("Epoch %02d: Training Avg. Loss: ""%7.3f" % (epoch, avg_loss), end=" ")
            if validation_set is not None:
                feed = {"tf_x:0": batch_x, "tf_y:0": batch_y, "is_train:0": False}
                valid_acc = self.sess.run("accuracy:0", feed_dict=feed)
                print("Validation Acc: %7.3f" % valid_acc)
            else:
                print()

    def predict(self, X_test, return_proba=False):
        feed = {"tf_x:0": X_test, "is_train:0": False}
        if return_proba:
            return self.sess.run("probabilities:0", feed_dict=feed)
        else:
            return self.sess.run("labels:0", feed_dict=feed)

    def save(self, epoch, path="./tflayers-model/"):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model in %s" % path)
        self.saver.save(self.sess, os.path.join(path, "model.ckpt"), global_step=epoch)

    def load(self, epoch, path):
        print("Loading model from %s" % path)
        self.saver.restore(self.sess, os.path.join(path, "model.ckpt-%d" % epoch))












