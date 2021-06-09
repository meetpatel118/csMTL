# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import tensorflow as tf

FLAGS = None

def main():
  
  # Import data
  train_data = load_data('Tstloooo.trn')
  test_data = load_data('Tstloooo.tst')
  validation_data = load_data('Tstloooo.val')

  train_xs = train_data[:,:-1]
  #train_xs = np.repeat(train_xs[:, np.newaxis], 1,axis=2)
  train_ys = train_data[:,-1]
  train_ys = train_ys.reshape(-1 ,1)
  #train_ys = np.repeat(train_ys[:, :, np.newaxis], 2, axis=2)

  test_xs = test_data[:,:-1]
  #test_xs = np.repeat(test_xs[:, np.newaxis], 1,axis=2)
  test_ys = test_data[:,-1]
  test_ys = test_ys.reshape(-1 ,1)
  #test_ys = np.repeat(test_ys[:, :, np.newaxis], 2, axis=2)

  validation_xs = validation_data[:,:-1]
  validation_ys = validation_data[:,-1]
  validation_ys = validation_data[:,-1].reshape(-1 ,1)

  i=0
  label = [[0 for x in range(2)] for y in range(len(train_ys))] 

  for i in range(len(train_ys)):
    if train_ys[i][0] == 0.1:
      label[i][0] = 0.1
      label[i][1] = 0.0
    elif train_ys[i][0] == 0.9:
      label[i][0] = 0.0
      label[i][1] = 0.9

  train_ys = np.reshape(label,(len(train_ys),-1))

  i=0
  label = [[0 for x in range(2)] for y in range(len(test_ys))] 
  
  for i in range(len(test_ys)):
    if test_ys[i][0] == 0.1:
      label[i][0] = 0.1
      label[i][1] = 0.0
    elif test_ys[i][0] == 0.9:
      label[i][0] = 0.0
      label[i][1] = 0.9

  test_ys = np.reshape(label,(len(test_ys),-1))

  epoch = 100
  learning_rate = 0.0001
  momentum = 0.6

  # Create the model
  x = tf.placeholder(tf.float32, [None, 9])

  w1 = tf.Variable(tf.random_normal([9, 5]))
  b1 = tf.Variable(tf.random_normal([5]))
  y1 = tf.matmul(x, w1) + b1
  
  w2 = tf.Variable(tf.random_normal([5, 2]))
  b2 = tf.Variable(tf.random_normal([2]))
  y = tf.matmul(y1, w2) + b2

  y = tf.nn.softmax(y)
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None,2])

  # The raw formulation of cross-entropy,
  #
  #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.


  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
  #train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # Train
  i = 0
  for i in range(epoch):
    sess.run(train_step, feed_dict={x: train_xs, y_: train_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print("")
  print ("Epoch :",epoch )
  print ("Learning Rate :",learning_rate )
  print ("Momentum :",momentum )
  print ("Accuracy :")
  print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

  #predictions = tf.argmax(y,1)
  #predictions = predictions.eval(feed_dict={x: test_xs}, session=sess)
  #predictions = predictions.reshape(-1 ,1)
  predictions = y.eval(feed_dict={x: test_xs}, session=sess)
  MSE = tf.losses.mean_squared_error(test_ys,predictions)
  print("MSE: %.4f" % sess.run(MSE))
  

def load_data(filename):
    """
    Loads the data from the  file into a numpy array

    :return: A numpy array with the flattened data
    """
    last_i = __file__.rfind(os.sep)
    current_dir = __file__[:last_i+1]

    with open(current_dir + filename, 'r') as lp_f:
        return np.loadtxt(lp_f, dtype=float, delimiter='	')

if __name__ == '__main__':
  main()
