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

import os
import numpy as np
import tensorflow as tf

FLAGS = None

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

  t1 = [tf.convert_to_tensor(train_xs[0]),tf.convert_to_tensor(train_xs[1])]
  t2 = [4,1]
  t2 = tf.convert_to_tensor(t2, tf.float32)
  
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

  epoch = 3000
  learning_rate = 0.0003
  momentum = 0.099

  # Create the model
  x = tf.placeholder(tf.float32, [None, 9])
  
  tf.set_random_seed(0)
  w1 = tf.Variable(tf.random_normal([9, 5]))
  tf.set_random_seed(0)
  b1 = tf.Variable(tf.zeros([5]))
  y1 = tf.matmul(x, w1) + b1
  
  w2 = tf.Variable(tf.random_normal([5, 2]))
  tf.set_random_seed(0)
  b2 = tf.Variable(tf.zeros([2]))
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
  
  no_of_tasks = 7
  no_of_example_per_task = 200 
  
  for i in range(no_of_tasks):
    individual_task_x = test_xs[np.where(test_xs[:,i]==1)]
    individual_task_y = test_ys[i*no_of_example_per_task:(i+1)*no_of_example_per_task, :]
    print ("Accuracy for Task ", i+1, " :")
    print (sess.run(accuracy, feed_dict={x: individual_task_x, y_: individual_task_y}))


    
#  
#  test_1_xs = [[0 for x in range(9)] for y in range(200)]
#  test_2_xs = [[0 for x in range(9)] for y in range(200)]
#  test_3_xs = [[0 for x in range(9)] for y in range(200)]
#  test_4_xs = [[0 for x in range(9)] for y in range(200)]
#  test_5_xs = [[0 for x in range(9)] for y in range(200)]
#  test_6_xs = [[0 for x in range(9)] for y in range(200)]
#  test_7_xs = [[0 for x in range(9)] for y in range(200)]
#
#  test_1_ys = [[0 for x in range(2)] for y in range(200)]
#  test_2_ys = [[0 for x in range(2)] for y in range(200)]
#  test_3_ys = [[0 for x in range(2)] for y in range(200)]
#  test_4_ys = [[0 for x in range(2)] for y in range(200)]
#  test_5_ys = [[0 for x in range(2)] for y in range(200)]
#  test_6_ys = [[0 for x in range(2)] for y in range(200)]
#  test_7_ys = [[0 for x in range(2)] for y in range(200)]
#
#  for i in range(len(test_xs)):
#    
#    if test_xs[i][0] == 1:
#      test_1_xs[i] = test_xs[i]
#      test_1_ys[i] = test_ys[i]
#    elif test_xs[i][1] == 1:
#      test_2_xs[i-200] = test_xs[i]
#      test_2_ys[i-200] = test_ys[i]
#    elif test_xs[i][2] == 1:
#      test_3_xs[i-400] = test_xs[i]
#      test_3_ys[i-400] = test_ys[i]
#    elif test_xs[i][3] == 1:
#      test_4_xs[i-600] = test_xs[i]
#      test_4_ys[i-600] = test_ys[i]
#    elif test_xs[i][4] == 1:
#      test_5_xs[i-800] = test_xs[i]
#      test_5_ys[i-800] = test_ys[i]
#    elif test_xs[i][5] == 1:
#      test_6_xs[i-1000] = test_xs[i]
#      test_6_ys[i-1000] = test_ys[i]
#    elif test_xs[i][6] == 1:
#      test_7_xs[i-1200] = test_xs[i]
#      test_7_ys[i-1200] = test_ys[i]
#      
#  print ("Accuracy :")
#  print ("Task 1:")
#  print (sess.run(accuracy, feed_dict={x: test_1_xs, y_: test_1_ys}))
#  print ("Task 2:")
#  print (sess.run(accuracy, feed_dict={x: test_2_xs, y_: test_2_ys}))
#  print ("Task 3:")
#  print (sess.run(accuracy, feed_dict={x: test_3_xs, y_: test_3_ys}))
#  print ("Task 4:")
#  print (sess.run(accuracy, feed_dict={x: test_4_xs, y_: test_4_ys}))
#  print ("Task 5:")
#  print (sess.run(accuracy, feed_dict={x: test_5_xs, y_: test_5_ys}))
#  print ("Task 6:")
#  print (sess.run(accuracy, feed_dict={x: test_6_xs, y_: test_6_ys}))
#  print ("Task 7:")
#  print (sess.run(accuracy, feed_dict={x: test_7_xs, y_: test_7_ys}))

  #predictions = tf.argmax(y,1)
  #predictions = predictions.eval(feed_dict={x: test_xs}, session=sess)
  #predictions = predictions.reshape(-1 ,1)
  predictions = y.eval(feed_dict={x: test_xs}, session=sess)
  MSE = tf.losses.mean_squared_error(test_ys,predictions)
  print("MSE: %.4f" % sess.run(MSE))
  