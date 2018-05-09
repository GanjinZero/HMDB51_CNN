# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:39:06 2018

@author: GanJinZERO
"""

import tensorflow as tf
import numpy as np
import time
import os 
  
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
start=time.clock()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean = 0, stddev = 0.03)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, k):
    return tf.nn.conv2d(x, W, strides=[1, k, k, 1], padding = 'SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1],
                          strides = [1, k, k, 1], padding = 'SAME')

print("Function init done.")
    
max_steps = 800  
batchsize = 1673
datasize = 5093
testsize = 1673

width = 60
height = 140
count = 1

#classset_train imageset_train
#classset_test imageset_test
#classset_train,imageset_train = generate_data(datasize)
#classset_test,imageset_test = generate_data(testsize)

#x_batch,y_batch=tf.train.batch([imageset_train, classset_train], batch_size=batchsize, capacity=32)

x_holder = tf.placeholder(tf.float32,[None, width,height,15])
x_image = tf.reshape(x_holder, [-1, width, height, 15])
x_image1, x_image2, x_image3, x_image4, x_image5 = tf.split(x_image,5,axis=3)
#x_image = tf.truncated_normal([1,560,240,3])
W_conv11 = weight_variable([7, 7, 3, 32])
b_conv11 = bias_variable([32])
h_conv11 = tf.nn.relu(conv2d(x_image1, W_conv11, 3) + b_conv11)
h_pool11 = max_pool(h_conv11,2)
#60*140*3->10*24*32
W_conv12 = weight_variable([7, 7, 3, 32])
b_conv12 = bias_variable([32])
h_conv12 = tf.nn.relu(conv2d(x_image2, W_conv12, 3) + b_conv12)
h_pool12 = max_pool(h_conv12,2)

W_conv13 = weight_variable([7, 7, 3, 32])
b_conv13 = bias_variable([32])
h_conv13 = tf.nn.relu(conv2d(x_image3, W_conv13, 3) + b_conv13)
h_pool13 = max_pool(h_conv13,2)

W_conv14 = weight_variable([7, 7, 3, 32])
b_conv14 = bias_variable([32])
h_conv14 = tf.nn.relu(conv2d(x_image4, W_conv14, 3) + b_conv14)
h_pool14 = max_pool(h_conv14,2)

W_conv15 = weight_variable([7, 7, 3, 32])
b_conv15 = bias_variable([32])
h_conv15 = tf.nn.relu(conv2d(x_image5, W_conv15, 3) + b_conv15)
h_pool15 = max_pool(h_conv15,2)
    
W_conv21 = weight_variable([3, 3, 32, 64])
b_conv21 = bias_variable([64])
h_conv21 = tf.nn.relu(conv2d(h_pool11, W_conv21, 1) + b_conv21)
h_pool21 = max_pool(h_conv21,2)

W_conv22 = weight_variable([3, 3, 32, 64])
b_conv22 = bias_variable([64])
h_conv22 = tf.nn.relu(conv2d(h_pool12, W_conv22, 1) + b_conv22)
h_pool22 = max_pool(h_conv22,2)

W_conv23 = weight_variable([3, 3, 32, 64])
b_conv23 = bias_variable([64])
h_conv23 = tf.nn.relu(conv2d(h_pool13, W_conv23, 1) + b_conv23)
h_pool23 = max_pool(h_conv23,2)

W_conv24 = weight_variable([3, 3, 32, 64])
b_conv24 = bias_variable([64])
h_conv24 = tf.nn.relu(conv2d(h_pool14, W_conv24, 1) + b_conv24)
h_pool24 = max_pool(h_conv24,2)

W_conv25 = weight_variable([3, 3, 32, 64])
b_conv25 = bias_variable([64])
h_conv25 = tf.nn.relu(conv2d(h_pool15, W_conv25, 1) + b_conv25)
h_pool25 = max_pool(h_conv25,2)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print (sess.run(tf.shape(h_conv2)))
#    print (sess.run(tf.shape(h_pool2)))
#10*24*32->5*12*64

h_pool21_flat = tf.reshape(h_pool21, [-1, 5*12*64])
h_pool22_flat = tf.reshape(h_pool22, [-1, 5*12*64])
h_pool23_flat = tf.reshape(h_pool23, [-1, 5*12*64])
h_pool24_flat = tf.reshape(h_pool24, [-1, 5*12*64])
h_pool25_flat = tf.reshape(h_pool25, [-1, 5*12*64])

state_size = 256
dim = 12*5*64

init_state = tf.zeros([batchsize, state_size])
rnn_inputs = [h_pool21_flat,h_pool22_flat,h_pool23_flat,h_pool24_flat,h_pool25_flat]

with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [dim + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [dim + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
    
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

W_fc1 = weight_variable([state_size, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(final_state, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 51])
b_fc2 = bias_variable([51])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y1_conv = tf.argmax(y_conv,1)

y_holder = tf.placeholder("float", [None, 51])
cross_entropy = -tf.reduce_sum(y_holder * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_holder,1)) #top-1
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
top_k_op = tf.nn.in_top_k(y_conv, tf.argmax(y_holder,1), 5) 
acc_k = tf.reduce_mean(tf.cast(top_k_op,"float"))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

print("test")

start_point = 0
end_point = batchsize
imageset_test = imageset_test000 / 255.0

for i in range(max_steps):
    #print(i)
    start_time = time.clock()
    #print(start_point)
    #print(end_point)
    #x_b, y_b = sess.run([x_batch, y_batch])
    if end_point>start_point:
        x_b = imageset_train[start_point:end_point]
        y_b = classset_train[start_point:end_point]
    else:
        x_b = np.concatenate((imageset_train[start_point:datasize],imageset_train[0:end_point]))
        y_b = np.concatenate((classset_train[start_point:datasize],classset_train[0:end_point]))
    start_point = end_point
    end_point = (start_point+batchsize)%datasize
    #rand = np.random.random([256, 60, 140, 3])
    #x_b = x_b + rand
    x_b = x_b / 255
    train_step.run(session = sess, feed_dict = {x_holder:x_b, y_holder:y_b,keep_prob:0.5}) 
    if (i % 10) == 0:
        train_accuracy = accuracy.eval(session = sess,feed_dict = {x_holder:x_b, y_holder:y_b, keep_prob:1.0})
        print("step %d, train_accuracy %g" %(i, train_accuracy))
        #yc = y_conv.eval(session=sess,feed_dict = {x_holder:x_b, y_holder:y_b, keep_prob:1.0})
        #print(yc)
        #print("K-train_accuracy %g" %(train_acc_k))
        end = time.clock();
        #top_k_op.eval(session=sess,feed_dict = {x_holder:imageset_test, y_holder:classset_test, keep_prob:1.0})
        #print(top_k_op)
        acc,acck,ycc=sess.run([accuracy,acc_k,y1_conv],feed_dict = {x_holder:imageset_test, y_holder:classset_test,keep_prob:1.0})
        print(ycc)
        print("test accuracy %g" %acc)
        print("K-test accuracy %g" %acck)
        print((end-start_time)*10)
        print(" ")

end = time.clock()
print(end-start)
