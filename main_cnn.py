# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:31:38 2018

@author: GanJinZERO
"""

import tensorflow as tf
import numpy as np
import time

start=time.clock()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean = 0, stddev = 0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, k):
    return tf.nn.conv2d(x, W, strides=[1, k, k, 1], padding = 'SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1],
                          strides = [1, k, k, 1], padding = 'SAME')
    
def generate_data(n):
    num = n
    label1 = np.random.randint(51,size = num)
    label = np.zeros([n,51])
    for j in range(1,n):
        label[j-1][label1[j-1]]=1
    images = np.random.random([num, 60, 140, 3])
    return label, images

print("Function init done.")
    
max_steps = 800 
batchsize = 512
datasize = 5092
testsize = 1672

width = 60
height = 140
count = 1

#classset_train imageset_train
#classset_test imageset_test
#classset_train,imageset_train = generate_data(datasize)
#classset_test,imageset_test = generate_data(testsize)

#x_batch,y_batch=tf.train.batch([imageset_train, classset_train], batch_size=batchsize, capacity=32)

x_holder = tf.placeholder(tf.float32,[None, width,height,3])
x_image = tf.reshape(x_holder, [-1, width, height, 3])
#x_image = tf.truncated_normal([1,560,240,3])
W_conv1 = weight_variable([7, 7, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 3) + b_conv1)
h_pool1 = max_pool(h_conv1,2)
#60*140*3->10*24*32
    
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
h_pool2 = max_pool(h_conv2,2)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print (sess.run(tf.shape(h_conv2)))
#    print (sess.run(tf.shape(h_pool2)))
#10*24*32->5*12*64

W_fc1 = weight_variable([5*12*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 51])
b_fc2 = bias_variable([51])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_holder = tf.placeholder("float", [None, 51])
cross_entropy = -tf.reduce_sum(y_holder * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_holder,1)) #top-1
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
top_k_op = tf.nn.in_top_k(y_conv, tf.argmax(y_holder,1), 5) 
acc_k = tf.reduce_mean(tf.cast(top_k_op,"float"))

cross_entropy_scalar=tf.summary.scalar('cross_entropy', cross_entropy)
accuracy_scalar=tf.summary.scalar('accuracy', accuracy)
five_accuracy_scalar=tf.summary.scalar('five_accuracy', acc_k)
#merged = tf.summary.merge_all()

sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
sess.run(tf.global_variables_initializer())
train_cross_entropy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/train'+'/cross_entropy', sess.graph)
train_accuracy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/train'+'/accuracy', sess.graph)
train_five_accuracy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/train'+'/five_accuracy', sess.graph)
test_cross_entropy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/test'+'/cross_entropy', sess.graph)
test_accuracy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/test'+'/accuracy', sess.graph)
test_five_accuracy_writer=tf.summary.FileWriter('./mygraph/cnn'+ '/test'+'/five_accuracy', sess.graph)
#tf.train.start_queue_runners()

print("test")

start_point = 0
end_point = batchsize

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
        cross_entropy_scalar_0, accuracy_scalar_0, five_accuracy_scalar_0 = sess.run([cross_entropy_scalar, accuracy_scalar, five_accuracy_scalar], feed_dict = {x_holder:x_b, y_holder:y_b, keep_prob:1.0})
#        print("step %d, train_accuracy %g" %(i, train_accuracy))
        train_cross_entropy_writer.add_summary(cross_entropy_scalar_0, i)
        train_accuracy_writer.add_summary(accuracy_scalar_0, i)
        train_five_accuracy_writer.add_summary(five_accuracy_scalar_0, i)
        #yc = y_conv.eval(session=sess,feed_dict = {x_holder:x_b, y_holder:y_b, keep_prob:1.0})
        #print(yc)
        #print("K-train_accuracy %g" %(train_acc_k))
        end = time.clock();
        #top_k_op.eval(session=sess,feed_dict = {x_holder:imageset_test, y_holder:classset_test, keep_prob:1.0})
        #print(top_k_op)
        cross_entropy_scalar_0, accuracy_scalar_0, five_accuracy_scalar_0 = sess.run([cross_entropy_scalar, accuracy_scalar, five_accuracy_scalar], feed_dict = {x_holder:imageset_test, y_holder:classset_test,keep_prob:1.0})
        test_cross_entropy_writer.add_summary(cross_entropy_scalar_0, i)
        test_accuracy_writer.add_summary(accuracy_scalar_0, i)
        test_five_accuracy_writer.add_summary(five_accuracy_scalar_0, i)        
#        print("test accuracy %g" %test_accuracy)
#        K_test_accuracy = acc_k.eval(session = sess,feed_dict = {x_holder:imageset_test, y_holder:classset_test, keep_prob:1.0})
#        print("K-test accuracy %g" %K_test_accuracy)
        print((end-start_time)*10)
        print(" ")

train_cross_entropy_writer.close()
train_accuracy_writer.close()
train_five_accuracy_writer.close()
test_cross_entropy_writer.close()
test_accuracy_writer.close()
test_five_accuracy_writer.close()
end = time.clock()
print(end-start)

