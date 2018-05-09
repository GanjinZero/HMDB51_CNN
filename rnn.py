# -*- coding: utf-8 -*-
"""
Created on Mon May  7 02:14:00 2018

@author: ACER
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


d1=23*400  #d1是样本量
d2=5   #d2是图片的一维长度
d3=12  #d2是图片的第二维长度
d4=64 #d2是图片的第三维长度，也就是卷积核数量
d5=5  #五张照片
def get_data(d1=23*400
             ,d2=5,d3=12,d4=64,d5=5):
    a1=np.random.random([d1,d2*d3*d4])
    a2=np.zeros([d1,d2*d3*d4,d5])
    label=np.zeros([d1,5])
    for t in range(d5):
        a2[:(d1//3),:,t]=a1[:(d1//3),:]/(t+1)
        label[:(d1//3),:]=0
        a2[(d1//3+1):(d1//3*2),:,t]=np.power(a1[(d1//3+1):(d1//3*2),:],t)
        label[(d1//3+1):(d1//3*2),:]=1
        a2[(d1//3*2+1):,:,t]=np.power(a1[(d1//3*2+1):,:],1/(t+1))
        label[(d1//3*2+1):,:]=2
    return a2,label

dat,label=get_data(d1,d2,d3,d4,d5)
dat=np.swapaxes(dat,1,2)

###输入的数据要具有（d1,d2*d3*d4,d5)的形式 ，中间一维即图片展平了，即 h_pool2_flat



batch_size = 256  #可以自己选
num_classes = 51  #应该改成51
state_size = 20  #可以自己选择
dim=d4*d2*d3
num_steps = 5
learning_rate = 0.001 #可以自己选


#############################################################
##############################################################
x = tf.placeholder(tf.float32, [batch_size, num_steps,dim], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])
rnn_inputs = [x[:,i,:] for i in range(num_steps)]#要变成序列
####因为用不好封装的核只好到网上下了手写的
#定义rnn_cell的权重参数，
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [dim + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#使之定义为reuse模式，循环使用，保持参数相同
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [dim + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    #定义rnn_cell具体的操作，这里使用的是最简单的rnn，不是LSTM
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

state = init_state
rnn_outputs = []
#循环num_steps次，即将一个序列输入RNN模型
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]


with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
            [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

##################################################
#################################################

def train_network(num_epochs, num_steps, state_size=20):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        #得到数据，因为num_epochs==5，所以外循环只执行五次
        training_loss = 0
        for idx in range(num_epochs):
            #保存每次执行后的最后状态，然后赋给下一次执行
            training_state = np.zeros((batch_size, state_size))
          
            #这是具体获得数据的部分
            for step in range(d1//batch_size):
               # print(step)
                train_dat=dat[batch_size*(step):batch_size*(step+1)]
                train_label=label[batch_size*(step):batch_size*(step+1)]
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:train_dat, y:train_label,init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    print("Average loss at step", step,
                          "for last 100 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses
training_losses = train_network(20,num_steps,state_size)
plt.plot(training_losses)
plt.show()



##############没有写在测试集上跑的结果#######