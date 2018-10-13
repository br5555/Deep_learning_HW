#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:22:51 2018

@author: branko
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import matplotlib.pyplot as plt
import math
tf.reset_default_graph()

plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
n_samples = mnist.train.num_examples

# training parameters
batch_size = 100
lr = 0.0002
n_epochs = 20

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=isTrain), 0.2)
        
        conv1 = tf.layers.conv2d(lrelu_, 128, [4, 4], strides=(2, 2),padding='same')
        lrelu_1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
        
        
        logits = tf.layers.conv2d(lrelu_1, 1, [8, 8], strides=(16, 16), padding='same')

        out = tf.nn.sigmoid(logits)
        
        return out, logits

# G(z)
def generator(z, isTrain=True):
    with tf.variable_scope('generator'):
        # 1st hidden layer
        conv = tf.layers.conv2d_transpose(z, 512, [4, 4], strides=(1, 1),reuse=None, padding='valid')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=isTrain))
        
        conv1 = tf.layers.conv2d_transpose(lrelu_, 256, [4, 4], strides=(2, 2),reuse=None, padding='valid')
        lrelu_1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))
        
        conv2 = tf.layers.conv2d_transpose(lrelu_1, 128, [4, 4], strides=(2, 2),reuse=None, padding='valid')
        lrelu_2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain))
        
        out = tf.layers.conv2d_transpose(lrelu_2, 1, [4, 4], strides=(2, 2),reuse=None, padding='valid')
        out = tf.nn.tanh(out)
        return out

def show_generated(G, N, shape=(32,32), stat_shape=(10,10), interpolation="bilinear"):
    """Visualization of generated samples
     G - generated samples
     N - number of samples
     shape - dimensions of samples eg (32,32)
     stat_shape - dimension for 2D sample display (eg for 100 samples (10,10)
    """
    
    image = (tile_raster_images(
        X=G,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')
    plt.show()
    

def gen_z(N, batch_size):
    z = np.random.normal(0, 1, (batch_size, 1, 1, N))
    return z

# input variables
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
z = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
    
# generator
G_z = generator(z, isTrain)
    
# discriminator
# real
D_real, D_real_logits = discriminator(x, isTrain, reuse=False)
# fake
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)


# labels for learning
true_labels = tf.ones(shape=[batch_size, 1, 1, 1], dtype=tf.float32)
fake_labels = tf.zeros(shape=[batch_size, 1, 1, 1], dtype=tf.float32)

#
mean, var = tf.nn.moments(D_fake_logits, [0], keep_dims=True)
D_fake_logits = tf.div(tf.subtract(D_fake_logits, mean), tf.sqrt(var))


mean, var = tf.nn.moments(D_real_logits, [0], keep_dims=True)
D_real_logits = tf.div(tf.subtract(D_real_logits, mean), tf.sqrt(var))
# loss for each network 
D_loss_real =tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=D_real_logits)
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=D_fake_logits)
D_loss = D_loss_real + D_loss_fake
G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=D_fake_logits)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(G_loss, var_list=G_vars)


# open session and initialize all variables
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()
std_uk = 0.0
mean_uk = 0.0
for i in range(train_set.shape[0]):
    std_uk += np.std(train_set[i,:,:,0])
    mean_uk += np.mean(train_set[i,:,:,0])
    
std_uk /= train_set.shape[0]
mean_uk /= train_set.shape[0]

for i in range(train_set.shape[0]):
    train_set[i,:,:,0] = (train_set[i,:,:,0] -mean_uk)/std_uk


    
print(mean_uk)
# input normalization
#train_set = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), train_set)


#fixed_z_ = np.random.uniform(-1, 1, (100, 1, 1, 100))
fixed_z_ = gen_z(100, 100)
total_batch = int(n_samples / batch_size)
for epoch in range(n_epochs):
    for iter in range(total_batch):
        # update discriminator
        
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]

        # update discriminator
        
        z_ = gen_z(100, batch_size)

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
                

        # update generator
        loss_g_, _ = sess.run([G_loss, G_optim], {x: x_, z: z_,isTrain: True})
            
    print('[%d/%d] loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), n_epochs, loss_d_, loss_g_))
    print("OK")
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
    show_generated(test_images, 100)
