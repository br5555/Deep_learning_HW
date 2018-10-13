#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 01:03:00 2018

@author: branko
"""

import tensorflow as tf
import numpy as np

class onelayer_RNN:
    
    def __init__(self, n_steps, n_inputs, n_neurons, n_outputs, learning_rate):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_inputs= n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        
        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.int32 , [None])
        
        self.basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons)
        self.outputs1, self.states1 = tf.nn.dynamic_rnn(self.basic_cell, self.X, dtype=tf.float32)
        
        self.logits = tf.layers.dense(self.states1, n_outputs)
        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        
        self.loss = tf.reduce_mean(self.xentropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.loss)
        self.correct = tf.nn.in_top_k(self.logits, self.y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
        
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
    def train(self, n_epochs = 100, batch_size = 150, X_train, y_train):
        n, _ = X_train.shape
        self.session.run(self.init)
        for epoch in range(n_epochs):
            for iteration in range(n // batch_size):
                X_batch = X_train[iteration*batch_size : (iteration + 1)*batch_size]
                y_batch = y_train[iteration*batch_size : (iteration + 1)*batch_size]
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                self.session.run(self.training_op, feed_dict={X:X_batch, y: y_batch})
            acc_train = self.session(self.accuracy, feed_dict={X:X_batch, y: y_batch})
            print(epoch, "Train accuracy", acc_train)
        
        
        
        