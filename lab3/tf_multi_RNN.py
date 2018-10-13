#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 01:32:05 2018

@author: branko
"""

import tensorflow as tf
import numpy as np
from dataset import data_processing
import os, sys
import re
from random import randint

class multilayer_RNN:
    
    def __init__(self,n_neurons, n_layers, n_steps, n_inputs, n_outputs, learning_rate):
        
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.int32, [None, self.n_steps])
        
        self.layers = [tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                                  activation=tf.nn.relu), output_size=n_outputs)
        
                        for layer in range(self.n_layers)]
        self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.layers)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X, dtype=tf.float32)
        
        self.states_concat = tf.concat(axis = 1, values=self.states)
        
        self.logits = self.outputs#tf.layers.dense(self.states_concat, self.n_outputs)
        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(self.xentropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gvs]
        self.train_op = self.optimizer.apply_gradients(self.capped_gvs)

        
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        
    def train(self, X_train, y_train, batch_size, n_epochs):
        
        n, _ = X_train.shape
        self.session.run(self.init)
        for epoch in range(n_epochs):
            for iteration in range(n // batch_size):
                X_batch = X_train[iteration*batch_size : (iteration + 1)*batch_size]
                y_batch = y_train[iteration*batch_size : (iteration + 1)*batch_size]
                X_batch = X_batch.reshape((-1, self.n_steps, self.n_inputs))
                y_batch = y_batch.reshape((-1, self.n_steps, self.n_inputs))
                self.session.run(self.training_op, feed_dict={self.X:X_batch, self.y: y_batch})
            acc_train = self.session(self.accuracy, feed_dict={self.X:X_batch, self.y: y_batch})
            print(epoch, "Train accuracy", acc_train)
            
    def train2(self, my_data_processing, num_epochs = 100):
        
        n_epochs = num_epochs
        self.session.run(self.init)
        while(True):
        
            new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
            
            if(new_epoch):
                self.saver.save(self.session, "./my_time_series_model")     
                new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
                num_epochs -= 1
                
                if(num_epochs == 0):
    
                    break
            else:
                batch_x = self.one_hot(batch_x, self.n_inputs)
#                batch_y = self.one_hot(batch_y, self.n_inputs)
                batch_x = batch_x.reshape((-1, self.n_steps, self.n_inputs))
                batch_y = batch_y.reshape((-1, self.n_steps))                
                _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.X:batch_x, self.y: batch_y})
                
                
#                acc_train = self.session(self.accuracy, feed_dict={self.X:batch_x, self.y: batch_y})
                print("epoch: ", n_epochs-num_epochs, " loss is ", loss)
                
    def one_hot(self, ulaz, duljina):
        n = ulaz.shape[0]
        izlaz = np.zeros((n,duljina))
        for i in range(n):
            izlaz[i][ulaz[i]] = 1
    
        return izlaz
    
    def predicition(self, my_data_processing, num_predicitons = 100):
        
        self.saver.restore(self.session, "./my_time_series_model")
        batch_nn = randint(0, my_data_processing.get_num_batches())
        for i in range(batch_nn):
            new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
            
        batch_x = self.one_hot(batch_x, self.n_inputs)
        for i in range(num_predicitons):
            
            batch_x = batch_x.reshape((-1, self.n_steps, self.n_inputs))
            outputs = self.session.run(self.outputs, feed_dict={self.X:batch_x})
            batch_ss, seq_len, vocab_len = outputs.shape
            batch_x = np.zeros_like(outputs)
            for i in range(batch_ss):
                for j in range(seq_len):
                    batch_x[i][j][np.argmax(outputs[i][j][:])] = 1
             
    
            print(my_data_processing.decode([np.argmax(outputs[-1][-1][:])]))
        
            
def main():
    # uses hardcoded paths
    hidden_size = 100
    root = '/home/branko/Documents/lab3/dlunizg.github.io/code/lab3/data'
    output_destination = 'selected_conversations.txt'
    
    output = os.path.join(root, output_destination)
    my_data_processing = data_processing(output, 200,30)    
    my_data_processing.preprocess(output)
    my_data_processing.create_minibatches()
    
    sequence_length, vocab_size, learning_rate = 30, my_data_processing.get_num_chars(), 1e-1
    my_rnn = multilayer_RNN(n_neurons=hidden_size, n_layers=3, n_steps=sequence_length, n_inputs=vocab_size
                            , n_outputs=vocab_size, learning_rate=learning_rate)
    my_rnn.train2(my_data_processing=my_data_processing)
            
def main_pred():
    # uses hardcoded paths
    hidden_size = 100
    root = '/home/branko/Documents/lab3/dlunizg.github.io/code/lab3/data'
    output_destination = 'selected_conversations.txt'
    
    output = os.path.join(root, output_destination)
    my_data_processing = data_processing(output, 200,30)    
    my_data_processing.preprocess(output)
    my_data_processing.create_minibatches()
    
    sequence_length, vocab_size, learning_rate = 30, my_data_processing.get_num_chars(), 1e-1
    my_rnn = multilayer_RNN(n_neurons=hidden_size, n_layers=3, n_steps=sequence_length, n_inputs=vocab_size
                            , n_outputs=vocab_size, learning_rate=learning_rate)
    my_rnn.predicition(my_data_processing=my_data_processing)            
           


if __name__ == '__main__':
#    main()
    main_pred()          


        