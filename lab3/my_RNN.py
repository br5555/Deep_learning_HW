#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:41:20 2018

@author: branko
"""
import os, sys
import re
import numpy as np
from collections import Counter
from collections import OrderedDict
from operator import itemgetter 
from random import randint 

class my_RNN2:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        np.random.seed(seed=32)
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        self.U = np.random.randn(hidden_size, vocab_size)*0.01 # ... input projection
        self.W = np.random.randn(hidden_size, hidden_size)*0.01 # .. hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1)) # ... input bias
        
        self.V = np.random.randn(vocab_size, hidden_size) # ... output projection
        self.c = np.zeros((vocab_size, 1)) # ... output bias
        
        #memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W),  np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)
    
    def get_W(self):
        return self.W
    
    def get_V(self):
        return self.V
    
    def get_U(self):
        return self.U
    
    def get_b(self):
        return self.b
    
    def get_c(self):
        return self.c

    def set_W(self, W):
        self.W = W
    
    def set_V(self, V):
        self.V = V
    
    def set_U(self, U):
        self.U = U
    
    def set_b(self, b):
        self.b = b
    
    def set_c(self, c):
        self.c = c    
    
    def rnn_step_forward(self, inputs,targets, h_prev):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity
        
        # x - input data (minibatch size x input dimension)
        # h_prev - previous hiddent state (minibatch size x  hidden size)
        # U - input projection matrix (input dimension x hidden size)
        #W - hidden to hidden projection matrix (hidden size x hidden size)
        #b - bias of shape (hidden size x 1)
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0

        for t in range(len(inputs)):
            
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            hs[t] = np.tanh( np.dot(self.U, xs[t]) + np.dot(self.W, hs[t-1]) + self.b)
            ys[t] = np.dot(self.V, hs[t])  + self.c
            
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])

        
        return hs[len(inputs) -1], hs, loss, ys, ps, xs
    
    def rnn_forward(self, x, targets, h0 ):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        minibatch_size = len(x) // self.sequence_length
        h_m, p_m, y_m, x_one_hot ={}, {}, {}, {}
        loss_all = 0
        for i in range(minibatch_size):
            x_m = x[i*(self.sequence_length): (i+1)*self.sequence_length]
            targets_m = targets[i*(self.sequence_length): (i+1)*self.sequence_length]
            
            h__, h, loss, y, p, x_step = self.rnn_step_forward(x_m, targets_m, h0 )
            if i == 0:
                h_m[-1] = h[-1]
            
            for k in p.keys():
                h_m[i*self.sequence_length + k] = h[k]
                loss_all += loss
                p_m[i*self.sequence_length + k] = p[k]
                y_m[i*self.sequence_length + k] = y[k]
                x_one_hot[i*self.sequence_length + k] = x_step[k]

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h_m, p_m, y_m, x_one_hot, loss_all 
    
    def rnn_step_backward(self, inputs, targets, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.
    
        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        hs = cache['hs']
        ps = cache['ps']
        xs = cache['xs']
        dhnext = np.zeros_like(hs[0])
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W),  np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dc += dy
            dh = np.dot(self.V.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            db += dhraw
            dU +=np.dot(dhraw, xs[t].T)
            dW +=np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.W.T, dhraw)
            
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)
        
        
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
    
        return dU, dW, dV, db, dc
    
    
    def rnn_backward(self, inputs, targets, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W),  np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)
        minibatch_size = len(inputs) // self.sequence_length 
        
        for i in range(minibatch_size):
            x_m, targets_m, hs, ps, xs = {}, {}, {}, {}, {}
            hs[-1] = cache['hs'][ -1]
            for k in range(self.sequence_length):
                x_m[k] = inputs[(i)*self.sequence_length + k]
                targets_m[k] = targets[(i)*self.sequence_length + k]
                hs[k] = cache['hs'][(i)*self.sequence_length + k]
                ps[k] = cache['ps'][(i)*self.sequence_length + k]
                xs[k] = cache['xs'][(i)*self.sequence_length + k]
            cache_m = {'hs': hs, 'ps': ps, 'xs': xs}
            dU, dW, dV, db, dc  = self.rnn_step_backward(x_m, targets_m, cache_m)
            
            self.memory_U += dU
            self.memory_b += db
            self.memory_c += dc
            self.memory_V += dV
            self.memory_W += dW
            
            
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
    
        return self.memory_U, self.memory_W, self.memory_b , self.memory_c, self.memory_V
    
    def rnn_pred(self, my_data_processing, num_predicitons=100, last = 0):
        batch_nn = randint(0, my_data_processing.get_num_batches())
        h0 = np.zeros((self.hidden_size, 1))
        
        chat = ''
        for i in range(batch_nn):
            new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
            
        for i in range(num_predicitons):
            h_m, p_m, y_m, x_one_hot, loss_all = self.rnn_forward(batch_x, batch_y, h0)
            new_batch = []
            for key in y_m.keys():
                new_batch += [np.argmax(y_m[key])]
                b = np.zeros_like(y_m[key])
                b[np.argmax(y_m[key])] = 1
                y_m[key] = b

            batch_x = np.asarray(new_batch)

            chat +=my_data_processing.decode([np.argmax(y_m[last])])[0]
        print(chat)

def main():
    hidden_size = 5
    sequence_length, vocab_size, learning_rate = 3, 70, 1e-1
    
    
    h0 = np.zeros((hidden_size, 1))
    my_rnn = my_RNN2(hidden_size, sequence_length, vocab_size, learning_rate)    
    batch_x = [1,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1,1,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1, 1,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1, 1,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1]
    batch_y = [2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1,44,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1,44,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1,44,2,3,4,5,6,7,4,3,2,1,3,45,67,54,3,2,1,44]
    h_m, p_m, y_m, x_one_hot, loss_all = my_rnn.rnn_forward(batch_x, batch_y, h0)
    cache_m = {'hs': h_m, 'ps': p_m, 'xs': x_one_hot}
    dU, dW, db , dc, dV=my_rnn.rnn_backward(batch_x, batch_y, cache_m)
    print(dU)
    print(dW)
    print(db)
    
    
if __name__ == '__main__':
    main()  

