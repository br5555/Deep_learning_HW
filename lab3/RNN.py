#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 03:01:04 2018

@author: branko
"""

import tensorflow as tf
import numpy as np
from collections import Counter
from collections import OrderedDict
from operator import itemgetter    

dictmap_inupt2ID = {'a': 3,
                    'b': 15,
                    'c':16,
                    'd':7,
                    'e':1,
                    'f':17,
                    'g':8,
                    'h':18,
                    'i':4,
                    'j':19,
                    'k':9,
                    'l':20,
                    'm':5,
                    'n':6,
                    'o':10,
                    'p':21,
                    'q':22,
                    'r':11,
                    's':12,
                    't':2,
                    'u':13,
                    'v':23,
                    'w':14,
                    'x':24,
                    'y':25,
                    'z':26,
                    '1':28,
                    '2':29,
                    '3':30,
                    '4':31,
                    '5':32,
                    '6':33,
                    '7':34,
                    '8':35,
                    '9':36,
                    '0':27,
                    '.':37,
                    ',':38,
                    '?':39,
                    ' ':40,
                    '\n':41
                        }
print(dictmap_inupt2ID['\n'])

s = 'branko \n ante'
print(s)
list_s = list(s)
ids = []
for i in list_s:
    ids += [dictmap_inupt2ID[i]]


print(ids)

dictmap_ID2input = { '3':'a',
                    '15':'b' ,
                    '16':'c',
                    '7':'d',
                    '1':'e',
                    '17':'f',
                    '8':'g',
                    '18':'h',
                    '4':'i',
                   '19': 'j',
                   '9': 'k',
                    '20':'l',
                    '5':'m',
                    '6':'n',
                    '10':'o',
                    '21':'p',
                    '22':'q',
                    '11':'r',
                    '12':'s',
                    '2':'t',
                    '13': 'u',
                    '23':'v',
                    '14':'w',
                    '24':'x',
                    '25':'y',
                    '26':'z',
                    '28':'1',
                    '29':'2',
                    '30':'3',
                   '31': '4',
                   '32': '5',
                    '33':'6',
                    '34':'7',
                    '35':'8',
                    '36':'9',
                    '27':'0',
                    '37':'.',
                    '38':',',
                    '39':'?',
                    '40':' ',
                    '41':'\n'
                        }
    
izlaz = ''
for i in ids:
    izlaz += dictmap_ID2input[str(i)]

print(izlaz)
    
print(dictmap_ID2input['1']+ dictmap_ID2input['2'] +dictmap_ID2input['7']+dictmap_ID2input['40']+dictmap_ID2input['41']+dictmap_ID2input['38'])
    
    

class data_processing:
    
#...
# code is nested in class definition , indentation is not representative,
# "np" stands for numpy.
    def __init__(self, path_file, batch_size, sequence_length):
        self.path_file = path_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    #f = open(fname, encoding="utf-8")
    def preprocess(self, input_file):
        with open(input_file, "r", encoding='utf8') as f:
            data = f.read() #python 3
            self.sorted_chars =list(OrderedDict( sorted(Counter(data).items(), key = itemgetter(1), reverse = True) ).keys() )
            #count and sort most frequent characters
            #self.sorted chars contains just the characters ordered descending by frequnecies
            self.char2id  = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
            #reverse the mapping
            self.id2char = {k:v for v,k in self.chars2id.items()}
            #convert the data to ids
            self.x  = np.array(list(map(self.char2id.get, data)))
            
    def encode(self, sequence):
        # returns the sequences encoded as integers
        encoeded_list = []
        
        for i in sequence:
            encoeded_list += [self.char2id[i]]
        
        return encoeded_list
    def decode(self, encoded_sequence):
        
        decoded_list = []
        
        for i in encoded_sequence:
            decoded_list += [self.id2char[str(i)]]
            
        return decoded_list

    
    def create_minibatches(self):
        self.num_batches = int(len(self.x) // (self.batch_size * self.sequence_length)) # calculate the number of batches
        
        self.batches_x = []
        self.iter = 0
        for i in range(self.num_batches):
            if (i+1)*self.batch_size * self.sequence_length != len(self.x):
                self.batches_x += [self.x[i*(self.batch_size*self.sequence_length):(i+1)*(self.batch_size*self.sequence_length)]]

                self.batches_y += [self.x[1+i*(self.batch_size*self.sequence_length):1+(i+1)*(self.batch_size*self.sequence_length)]]
            else:
                #nemam index len(self.x)
                self.batches_x += [self.x[i*(self.batch_size*self.sequence_length)-1:(i+1)*(self.batch_size*self.sequence_length)-1]]

                self.batches_y += [self.x[1+i*(self.batch_size*self.sequence_length):len(self.x)-1]]

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?
    
        #######################################
        #       Convert data to batches       #
        #######################################
    # ...
    # Code is nested in class definition, indentation is not representative.
    def next_minibatch(self):
        # ...
    
        batch_x, batch_y = None, None
        new_epoch = false
        if self.iter >= self.num_batches:
            new_epoch = true
            self.iter = 0
        else:
            batch_x = self.batches_x[self.iter]
            batch_y = self.batches_y[self.iter]
            self.iter +=1
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        
        return new_epoch, batch_x, batch_y


         
aj = 'abcdafe aca  fa  sass f sf sa a fa  ss ssssss \n fff'
list_aj = list(aj) 
print(list_aj[1:len(list_aj)-1])  
print(Counter(list_aj))  
AJMO = list(OrderedDict(sorted(Counter(list_aj).items(), key = itemgetter(1), reverse = True)).keys())
print(AJMO)
ajmo_2 = dict(zip(AJMO, range(len(AJMO)))) 
print(ajmo_2)  
arr = np.array(ajmo_2)
    
    
def convert_one_hot(number, size_x):
        one_hot = [0] * size_x
        if(number < size_x):
            one_hot[number] = 1
        
        return one_hot


print(convert_one_hot(5, 10))

class my_AdaGrad:
    def __init__(self, global_rate, init_params, data_processing_inst, max_iter, model):
        self.global_rate = global_rate
        self.init_params = init_params
        self.num_stab = 1*e-7
        self.r = [0]*len(init_params) #gradient accumulation
        self.data_processing_ = data_processing_inst
        self.max_iter = max_iter
        self.model = model
        
    def algh_run(self):
        
        for i in range(self.max_iter):
            
            while(true):
                new_epoch , batch_x, batch_y=self.data_processing_.next_minibatch()
 
               if(new_epoch):
                    break

                batch_pred_y = model.outputs(batch_x) 
                for j in range(batch_size):
                    grad + = batch_pred_y[j] - convert_one_hot(batch_y[j], len( batch_pred_y[j]))
                grad /=  batch_size             
                self.r += np.multiply(grad,grad)
                delta_params = self.global_rate*(np.divide(grad, self.num_stab + sqrt(self.r)))
                self.init_params += delta_params
        
        return self.init_params
                

class my_RNN2:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
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
        
    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity
        
        # x - input data (minibatch size x input dimension)
        # h_prev - previous hiddent state (minibatch size x  hidden size)
        # U - input projection matrix (input dimension x hidden size)
        #W - hidden to hidden projection matrix (hidden size x hidden size)
        #b - bias of shape (hidden size x 1)
        sequence_length, input_dimension = x.shape
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        for t in xrange(sequence_length):
            hs[t] = np.tanh(np.dot(self.U, xs[t]) + np.dot(self.W, hs[t-1]) + self.b)
            ys[t] = np.dot(self.V, hs[t]) + self.c
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][target[t], 0])
            
            
        
        #return the new hidden state and a tuple of values needed for backward
        
        return h_current, chache
    
    
    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        minibatch_size, sequence_length, input_dimension = x.shape
        h_current = h0
        h, cache = {}, {}
        
        for m in xrange(minibatch_size):
            x_current = x[m][:][:].reshape(sequence_length, input_dimension)
            h_current, chache = self.rnn_step_forward(x_current, np.zeros((hidden size ,1)))
            h[t] = h_current
            cache[t] = chache
    
        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
    
        return h, cache
    
    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.
    
        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
    
        dh_prev, dU, dW, db = None, None, None, None
        
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh +=np.dot(dhraw, xs[t].T)
            dWhh +=np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparams in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
    
        return dh_prev, dU, dW, db
    
    
    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        dU, dW, db = None, None, None
    
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
    
        return dU, dW, db



        
class my_RNN:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
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
        
    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity
        
        # x - input data (minibatch size x input dimension)
        # h_prev - previous hiddent state (minibatch size x  hidden size)
        # U - input projection matrix (input dimension x hidden size)
        #W - hidden to hidden projection matrix (hidden size x hidden size)
        #b - bias of shape (hidden size x 1)
        minibatch_size, input_dimension = x.shape
        ps = {}
        h_current = np.tanh(np.dot(x, self.U) + np.dot(h_prev, self.W) + self.b)
        ys = np.dot(h_current, self.V) + self.c
        
        for t in xrange(minibatch_size):
            ps[t] = np.exp(ys[t])/ np.sum(np.exp(ys[t]))
            
        chache = (ys, ps)
        
        #return the new hidden state and a tuple of values needed for backward
        
        return h_current, chache
    
    
    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        minibatch_size, sequence_length, input_dimension = x.shape
        h_current = h0
        h, cache = {}, {}
        
        for t in xrange(sequence_length):
            x_current = x[:][t][:].reshape(minibatch_size, input_dimension)
            h_current, chache = self.rnn_step_forward(x_current, h_current, self.U, self.W, self.b)
            h[t] = h_current
            cache[t] = chache
    
        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
    
        return h, cache
    
    def rnn_step_backward(self, grad_next, cache):
        np.clip(grad_next, -5, 5)
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.
    
        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
    
        dh_prev, dU, dW, db = None, None, None, None
        
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh +=np.dot(dhraw, xs[t].T)
            dWhh +=np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparams in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
    
        return dh_prev, dU, dW, db
    
    
    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        dU, dW, db = None, None, None
    
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
    
        return dU, dW, db