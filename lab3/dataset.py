#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:33:09 2018

@author: branko
"""
import os, sys
import re
import numpy as np
from collections import Counter
from collections import OrderedDict
from operator import itemgetter
import numpy as np
  
from my_ADAGRAD import My_ADAGRAD
from my_RNN import my_RNN2
root = '/home/branko/Documents/lab3/dlunizg.github.io/code/lab3/data'
output_destination = 'selected_conversations.txt'

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
            print (data)
            #count and sort most frequent characters
            #self.sorted chars contains just the characters ordered descending by frequnecies
            self.char2id  = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
            #reverse the mapping
            self.id2char = {k:v for v,k in self.char2id.items()}
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
            decoded_list += [self.id2char[i]]
            
        return decoded_list
    
    def get_num_chars(self):
        return len(self.char2id.keys())
    def get_num_batches(self):
        return self.num_batches
    
    def create_minibatches(self):
        self.num_batches = int(len(self.x) // (self.batch_size * self.sequence_length)) # calculate the number of batches
        self.batches_x = []
        self.batches_y = []
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
        new_epoch = False
        if self.iter >= self.num_batches:
            new_epoch = True
            self.iter = 0
        else:
            batch_x = self.batches_x[self.iter]
            batch_y = self.batches_y[self.iter]
            self.iter +=1
        
        return new_epoch, batch_x, batch_y
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        

def main():
    # uses hardcoded paths
    hidden_size = 100
    
    
    output = os.path.join(root, output_destination)
    my_data_processing = data_processing(output, 200,30)    
    my_data_processing.preprocess(output)
    my_data_processing.create_minibatches()
    
    sequence_length, vocab_size, learning_rate = 30, my_data_processing.get_num_chars(), 1e-1
    
    h0 = np.zeros((hidden_size, 1))
    my_rnn = my_RNN2(hidden_size, sequence_length, vocab_size, learning_rate)
    init_params = {'W': my_rnn.get_W(), 'U': my_rnn.get_U(), 'V': my_rnn.get_V(), 'b': my_rnn.get_b(), 'c': my_rnn.get_c()}
    adagrad = My_ADAGRAD(learning_rate, init_params)
    num_epochs = 100
    U_new = np.loadtxt('U.txt')
    W_new = np.loadtxt('W.txt')
    V_new = np.loadtxt('V.txt')
    b_new = np.loadtxt('b.txt').reshape(-1,1)
    c_new = np.loadtxt('c.txt').reshape(-1,1)

    my_rnn.set_W(W_new)
    my_rnn.set_U(U_new)
    my_rnn.set_V(V_new)
    my_rnn.set_b(b_new)
    my_rnn.set_c(c_new)
    while(True):
        
        new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
        
        if(new_epoch):
            
            new_epoch, batch_x, batch_y = my_data_processing.next_minibatch()
            num_epochs -= 1
            
            if(num_epochs == 0):
                
                np.savetxt('W.txt',W_new,fmt='%.6f')
                np.savetxt('U.txt',U_new,fmt='%.6f')
                np.savetxt('V.txt',V_new,fmt='%.6f')
                np.savetxt('b.txt',b_new,fmt='%.6f')
                np.savetxt('c.txt',c_new,fmt='%.6f')
                break
        
        else:
            
            h_m, p_m, y_m, x_one_hot, loss_all = my_rnn.rnn_forward(batch_x, batch_y, h0)
            cache_m = {'hs': h_m, 'ps': p_m, 'xs': x_one_hot}
            dU, dW, db , dc, dV=my_rnn.rnn_backward(batch_x, batch_y, cache_m)
            old_params = {'W': my_rnn.get_W(), 'U': my_rnn.get_U(), 'V': my_rnn.get_V(), 'b': my_rnn.get_b(), 'c': my_rnn.get_c()}
            grads = {'dW': dW, 'dU': dU, 'dV': dV, 'db': db, 'dc': dc}

            update_params = adagrad.run_algh(old_params, grads)

            U_new = update_params['U']
            W_new = update_params['W']
            V_new = update_params['V']
            b_new = update_params['b']
            c_new = update_params['c']
            
            my_rnn.set_W(W_new)
            my_rnn.set_U(U_new)
            my_rnn.set_V(V_new)
            my_rnn.set_b(b_new)
            my_rnn.set_c(c_new)
            
            print("loss is: ", loss_all)

            
            
            

def main_pred():
    # uses hardcoded paths
    hidden_size = 100
    
    
    output = os.path.join(root, output_destination)
    my_data_processing = data_processing(output, 200,30)    
    my_data_processing.preprocess(output)
    my_data_processing.create_minibatches()
    
    sequence_length, vocab_size, learning_rate = 30, my_data_processing.get_num_chars(), 1e-1
    
    my_rnn = my_RNN2(hidden_size, sequence_length, vocab_size, learning_rate)

    U_new = np.loadtxt('U.txt')
    W_new = np.loadtxt('W.txt')
    V_new = np.loadtxt('V.txt')
    b_new = np.loadtxt('b.txt').reshape(-1,1)
    c_new = np.loadtxt('c.txt').reshape(-1,1)

    my_rnn.set_W(W_new)
    my_rnn.set_U(U_new)
    my_rnn.set_V(V_new)
    my_rnn.set_b(b_new)
    my_rnn.set_c(c_new)
    batch_size = 200
    
    my_rnn.rnn_pred(my_data_processing, num_predicitons=100, last=batch_size*sequence_length -1)

if __name__ == '__main__':
#    main()     
    main_pred()