#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:16:17 2018

@author: branko
"""
import os, sys
import re
import numpy as np
from collections import Counter
from collections import OrderedDict
from operator import itemgetter 

class My_ADAGRAD:
    def __init__(self, learning_rate, init_params):
        self.learning_rate = learning_rate
        self.stability = 1e-8
        
        U = init_params['U']
        W = init_params['W']
        V = init_params['V']
        b = init_params['b']
        c = init_params['c']
        
        self.mU, self.mW, self.mV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
        self.mb, self.mc = np.zeros_like(b), np.zeros_like(c)
        
    def run_algh(self,params, grads):
        
        U = params['U']
        W = params['W']
        V = params['V']
        b = params['b']
        c = params['c']
        
        dU = grads['dU']
        dW = grads['dW']
        dV = grads['dV']
        db = grads['db']
        dc = grads['dc']
        

        for param, dparam, mem in zip([U, W, V, b, c], 
                                      [dU, dW, dV, db, dc], 
                                      [self.mU, self.mW, self.mV, self.mb, self.mc]):
            mem += np.multiply(dparam,dparam)
            param += -self.learning_rate * np.divide(dparam , np.sqrt(mem + self.stability))

        update_params = {'W': W, 'U': U, 'V': V, 'b': b, 'c': c}
        
        return update_params