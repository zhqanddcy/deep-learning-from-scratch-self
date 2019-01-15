# -*- coding: utf-8 -*-

import os,sys
sys.path.append(os.pardir)
import train_neuralnet
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = train_neuralnet.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = train_neuralnet.softmax(a2)
        return y
    
    #x:输入数据， t:监督数据
    def loss(self, x, t):
        y = self.predict(x)        
        return train_neuralnet.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = train_neuralnet.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = train_neuralnet.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = train_neuralnet.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = train_neuralnet.numerical_gradient(loss_W, self.params['b2'])
        return grads


