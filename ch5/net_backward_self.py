# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from layer_naive import *
from collections import OrderedDict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
   
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#交叉熵误差
def cross_entropy_error(y, t):
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)        
        x[idx] = tmp_val 
        it.iternext()
    return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    #x:输入数据， t:监督数据
    def numerical_loss(self, x, t):
        y = self.predict(x)        
        return cross_entropy_error(y, t)
    
    #x:输入数据， t:监督数据
    def backward_loss(self, x, t):
        y = self.predict(x)        
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)       
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def get_numerical_gradient(self, x, t):
        loss_W = lambda W: self.numerical_loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def get_backward_gradient(self, x, t):
        # forward
        self.backward_loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
        pass


class DeepLearningSimple:
    def __init__(self):
        (self.x_train, self.t_train), (self.x_test, self.t_test) = load_mnist(normalize=True, one_hot_label=True)
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        #超参数
        self.iters_num = 10000
        self.train_size = self.x_train.shape[0]
        self.batch_size = 100
        self.learning_rate = 0.1
        #平均每个epoch的重复次数
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        pass
    
    def learningSimple(self):
        for i in range(self.iters_num):
            #获取mini-batch
            batch_mask = np.random.choice(self.train_size, self.batch_size)    
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            #计算梯度    
            #grad = self.network.get_numerical_gradient(x_batch, t_batch)    
            grad = self.network.get_backward_gradient(x_batch, t_batch)
            #更新参数
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.network.params[key] -= self.learning_rate * grad[key]
            #记录学习过程
            loss = self.network.backward_loss(x_batch, t_batch)
            self.train_loss_list.append(loss)
            #计算每个epoch的识别精度
            if i%self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print('train acc, test acc | ' + str(test_acc) + ',' + str(test_acc))
        pass
    
    def showAccuracy(self):
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(self.train_acc_list))
        plt.plot(x, self.train_acc_list, label='train acc')
        plt.plot(x, self.test_acc_list, label='test acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()
        pass

if(__name__ == '__main__'):
    netSelf = DeepLearningSimple()
    netSelf.learningSimple()
    netSelf.showAccuracy()
    pass


