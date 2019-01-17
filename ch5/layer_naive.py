# -*- coding: utf-8 -*-

import numpy as np

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

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forword(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
 
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out 
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out)  * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b        
        self.x = None
        self.original_x_shape = None        
        self.dW = None
        self.db = None

    def forward(self, x):        
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)        
        dx = dx.reshape(*self.original_x_shape)  
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    
def buy_apple():
    apple = 100
    apple_num = 2
    tax = 1.1
    #layer
    multi_app_layer = MulLayer()
    multi_tax_layer = MulLayer()
    #forward
    apple_price = multi_app_layer.forword(apple, apple_num)
    price = multi_tax_layer.forword(apple_price, tax)    
    print(price)
    #backward
    dprice = 1
    dapple_price, dtax = multi_tax_layer.backward(dprice)
    dapple, dapple_num = multi_app_layer.backward(dapple_price)
    print(dapple, dapple_num, dtax)
    pass

def buy_apple_orange():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    #layer
    multi_app_layer = MulLayer()
    multi_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    multi_tax_layer = MulLayer()
    #forward
    apple_price = multi_app_layer.forword(apple, apple_num)
    orange_price = multi_orange_layer.forword(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = multi_tax_layer.forword(all_price, tax)    
    print(price)
    #backward
    dprice = 1
    dall_price, dtax = multi_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = multi_orange_layer.backward(dorange_price)
    dapple, dapple_num = multi_app_layer.backward(dapple_price)
    print(dapple, dapple_num, dorange, dorange_num, dtax)
    pass

if(__name__ == '__main__'):
    buy_apple()
    buy_apple_orange()
    pass


















