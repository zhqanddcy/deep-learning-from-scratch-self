# coding: utf-8
import numpy as np
import os, sys
sys.path.append(os.pardir)

#均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

#交叉熵误差
def cross_entropy_error_n_onehot(y, t):
#    delta = 1e-7
#    return -np.sum(t* np.log(y + delta))    
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#交叉熵误差
def cross_entropy_error(y, t):
#    delta = 1e-7
#    return -np.sum(t* np.log(y + delta))    
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]    
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def function_2(x):
    return x[0]**2 + x[1]**2

#梯度
def numerical_gradient_1(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(x.size)
    for idx in range(x.size):
        tmp_val= x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

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

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
   
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        print(x)
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
    
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

if(__name__ == '__main__'):
#    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
#    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
#    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
    pass







