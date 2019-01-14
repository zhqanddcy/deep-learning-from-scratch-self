# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
import perceptionFunc

def loadMnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    pass

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    pass

def mnist_show():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img)
    pass

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
    pass

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = perceptionFunc.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = perceptionFunc.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = perceptionFunc.softmax(a3)
    return y

def test0():
    x, t = get_data()
    print(len(x), len(t))
    network = init_network()
    
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        print(y)
        p = np.argmax(y)
        if(p == t[i]):
            accuracy_cnt += 1
    print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))
    pass

def test1():
    x, t = get_data()
    print(len(x), len(t))
    network = init_network()
    
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)        
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])        
    print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))
    pass

if(__name__ == '__main__'):
    #test0()
    test1()
    #mnist_show()
    pass




