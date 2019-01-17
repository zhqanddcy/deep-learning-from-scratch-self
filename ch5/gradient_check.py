# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from net_backward_self import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.get_numerical_gradient(x_batch, t_batch)    
grad_bacprop = network.get_backward_gradient(x_batch, t_batch)

# 求各个权重的绝对误差平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_bacprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))


















