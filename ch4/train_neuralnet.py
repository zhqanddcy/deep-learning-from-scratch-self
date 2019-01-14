# coding: utf-8
import numpy as np

#均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

#交叉熵误差
def cross_entropy_error(y, t):
#    delta = 1e-7
#    return -np.sum(t* np.log(y + delta))
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size










