# -*- coding: utf-8 -*-


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


















