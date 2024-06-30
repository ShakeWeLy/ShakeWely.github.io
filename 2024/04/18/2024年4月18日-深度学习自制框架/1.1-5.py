# 作者：Weakliy
# 创建日期：18/4/2024 下午8:25
# 描述：这个文件实现了一些功能。
import numpy as np


class Variable():
    def __init__(self, data):
        self.data = data


# data1 = np.array(3.0)
# x = Variable(data1)
# print(x.data)


class Function():
    def __call__(self, input):  # 使这个类的实例就可以像函数一样被调用
        x = input.data
        # x = x**2
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()  # 这意味着在Function类的子类中，如果没有重写forward方法，调用forward方法会引发NotImplementedError异常，提示该方法需要在子类中实现。


# f = Function()
# y = f(x)
# print(type(y))
# print(y.data)


class Square(Function):
    def forward(self, x):
        return x ** 2


# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(type(y))
# print(y.data)
# print("--------------------------------")


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# A = Square()
# B = Exp()
# C = Square()
# x = Variable(np.array(2.0))
# a = A(x)
# b = B(a)
# y = C(b)
# print(y.data)
# print("--------------------------------")


# 利用微小的差值获得函数变化量的方法叫作 数值微分
# 前向差分近似不如中心差分近似
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2))
dy = numerical_diff(f=f, x=x)
print(dy)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f,x)
print(dy)
# print("--------------------------------")


# 使用数值微分的结果检查反向传播的实现是否正确 。 这种做法叫作 梯度检验

