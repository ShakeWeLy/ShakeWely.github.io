# 作者：Weakliy
# 创建日期：18/4/2024 下午9:28
# 描述：这个文件实现了一些功能。
import numpy as np


class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None  # 导数值


class Function():
    def __call__(self, input):  # 使这个类的实例就可以像函数一样被调用
        x = input.data
        # x = x**2
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()  # 这意味着在Function类的子类中，如果没有重写forward方法，调用forward方法会引发NotImplementedError异常，提示该方法需要在子类中实现。

    def backword(self, x):
        raise  NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backword(self, gy):
        x = self.input.data
        gx = 2 * x * gy  #
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backword(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 正向传播
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 反向传播
y.grad = np.array(1.0)
b.grad = C.backword(y.grad)
a.grad = B.backword(b.grad)
x.grad = A.backword(a.grad)
print(x.grad)


