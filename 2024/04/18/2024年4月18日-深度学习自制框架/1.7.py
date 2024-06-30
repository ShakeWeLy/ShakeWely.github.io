# 作者：Weakliy
# 创建日期：18/4/2024 下午9:58
# 描述：这个文件实现了一些功能。
# 作者：Weakliy
# 创建日期：18/4/2024 下午9:28
# 描述：这个文件实现了一些功能。
import numpy as np


class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  # creator

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function():
    def __call__(self, input):
        x = input.data
        # x = x**2
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 记录创造者
        self.input = input
        self.output = output  # 记录自己也是创造者动态建立 " 连接"这 一 机制的核心 。 为了兼顾下一个步骤，将输山设置为实例变量t output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backword(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy  #
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
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
assert y.creator == C # 断言语句，用于在程序中进行条件检查。当 assert 后面的条件为 False 时，程序将会触发 AssertionError 异常，导致程序中断
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# 初始化节点和变量
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))

# 前向传播
a = A(x)
b = B(a)
y = C(b)

# 反向传播
y.grad = np.array(1.0)
y.backward()

# 打印输入变量的梯度
print(x.grad)
