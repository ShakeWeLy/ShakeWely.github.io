# 作者：Weakliy
# 创建日期：18/4/2024 下午11:14
# 描述：这个文件实现了一些功能。

# 作者：Weakliy
# 创建日期：18/4/2024 下午9:58
# 描述：这个文件实现了一些功能。
# 作者：Weakliy
# 创建日期：18/4/2024 下午9:28
# 描述：这个文件实现了一些功能。
import numpy as np


class Variable():
    def __init__(self, data):  # 判断类型
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(' {} is not supported '.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None  # creator

    def set_creator(self, func):
        self.creator = func

    def backward(self):  # 将递归变成了循环
        if self.grad is None:  # 初始化
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator:
                funcs.append(x.creator)


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


def square(x):
    f = Square()
    return f(x)


def exp(x):
    return Exp()(x)

# 正向传播
x = Variable(np.array(0.5))
y = square((exp(square(x))))  # 2

# 反向传播
y.grad = np.array(1.0)
y.backward()
print(x.grad)
