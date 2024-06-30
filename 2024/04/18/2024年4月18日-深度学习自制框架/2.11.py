# 作者：Weakliy
# 创建日期：19/4/2024 下午9:38
# 描述：这个文件实现了使其可以处理nJ变长的输入和输出。

import numpy as np


class Variable:
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


class Function:
    def __call__(self, inputs):
        xs =  [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(np.array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)  # 记录创造者
        self.inputs = inputs
        self.outputs = outputs  # 记录自己也是创造者动态建立 " 连接"这 一 机制的核心 。 为了兼顾下一个步骤，将输山设置为实例变量t output
        return outputs

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


class Add(Function) :
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)  # 为什么要括号？


xs = [Variable(np.array(2)), Variable(np.array(3))] # 初始化为列表
f = Add()
ys = f(xs)  # ys是无组
y = ys[0]
print(y.data)
print(ys)



