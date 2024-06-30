# 作者：Weakliy
# 创建日期：19/4/2024 下午9:38
# 描述：这个文件实现了使其对实现Add类的人来说， DcZero就更好写了。

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
    def __call__(self, *inputs):  # *号的作用：调用具有任意个参数(可变长参数)的雨数
        xs =  [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # 为什么要变成元组？ 方便遍历解包
            ys = (ys,)
        outputs = [Variable(np.array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)  # 记录创造者
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

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
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add(x0, xl):
    return Add()(x0, x1)


x1 = Variable(np.array(2)) # 初始化为列表
x2 = Variable(np.array(3))
f = Add()
ys = f(x1, x2)
print(y.data)
print(type(y))




