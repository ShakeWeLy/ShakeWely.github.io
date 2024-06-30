# 作者：Weakliy
# 创建日期：20/4/2024 下午4:02
# 描述：这个文件实现了减少内存使用量的模式。


import numpy as np
import weakref


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(' {} is not supported '.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # generation

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # generation


    def backward(self, retain_grad=False):  # 将递归变成了循环
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []  # generation
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
            funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)


        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator:
                    # funcs.append(x.creator)
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # 弱引用 引用计数将变为0，导数的数据会从内存中被删除


    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs =  [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs]  #

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backword(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0].data
        gx = 2 * x * gy  #
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)


def add(x0, x1):
    return Add()(x0, x1)


# 18.1
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print("y.grad:", y.grad)
print("t.grad:", t.grad)
print("x0.grad:", x0.grad)
print("xl.grad:", x1.grad)
# todo:设置前后对比


# 18.2
Config.enable_backprop = True
x = Variable(np.ones((16, 166, 166)))
y = square(square(square(x)))
y.backward()
print(x.grad)
Config.enable_backprop = False
x = Variable(np.ones((166, 166, 166)))
y = square(square(square(x)))


# with using_config("enable_backprop", False):
#     x = Variable(np.array(2.0))
#     y = square(x)

import contextlib
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)  #
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

# with using_config('enable_backprop', False):
#     x = Variable(np.array(2.0))
#     y = square(x)

def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

