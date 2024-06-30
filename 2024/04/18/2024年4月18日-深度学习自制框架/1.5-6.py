# 作者：Weakliy
# 创建日期：19/4/2024 下午10:42
# 描述：这个文件实现了一些复杂计算图。


import numpy as np


class Variable:
    def __init__(self, data):  # 判断类型
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(' {} is not supported '.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # generation

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()  # 不允许包含重复的元素，并且不保留元素的顺序

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
            funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        # funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs): # gxs和f.inputs的每个元素都是一一对应的
                # x.grad = gx  # 2.3 相同的变量会被覆盖

                # 2.4
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # print(type(x.grad))
                # print(x.grad)

                if x.creator:
                    # funcs.append(x.creator)
                    add_func(x.creator)

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs =  [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])  #

        for output in outputs:
            output.set_creator(self)
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
    f = Square()
    return f(x)

def add(x0, x1):
    return Add()(x0, x1)


# 定义变量 x
x = Variable(np.array(2))
# 对 x 进行操作
a = square(x)
y = add(square(a), square(a))
# 计算 y 的梯度
y.backward()
# 打印结果
print("y:", y.data)
print("x:", x.data)
print("x.grad:", x.grad)


