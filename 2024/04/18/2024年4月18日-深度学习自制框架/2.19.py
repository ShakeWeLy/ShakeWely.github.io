# 作者：Weakliy
# 创建日期：20/4/2024 下午5:15
# 描述：这个文件实现了一些功能。

# 作者：Weakliy
# 创建日期：20/4/2024 下午4:02
# 描述：这个文件实现了减少内存使用量的模式。


import numpy as np
import weakref


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(' {} is not supported '.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # generation
        self.name = name

    def __len__(self):
        return len(self.data)
    @property  # ?
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size

    # 写repr一方法来自定义print函数输出的字符串
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

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


x = Variable(np.array([1, 2, 3]))
print("x:", x)
print("x.name:", x.name)
print("x.size:", x.size)
print("x.ndim:", x.ndim)
print("x.shape:", x.shape)
print("len(x)",len(x))
print("------------")

x = Variable(None)
print("x:", x)
print("x.name:", x.name)
# print("x.size:", x.size)
# print("x.ndim:", x.ndim)
# print("x.shape:", x.shape)
# print("len(x)",len(x))
print("------------")


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print("x:", x)
print("x.name:", x.name)
print("x.size:", x.size)
print("x.ndim:", x.ndim)
print("x.shape:", x.shape)
print("len(x)",len(x))
print("------------")






