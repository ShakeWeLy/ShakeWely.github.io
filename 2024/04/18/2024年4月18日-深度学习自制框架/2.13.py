# 作者：Weakliy
# 创建日期：19/4/2024 下午10:04
# 描述：这个文件实现了一些功能。


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
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs): # gxs和f.inputs的每个元素都是一一对应的
                x.grad = gx  # 2.3 相同的变量会被覆盖

                # 2.14
                # if x.grad is None:
                #     x.grad = gx
                # else:
                #     x.grad = x.grad + gx
                # print(type(x.grad))
                # print(x.grad)

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

# 2.13
x = Variable(np.array(2))
y = Variable(np.array(3))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)


# 2.14.1
x = Variable(np.array(3))
y = add(x,x)
y.backward()
print(x.grad)  # 1 实际为2



