import numpy as np


# 卷积函数
def conv_fun(cache):
    x, w, b = cache["a"], cache["w"], cache["b"]
    pad, stride = cache["pad"], cache["stride"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # numpy提供的可以填充0的api,constant代表用一样的值填充 前两维不填,后两维各自填充0 pad行
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = int((H + 2 * pad - HH) / stride) + 1
    W_new = int((W + 2 * pad - WW) / stride) + 1
    s = stride
    out = np.zeros((N, F, H_new, W_new))
    for i in range(N):  # ith image
        for f in range(F):  # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    out[i, f, j, k] = np.sum(x_padded[i, :, j * s:(HH + j * s), k * s:(WW + k * s)] * w[f]) + b[f]
    return out


# Relu函数
def Relu(x):
    return np.maximum(0, x)


# 前向池化
def max_pool_forward(cache):
    x, HH, WW, s = cache["net"], cache["HH"], cache["WW"], cache["s"]
    N, C, H, W = x.shape
    H_new = 1 + int((H - HH) / s)
    W_new = 1 + int((W - WW) / s)
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    # 定位到某个窗口
                    window = x[i, j, k * s:HH + k * s, l * s:WW + l * s]
                    # 找到该窗口的最大值，然后赋值
                    out[i, j, k, l] = np.max(window)
    return out


# 全连接
def fc(net, w, b):
    N = net.shape[0]
    # 把每个像素提取出来
    x_row = net.reshape(N, -1)
    out = np.dot(x_row, w) + b
    return out


# softmax损失函数
def softmax_loss(x, y):
    y = y.reshape(y.shape[0])
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


# 反向池化
def max_pool_backward(dout, cache):
    x, HH, WW, s = cache["net"], cache["HH"], cache["WW"], cache["s"]
    N, C, H, W = x.shape
    H_new = 1 + int((H - HH) / s)
    W_new = 1 + int((W - WW) / s)
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k * s:HH + k * s, l * s:WW + l * s]
                    m = np.max(window)
                    # fp [1 3;2 2]-->[3] bp [3] --> [0 3;0 0]
                    dx[i, j, k * s:HH + k * s, l * s:WW + l * s] = (window == m) * dout[i, j, k, l]

    return dx


# 反向relu
def relu_backward(dout, x):
    dx = dout
    dx[x <= 0] = 0
    return dx


# 反向全连接
def fc_backward(dout, cache):
    x, w, b = cache["a"], cache["w"], cache["b"]
    dx = np.dot(dout, w.T)  # (N,D)
    dx = np.reshape(dx, x.shape)  # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)  # (N,D)
    dw = np.dot(x_row.T, dout)  # (D,M)
    db = np.sum(dout, axis=0, keepdims=True)  # (1,M)
    return dx, dw, db


# 反向全连接relu
def fc_relu_backward(dout, f1, cache):
    da = relu_backward(dout, f1)
    dx, dw, db = fc_backward(da, cache)
    return dx, dw, db


# 反向卷积 dout 上一层传下来的梯度
def conv_backward(dout, cache):
    x, w, b = cache["a"], cache["w"], cache["b"]
    pad, stride = cache["pad"], cache["stride"]
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + int((H + 2 * pad - HH) / stride)
    W_new = 1 + int((W + 2 * pad - WW) / stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):  # ith image
        for f in range(F):  # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    window = x_padded[i, :, j * s:HH + j * s, k * s:WW + k * s]
                    # db = dout //  dw=dout*x // dx = dout*w
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j * s:HH + j * s, k * s:WW + k * s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad + H, pad:pad + W]

    return dx, dw, db


#  反向卷积relu池化
def conv_relu_pool_backward(dout, cache, cache_pool):
    cv1 = cache_pool["cv"]
    ds = max_pool_backward(dout, cache_pool)
    da = relu_backward(ds, cv1)
    dx, dw, db = conv_backward(da, cache)
    return dx, dw, db



