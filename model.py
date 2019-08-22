from cat.util import *


def init(weight_scale, num_filters, num_classes, filter_size, test):
    C, H, W = test.shape
    W1 = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    b1 = np.zeros(num_filters)
    W2 = weight_scale * np.random.randn(int(num_filters * H * W / 4), 20)
    b2 = np.zeros(20)
    W3 = weight_scale * np.random.randn(20, num_classes)
    b3 = np.zeros(num_classes)
    return W1, b1, W2, b2, W3, b3


def cnn(x,y,W1, b1, W2, b2, W3, b3, filter_size, reg):
    # conv1
    pad = int((filter_size - 1) / 2)  # s=1 时，p = (f-1)/2
    stride = 1
    cache1 = {"a": x, "w": W1, "b": b1, "pad": pad, "stride": stride}
    cv1 = conv_fun(cache1)
    net = Relu(cv1)
    print("卷积后的形状:",net.shape)

    # poool1
    HH, WW, s = 2, 2, 2
    cache_pool = {"cv": cv1, "net": net, "HH": HH, "WW": WW, "s": s}
    a1 = max_pool_forward(cache_pool)
    cache2 = {"a": a1, "w": W2, "b": b2}
    print("池化后的形状:",a1.shape)
    # fc2
    f1 = fc(a1, W2, b2)
    a2 = Relu(f1)
    cache3 = {"a": a2, "w": W3, "b": b3}
    # fc3
    scores = fc(a2, W3, b3)
    # bp
    data_loss, dscores = softmax_loss(scores, y)  # 损失值 分类结果
    da2, dW3, db3 = fc_backward(dscores, cache3)
    da1, dW2, db2 = fc_relu_backward(da2, f1, cache2)
    dX, dW1, db1 = conv_relu_pool_backward(da1, cache1, cache_pool)

    # 添加正则项的导数 1/2*reg*(w)^2-->w
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3

    reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3])
    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return loss,grads

def fpp(x,y,params, filter_size):
    W1, b1, W2, b2, W3, b3 = params["W1"], params["b1"], params["W2"], params["b2"], params["W3"], params["b3"]
    pad = int((filter_size - 1) / 2)  # s=1 时，p = (f-1)/2
    stride = 1
    cache1 = {"a": x, "w": W1, "b": b1, "pad": pad, "stride": stride}
    cv1 = conv_fun(cache1)
    net = Relu(cv1)
    HH, WW, s = 2, 2, 2
    cache_pool = {"cv": cv1, "net": net, "HH": HH, "WW": WW, "s": s}
    a1 = max_pool_forward(cache_pool)
    f1 = fc(a1, W2, b2)
    a2 = Relu(f1)
    scores = fc(a2, W3, b3)
    data_loss, dscores = softmax_loss(scores, y)  # 损失值 分类结果
    return dscores

def check_accuracy(X, y, params,filter_size ):
    scores = fpp(X,y,params,filter_size)
    y_pred = np.argmax(scores, axis=1)
    count = 0
    # print(y_pred)
    # print(y)
    for i in range(len(y)):
        if y_pred[i]==y[i]:
            count +=1
    acc = count/len(y)
    return acc
