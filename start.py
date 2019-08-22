from cat.lr_utils import load_dataset
from cat.model import *
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
x_orig, y_orig, x_test, y_test, classes = load_dataset()
x_orig = x_orig.transpose(0, 3, 1 ,2).astype("float")
# 1 数据提取
val_index = np.random.choice(range(x_orig.shape[0]), int(x_orig.shape[0] * 0.2))
x_val,y_val = x_orig[val_index],y_orig.T[val_index]
x = x_orig
y = y_orig.T
print(x.shape,y.shape,x_val.shape,y_val.shape)
# 2 参数确定
num_classes = 2
num_filters = 2
weight_scale = 1e-3
reg = 0# 这里没有添加正则项
filter_size = 7
max_iter = 1000
batch_size = 20
learning_rate = 1e-2
loss_history = []
val_acc_history = []
num_train = x.shape[0]
best_val_acc = 0
# 3 初始化
W1, b1, W2, b2, W3, b3 = init(weight_scale, num_filters, num_classes, filter_size, x[0])
params = {}
params["W1"], params["b1"], params["W2"], params["b2"], params["W3"], params["b3"] = W1, b1, W2, b2, W3, b3

# 4 迭代
for i in range(max_iter):
    num_iterations = int(max(num_train / batch_size, 1))
    # 分批次训练
    best_params = {}
    for k, v in params.items():
        best_params[k] = v.copy()
    for t in range(num_iterations):
        batch_mask = np.random.choice(num_train, batch_size)
        # 分批次
        X_batch = x[batch_mask]
        y_batch = y[batch_mask]
        start = time.time()
        # 100张图片一次传播消耗81.05327773094177
        loss, grads = cnn(X_batch, y_batch, W1, b1, W2, b2, W3, b3, filter_size, reg)
        print("第%d次迭代消耗%.2f s"%(i * num_iterations + t + 1 ,time.time()-start))
        loss_history.append(loss)
        # 更新参数
        for p, w in params.items():
            dw = grads[p]
            params[p] = w - learning_rate * dw
        val_acc = check_accuracy(x_val, y_val, params, filter_size)
        val_acc_history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            for k, v in params.items():
                best_params[k] = v.copy()
        print("次数:%d,损失:%.5f,验证准确率:%.4f"%(i * num_iterations + t + 1, loss, val_acc))
    params = best_params

# 5 验证结果
check_accuracy(x_test, y_test, params, filter_size, num_samples=batch_size)


