### 1 Softmax

(a) softmax常数不变性

证明softmax不受输入的常数偏移影响，即
$$
softmax(x)=softmax(x+c)
$$
解:
$$
(softmax(x+c))_i=\frac{e^{x_i+c}}{\sum_{j}e^{x_j+c}}=\frac{e^{x_i} \times e^c}{e^c \times \sum_je^{x_j}}=\frac{e^{x_i}}{\sum_je^{x_j}}
$$
(b) Python实现

```pyth
def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        max_x = np.max(x, axis=1).reshape((x.shape[0], 1))
        x = x - max_x
        x = np.exp(x)
        x = x * (1.0/np.sum(x, axis=1)).reshape((x.shape[0], 1))


        #raise NotImplementedError
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        max_x = np.max(x)
        x = x - max_x
        x = np.exp(x)
        x = x/np.sum(x)
        #raise NotImplementedError
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x
```

2 神经网络基础

(a) sigmoid梯度
$$
\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{1+e^x}
$$

$$
\frac{\partial \sigma(x)}{\partial x}=\frac{e^x \times (1+e^x)-(e^x \times e^x)}{(1+e^x)^2}=\frac{e^x+(e^x \times e^x)-(e^x \times e^x)}{(1+e^x)^2}=\frac{e^x}{1+e^x} \times \frac{1}{1+e^x}\\
=\sigma(x) \times \sigma(-x)=\sigma(x) \times (1-\sigma(x))
$$

(b) 交叉熵损失函数的梯度

请推导交叉熵损失函数关于softmax输入$\theta$的梯度, 记假设函数为
$$
\hat{y}=softmax(\theta)
$$
交叉熵损失函数为:
$$
\begin{equation} 
    CE(y,\hat{y}) = – \sum_{i} y_{i} \times log(\hat{y_{i}}) 
\end{equation}
$$
其中$y$是one-hot向量, $\hat{y}$是概率向量。

解: 

记$S$为softmax函数, 引入下列记号:
$$
f_i=e^{\theta_i} \\
g_i=\sum^K_{k=1}e^{\theta_k} \\
S_i=\frac{f_i}{g_i} \\
\frac{\partial S_i}{\partial \theta_j}=\frac{f'_ig_i-g'_if_i}{g_i^2}
$$
这里的下标表示取标量分别求导, 当$i=j$时:
$$
f'_i=f_i;g'_i=e^{\theta_j} \\
\frac{\partial S_i}{\partial \theta_j}=\frac{e^{\theta_i}\sum_ke^{\theta_k}-e^{\theta_j}e^{\theta_i}}{(\sum_ke^{\theta_k})^2}=\frac{e^{\theta_i}}{\sum_ke^{\theta_k}} \times \frac{\sum_ke^{\theta_k}-e^{\theta_j}}{\sum_ke^{\theta_k}}=S_i \times (1-S_i)
$$
当$i \not = j$时:
$$
f'_i=0;g'_i=e^{\theta_j} \\
\frac{\partial S_i}{\partial \theta_j}=\frac{0-e^{\theta_i}e^{\theta_j}}{(\sum_ke^{\theta_k})^2}=-\frac{e^{\theta_i}}{\sum_ke^{\theta_k}} \times \frac{e^{\theta_j}}{\sum_ke^{\theta_k}}=-S_i \times S_j
$$
接下来正式推导:
$$
\frac{\partial CE}{\partial \theta_i}=-\sum_ky_k \frac{\partial logS_k}{\partial \theta_i}=-\sum_ky_k\frac{1}{S_k}\frac{\partial S_k}{\partial \theta_i}
=-y_i(1-S_i)-\sum_{k \not = i}y_k \frac{1}{S_k}(-S_i \times S_k) \\
=-y_i(1-S_i)+\sum_{k \not = i}y_kS_i=-y_i+y_iS_i+\sum_{k \not = i}y_kS_i\\
=S_i(\sum_ky_k)-y_i
$$
由于概率归一化$\sum_ky_k=1$:
$$
\frac{\partial CE}{\partial \theta_i}=S_i-y_i
$$
