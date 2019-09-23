# PyTorch 实现 Pop-Art 算法

PyTorch的实现方式决定了在实现机器学习相关算法时相较于Tensorflow更直观, 相较于Keras自定义更方便。

## 验证

首先, 验证对PyTorch几个核心组件(`backward()`, `step()`, `forward`)的理解是否有误。为此, 设计两组网络结构如下:  

* 由`LowerLayers` + `UpperLayer`组成  
* `UnifiedModel`是以上两部分堆叠而成

`LowerLayers`结构如下:  
```
class LowerLayers(torch.nn.Module):
    def __init__(self, n_in, H):
        super(LowerLayers, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H, H)
        self.hidden3 = torch.nn.Linear(H, H)

    def forward(self, x):
        h_tanh = torch.tanh(self.input_linear(x))
        h_tanh = torch.tanh(self.hidden1(h_tanh))
        h_tanh = torch.tanh(self.hidden2(h_tanh))
        h_tanh = torch.tanh(self.hidden3(h_tanh))
        return h_tanh
```  
`UpperLayer`结构如下:  
```
class UpperLayer(torch.nn.Module):
    def __init__(self, H, n_out):
        super(UpperLayer, self).__init__()
        self.output_linear = torch.nn.Linear(H, n_out)
        torch.nn.init.ones_(self.output_linear.weight)
        torch.nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        y_pred = self.output_linear(x)
        return y_pred
```

需要对比的是这两种网络在初始权重|训练集一致的情况下, 通过各自定义的`loss`, `optimizer`执行相应的`backward`和`step`操作后, 权重的更新是否仍然一致。如果理解无误, 那么两个网络在更新后, 对应层的权重参数应该仍然一致。

### 对应的操作

* `LowerLayers` + `UpperLayer`模式

```
lower_layer = LowerLayers(16, 10)
upper_layer = UpperLayer(10, 1)

opt_lower = torch.optim.SGD(lower_layer.parameters(), lr)
opt_upper = torch.optim.SGD(upper_layer.parameters(), lr)

loss_func = torch.nn.MSELoss()
loss = loss_func(upper_layer(lower_layer(sample_x)), sample_y)

loss.backward()
opt_lower.step()
opt_upper.step()
```

**注意**: 其中两部分分别设置了`optimizer`, 相应传入的参数为各自网络的参数(`opt_lower`为`lower_layer.parameters()`, `opt_upper`为`upper_layer.parameters()`); 而在计算loss时, 仍然按照顺序**连续**计算出预测的y, 与标的的y计算均方差。如此, 在调用`loss.backward()`时将分别计算出loss对两个网络各层参数的梯度。而更新权重时, 需要分别调用各自 `optimizer` 的 `step()` 函数完成参数更新。

* `UnifiedModel` 模式

```
unified_layer = UnifiedModel(16, 10, 1)

loss_func = torch.nn.MSELoss()
loss = loss_func(unified_layer(sample_x), sample_y)

loss.backward()

with torch.no_grad():
    for para in unified_layer.parameters():
        para -= lr * para.grad
```

此处未设置`optimizer`, 手动实现`SGD`的`step`函数。  

**Tips**: 控制PyTorch复现的方式是在调用任何PyTorch相关函数前设置`torch.manual_seed()`。

### 实验结果

对比两组实验的结果分别如下:  

* `LowerLayers` + `UpperLayer` 模式

```
Parameter containing:
tensor([[ -6.5463,  14.0888,  14.7738, -21.8954,  11.8038,  -9.8505,  -1.7764,
         -27.6084,  26.4289,  22.8444]], requires_grad=True)
Parameter containing:
tensor([85.1712], requires_grad=True)
```

* `UnifiedModel` 模式

```
Parameter containing:
tensor([[ -6.5463,  14.0888,  14.7738, -21.8954,  11.8038,  -9.8505,  -1.7764,
         -27.6084,  26.4289,  22.8444]], requires_grad=True)
Parameter containing:
tensor([85.1712], requires_grad=True)
```

其中打印的分别是最后一层的权重和bias完全一致(第一种模式中相当于`UpperLayer`的参数, 第二种模式中相当于`UnifiedModel`最后一层的参数)。

此外, 对于底层(`LowerLayers`)的参数也同样是一致的(结果在此略过)。

### 小结

通过两组实验的对比, 明确了PyTorch中的几个关键函数`backward`, `step`, `forward`的使用方式。与此同时, 将网络拆分为两部分的测试有助于实现Pop-Art算法(其核心思路是将网络最后一层分离, 进行normalization后再做训练)。

### 杂项

[`separate_model.py`](separate_model.py)中commented部分还包括对`backward`的输入参数`grad_tensors`的测试。该参数作为所求梯度的权重对应乘上, 即含义为**被求导变量**的**前序梯度**。与tensorflow中的`tf.gradients`的第三个参数`grad_ys`功能一致。该参数默认为标量1。因为一般而言, 被执行`backward`的变量为标量。而实际也可以对向量执行`backward`, 相应地若要考虑其前序梯度则需要在该参数填上与向量维度一致的`Tensor`。