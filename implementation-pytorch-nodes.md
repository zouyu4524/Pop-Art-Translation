# PyTorch 实现 Pop-Art 算法

PyTorch的实现方式决定了在实现机器学习相关算法时相较于Tensorflow更直观, 相较于Keras自定义更方便。

<!-- MarkdownTOC autolink="true" -->

- [验证](#%E9%AA%8C%E8%AF%81)
    - [对应的操作](#%E5%AF%B9%E5%BA%94%E7%9A%84%E6%93%8D%E4%BD%9C)
    - [实验结果](#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C)
    - [小结](#%E5%B0%8F%E7%BB%93)
    - [杂项](#%E6%9D%82%E9%A1%B9)
- [算法实现](#%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0)
    - [算法概述](#%E7%AE%97%E6%B3%95%E6%A6%82%E8%BF%B0)
    - [实验结果](#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C-1)

<!-- /MarkdownTOC -->


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

## 算法实现

[论文](https://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf)中共给出了两个算法: 1) Pop-Art 和 2) Normalized SGD, 并证明了两个算法的**等效性**。  
本节分别实现这两种算法并验证其等效性。

### 算法概述

* **算法1: Pop-Art**

![Pop-Art](https://user-images.githubusercontent.com/16682999/64672136-33788080-d49d-11e9-9771-bc07a48e99b6.png)

如图, 算法1中的关键在于对`UpperLayer`(即参数`W`和`b`)的**两次**更新, 其中第一次是自适应scale(即ART)的同时保留已训练样本的结果(即POP)。第二次是优化算法下的参数更新。需要注意的是: 第一次的更新仅改变参数值(`.data`), 不改变参数的梯度(`.grad`)。另一方面, 有了上一节中拆分网络的经验, 算法1中对两层网络的梯度计算也可以归并为一步`loss.backward()`计算。再分别通过各自网络的优化算子(`optimizer`)调用`step()`更新参数。

* **算法2: Normalized SGD**

![Normalized-SGD](https://user-images.githubusercontent.com/16682999/64753079-747f9c00-d554-11e9-9bb3-e93407d09307.png)

如图, 算法2的关键是通过缩放后的目标差值和`UpperLayer`的权重参数更新`LowerLayers`的参数。为了实现这一点, 我们需要**暂时**地放缩`UpperLayer`的权重数值(`.data`)以及目标差值。同样地, 调用`loss.backward()`将计算两层网络的梯度, 其中`UpperLayer`的梯度不受缩放的影响(因为梯度中无`W`), 但权重数值本身被缩放, 需要在完成梯度计算后**还原**。

**注意**: *这里有一个重要的trick。即缩放|还原`UpperLayer`的权重数值时务必使用`in-place`的方式: 即`W.data *= scalar`这样, 原因在于这种操作方式不会变更`W.data`在内存中的物理地址, 因而计算梯度(`backward`)时, 通过计算图寻到的`W`, 其中的`W.data`便是更改后的结果。否则, 如果采用非`in-place`的更新方式, 如`W.data = W.data * scalar`, 那么实际上新开辟了一个内存地址, 存储更新后的`W.data`, 而计算梯度时, 通过计算图找到的`W`中的`W.data`实际上仍然是原来的`W.data`。* <sup>[1]</sup>


### 实验结果

分别实现以上两种算法后, 进行对比测试: 两种算法分别以相同的随机数种子初始化两层网络权重, 对比单步的网络更新后`LowerLayers`网络的权重变化。结果如下:  

* 以`LowerLayers`各层的bias对比来看
```
# 算法1: Pop-Art
tensor([-0.1496, -0.1486, -0.2989,  0.0683, -0.1775, -0.2820,  0.2773, -0.2053,
        -0.0360,  0.0906])
tensor([ 0.2812, -0.0810,  0.1392,  0.2818,  0.1046,  0.3161,  0.1640,  0.1966,
        -0.1107,  0.1517])
tensor([-0.0687,  0.0421,  0.1568, -0.2215,  0.2655, -0.0343, -0.2649, -0.1710,
         0.2799,  0.2893])
```

```
# 算法2: Normalized SGD
tensor([-0.1496, -0.1486, -0.2989,  0.0683, -0.1775, -0.2820,  0.2773, -0.2053,
        -0.0360,  0.0906])
tensor([ 0.2812, -0.0810,  0.1392,  0.2818,  0.1046,  0.3161,  0.1640,  0.1966,
        -0.1107,  0.1517])
tensor([-0.0687,  0.0421,  0.1568, -0.2215,  0.2655, -0.0343, -0.2649, -0.1710,
         0.2799,  0.2893])
```
可见两种算法下更新后的`LowerLayers`各层参数完全一致。另一方面, 由于两种算法对`UpperLayer`的处理方式不同, 其权重更新后有所不同, 但两个算法最后给出的predict结果则如下:  

```
# 算法1: Pop-Art
tensor([3.9995])
```

```
# 算法2: Normalized SGD
tensor([3.9995])
```
结果完全一致, 印证了论文2.3节中的**Proposition 3**。

[1]. [What is the recommended way to re-assign/update values in a variable (or tensor)?](https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125)