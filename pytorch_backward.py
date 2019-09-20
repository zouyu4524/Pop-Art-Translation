import torch as t
from torch.autograd import Variable as v

# simple gradient
a = t.tensor([2, 3], dtype=t.float, requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()
print('*'*10)
print('=====simple gradient======')
print('input')
print(a)
print('compute result is')
print(out.item())
print('input gradients are')
print(a.grad.data)

# backward on non-scalar output
m = t.tensor([[2, 3]], dtype=t.float, requires_grad=True)
n = t.zeros(1, 2)
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
n.backward(t.tensor([[1, 1]], dtype=t.float))
print('*'*10)
print('=====non scalar output======')
print('input')
print(m)
print('input gradients are')
print(m.grad.data)

# jacobian
j = t.zeros(2 ,2)
k = t.zeros(1, 2)
m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
k.backward(t.tensor([[1, 0]], dtype=t.float), retain_graph=True)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(t.tensor([[0, 1]], dtype=t.float))
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)

# compute jacobian matrix
x = t.tensor([2, 1], dtype=t.float).view(1, 2)
x = x.clone().detach().requires_grad_(True)
y = t.tensor([[1, 2], [3, 4]], dtype=t.float).clone().detach().requires_grad_(True)

z = t.mm(x, y)
jacobian = t.zeros((2, 2))
z.backward(t.tensor([[1, 0]], dtype=t.float), retain_graph=True)  # dz1/dx1, dz1/dx2
jacobian[:, 0] = x.grad.data
x.grad.data.zero_()
z.backward(t.tensor([[0, 1]], dtype=t.float))  # dz2/dx1, dz2/dx2
jacobian[:, 1] = x.grad.data
print('=========jacobian========')
print('x')
print(x)
print('y')
print(y)
print('compute result')
print(z)
print('jacobian matrix is')
print(jacobian)