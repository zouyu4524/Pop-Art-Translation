import torch

x = torch.tensor([1.], requires_grad=True)

# print(id(x))
# with torch.no_grad():
z = x**2
x.data *= 3.
x.data = x.data + 1.
# print(id(x))
#
z.backward()
#
print(x.grad)