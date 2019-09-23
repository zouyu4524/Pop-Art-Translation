import torch

x = torch.tensor([1.], requires_grad=True)

# print(id(x))
# with torch.no_grad():
x.data *= 3.
x.data = x.data + 1.
# print(id(x))

z = x**2
#
z.backward()
#
print(x.grad)