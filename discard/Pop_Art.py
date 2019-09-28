import torch
from separate_model import LowerLayers, UpperLayer
import numpy as np
import pickle


class PopArt:
    def __init__(self, n_in, H, n_out):
        self.lower_layers = LowerLayers(n_in, H)
        self.upper_layer  = UpperLayer(H, n_out)
        self.sigma = torch.tensor(1., dtype=torch.float)  # consider scalar first
        self.sigma_new = None
        self.mu = torch.tensor(0., dtype=torch.float)
        self.mu_new = None
        self.nu = self.sigma**2 + self.mu**2 # second-order moment
        self.beta = 10.**(-0.5)
        self.lr = 0.5*10.**(-2.5)
        self.loss_func = torch.nn.MSELoss()
        self.loss = None
        self.opt_lower = torch.optim.SGD(self.lower_layers.parameters(), self.lr)
        self.opt_upper = torch.optim.SGD(self.upper_layer.parameters(), self.lr)

    def update_statistics(self, y):
        self.mu_new = (1. - self.beta) * self.mu + self.beta * y
        self.nu = (1. - self.beta) * self.nu + self.beta * y**2
        self.sigma_new = np.sqrt(self.nu - self.mu_new**2)

    def update_upper_layer_stage1(self):
        relative_sigma = (self.sigma / self.sigma_new)
        self.upper_layer.output_linear.weight.data *= relative_sigma
        self.upper_layer.output_linear.bias.data *= relative_sigma
        self.upper_layer.output_linear.bias.data += (self.mu-self.mu_new)/self.sigma_new
        # update statistics
        self.sigma = self.sigma_new
        self.mu = self.mu_new

    def normalize(self, y):
        return (y - self.mu) / self.sigma

    def denormalize(self, y):
        return self.sigma * y + self.mu

    def forward(self, x, y):
        self.update_statistics(y)
        self.update_upper_layer_stage1()
        y_pred = self.upper_layer(self.lower_layers(x))
        self.loss = self.loss_func(y_pred, self.normalize(y))
        return y_pred

    def backward(self):
        self.opt_lower.zero_grad()
        self.opt_upper.zero_grad()
        self.loss.backward()

    def step(self):
        self.opt_lower.step()
        self.opt_upper.step()


class NormalizedSGD:
    def __init__(self, n_in, H, n_out):
        self.lower_layers = LowerLayers(n_in, H)
        self.upper_layer = UpperLayer(H, n_out)
        self.sigma = torch.tensor(1., dtype=torch.float)  # consider scalar first
        self.mu = torch.tensor(0., dtype=torch.float)
        self.nu = self.sigma ** 2 + self.mu ** 2  # second-order moment
        self.beta = 10. ** (-0.5)
        self.lr = 0.5*10. ** (-2.5)
        self.loss_func = torch.nn.MSELoss()
        self.loss = None
        # self.intermediate = None  # store output of lower layers
        self.y = None
        self.opt_lower = torch.optim.SGD(self.lower_layers.parameters(), self.lr)
        self.opt_upper = torch.optim.SGD(self.upper_layer.parameters(), self.lr)

    def update_statistics(self, y):
        self.mu = (1. - self.beta) * self.mu + self.beta * y
        self.nu = (1. - self.beta) * self.nu + self.beta * y**2
        self.sigma = np.sqrt(self.nu - self.mu**2)

    def normalize(self):
        self.upper_layer.output_linear.weight.data /= self.sigma**2
        # self.loss.data /= self.sigma.item()

    def denormalize(self):
        self.upper_layer.output_linear.weight.data *= self.sigma**2
        # self.loss.data *= self.sigma.item()

    def forward(self, x, y):
        self.update_statistics(y)
        self.intermediate = self.lower_layers(x).clone().detach()
        y_pred = self.upper_layer(self.lower_layers(x))
        self.y = y
        self.loss = self.loss_func(y_pred, y)
        return y_pred

    def backward(self):
        self.opt_lower.zero_grad()
        self.opt_upper.zero_grad()
        # scale weights of upper layer
        self.normalize()
        self.loss.backward()
        # scale back weights of upper layer
        self.denormalize()


    def step(self):
        self.opt_lower.step()
        self.opt_upper.step()


if __name__ == "__main__":

    with open('../dataset/dataset-with-weired-value.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)

    sample_x = torch.tensor(x[0], dtype=torch.float)
    sample_y = torch.tensor([y[0]], dtype=torch.float)

    torch.manual_seed(0)
    pop_art = PopArt(16, 10, 1)
    y_pred = pop_art.forward(sample_x, sample_y)
    pop_art.backward()
    pop_art.step()

    torch.manual_seed(0)
    normalized_sgd = NormalizedSGD(16, 10, 1)
    y_pred2 = normalized_sgd.forward(sample_x, sample_y)
    normalized_sgd.backward()
    normalized_sgd.step()

    print(pop_art.lower_layers.hidden1.bias.data)
    print(pop_art.lower_layers.hidden2.bias.data)
    print(pop_art.lower_layers.hidden3.bias.data)

    print(normalized_sgd.lower_layers.hidden1.bias.data)
    print(normalized_sgd.lower_layers.hidden2.bias.data)
    print(normalized_sgd.lower_layers.hidden3.bias.data)