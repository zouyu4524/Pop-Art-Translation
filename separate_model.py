import torch
import pickle

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


class UpperLayer(torch.nn.Module):
    def __init__(self, H, n_out):
        super(UpperLayer, self).__init__()
        self.output_linear = torch.nn.Linear(H, n_out)
        torch.nn.init.ones_(self.output_linear.weight)
        torch.nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        y_pred = self.output_linear(x)
        return y_pred


class UnifiedModel(torch.nn.Module):
    def __init__(self, n_in, H, n_out):
        super(UnifiedModel, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H, H)
        self.hidden3 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, n_out)
        torch.nn.init.ones_(self.output_linear.weight)
        torch.nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        h_tanh = torch.tanh(self.input_linear(x))
        h_tanh = torch.tanh(self.hidden1(h_tanh))
        h_tanh = torch.tanh(self.hidden2(h_tanh))
        h_tanh = torch.tanh(self.hidden3(h_tanh))
        y_pred = self.output_linear(h_tanh)
        return y_pred

if __name__ == "__main__":

    # prepare data
    with open('dataset-with-weired-value.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)

    sample_x = torch.tensor(x[0], dtype=torch.float)
    sample_y = torch.tensor([y[0]], dtype=torch.float)

    lr = 0.1  # learning rate

    # case 1: separated model
    torch.manual_seed(0)
    lower_layer = LowerLayers(16, 10)
    upper_layer = UpperLayer(10, 1)

    opt_lower = torch.optim.SGD(lower_layer.parameters(), lr)
    opt_upper = torch.optim.SGD(upper_layer.parameters(), lr)

    loss_func = torch.nn.MSELoss()
    loss = loss_func(upper_layer(lower_layer(sample_x)), sample_y)

    # loss_lower = lower_layer(sample_x)  # intermediate output of the lower layers
    # with torch.no_grad():
    #     delta = upper_layer(lower_layer(sample_x)) - sample_y
    #
    # loss_lower.backward([2* delta * upper_layer.output_linear.weight])
    loss.backward()
    opt_lower.step()
    opt_upper.step()

    with torch.no_grad():
        # for i, para in enumerate(lower_layer.parameters()):
        #     if i % 2 == 1:
        #         # para -= 2 * lr * para.grad * upper_layer.output_linear.weight.reshape(10, ) * delta
        #         # continue  # skip bias term update
        #         pass
        #     else:
        #         para -= 2 * lr * para.grad * upper_layer.output_linear.weight.t() * delta
        # for para in lower_layer.parameters():
        #     para -= lr * para.grad

        # print(lower_layer.hidden1.weight[-1, :])
        # print(lower_layer.hidden2.weight[-1, :])
        # print(lower_layer.hidden3.weight[-1, :])
        # print(lower_layer.hidden1.bias)
        # print(lower_layer.hidden2.bias)
        # print(lower_layer.hidden3.bias)
        print(upper_layer.output_linear.weight)
        print(upper_layer.output_linear.bias)

    # case 2: unified model
    torch.manual_seed(0)
    unified_layer = UnifiedModel(16, 10, 1)

    loss_func = torch.nn.MSELoss()
    loss = loss_func(unified_layer(sample_x), sample_y)

    loss.backward()

    with torch.no_grad():
        for para in unified_layer.parameters():
            para -= lr * para.grad

        # print(unified_layer.hidden1.weight[-1, :])
        # print(unified_layer.hidden2.weight[-1, :])
        # print(unified_layer.hidden3.weight[-1, :])
        # print(unified_layer.hidden1.bias)
        # print(unified_layer.hidden2.bias)
        # print(unified_layer.hidden3.bias)
        print(unified_layer.output_linear.weight)
        print(unified_layer.output_linear.bias)