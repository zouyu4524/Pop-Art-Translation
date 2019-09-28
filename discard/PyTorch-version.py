import torch
from utils import *
import time

class MyModel(torch.nn.Module):
    def __init__(self, n_in, H, n_out):
        super(MyModel, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, n_out)

    def forward(self, x):
        h_relu = torch.tanh(self.input_linear(x))
        for _ in range(2):
            h_relu = torch.tanh(self.hidden(h_relu))
        y_pred = self.output_linear(h_relu)
        return y_pred


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


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def build_model(learning_rate):
    model = MyModel(16, 10, 1)
    loss_func = torch.nn.MSELoss(reduction="sum")
    # loss_func = None
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return model, loss_func, optimizer


class Basic:
    def __init__(self, lr):
        self.model, self.loss_func, self.optim = build_model(lr)

    def training_model(self, x, y):
        raise NotImplementedError


class SGD_only(Basic):
    def training_model(self, x, y):
        rmse = []
        for i, sample in enumerate(x):
            # load one sample per step
            input = torch.tensor(sample, dtype=torch.float)
            output = torch.tensor([y[i]], dtype=torch.float)

            # predict output with current model
            pred = self.model(input)

            # record rmse
            with torch.no_grad():
                rmse.append(RMSELoss(pred, output).item())

            loss = self.loss_func(pred, output)
            # loss = RMSELoss(pred, output)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return moving_average(rmse)


class Art_only(Basic):
    def __init__(self, lr):
        super().__init__(lr)
        self.mu = 0.
        self.nu = 1.
        self.beta = 10.0**(-4.)

    def normalize(self, y):
        normalized_y = (y - self.mu) / np.sqrt(self.nu - self.mu ** 2)
        if hasattr(normalized_y, "__len__"):
            return normalized_y
        else:
            return np.asarray([normalized_y])

    def update_statics(self, yi):
        self.mu = (1 - self.beta) * self.mu + self.beta * yi
        self.nu = (1 - self.beta) * self.nu + self.beta * (yi** 2)

    def training_model(self, x, y):
        rmse = []
        # self.mu = y[0]
        # self.nu = (y[0]*1.0)**2
        for i in range(0, len(x)):
            # update mu and nu
            self.update_statics(y[i])

            # load one sample per step
            input = torch.tensor(x[i], dtype=torch.float)
            # output = torch.tensor(y[i], dtype=torch.float)
            # art
            normalized_output = torch.tensor(self.normalize(y[i]), dtype=torch.float)

            # predict output with current model
            pred = self.model(input)

            # record rmse
            with torch.no_grad():
                rmse.append(np.sqrt(self.nu - self.mu**2)
                            * RMSELoss(pred, normalized_output).item())

            loss = self.loss_func(pred, normalized_output)
            # loss = RMSELoss(pred, normalized_output)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return moving_average(rmse)


class Pop_Art:
    def __init__(self, n_in, H, n_out):
        self.lower_layers = LowerLayers(n_in, H)
        self.upper_layer = UpperLayer(H, n_out)

    def forward(self, x):
        return self.upper_layer(self.lower_layers(x))


if __name__ == "__main__":

    # torch.manual_seed(0)
    # lower_layer = LowerLayers(16, 10)
    # upper_layer = UpperLayer(10, 1)
    #
    # # unified_layer = UnifiedModel(16, 10, 1)
    #
    # with open('dataset-with-weired-value.pkl', 'rb') as f:
    #     x = pickle.load(f)
    #     y = pickle.load(f)
    #
    # sample_x = torch.tensor(x[0], dtype=torch.float)
    # sample_y = torch.tensor([y[0]], dtype=torch.float)
    #
    # # loss_func = torch.nn.MSELoss()
    # #
    # # loss = loss_func(unified_layer(sample_x), sample_y)
    #
    # # loss.backward()
    # loss_lower = lower_layer(sample_x).sum()
    # with torch.no_grad():
    #     delta = upper_layer(lower_layer(sample_x)) - sample_y
    #
    # lr = 0.1
    # loss_lower.backward()
    #
    # with torch.no_grad():
    #     for i, para in enumerate(lower_layer.parameters()):
    #         if i % 2 == 1:
    #             continue
    #         para -= 2 * lr * para.grad * upper_layer.output_linear.weight.t() * delta
    #
    # #
    # # output_intermediate = lower_layer(sample_x).sum()
    # # output_intermediate.backward()
    # # # for out in output_intermediate:
    # # #     out.backward()
    # #
    # # for j in lower_layer.parameters():
    # #     print(j.grad)
    #
    learning_rate = 0.5*10.0 ** (-4.5)

    rmses = []
    for seed in range(50):
        print("Running for seed {:d}.".format(seed))
        start_tic = time.time()
        # build network
        torch.manual_seed(seed)
        # model, loss_func, optimizer = build_model(learning_rate)
        agent = SGD_only(learning_rate)

        # generate dataset
        os.system("python dataset_generator.py -s {seed:d}".format(seed=seed))

        # load dataset
        with open('dataset-with-weired-value.pkl', 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)

        rmse = agent.training_model(x, y)
        rmses.append(rmse)
        print("Time elapsed {:.2f} seconds for run {:d}.".format(time.time()-start_tic, seed))

    m, l, u = median_and_percentile(rmses, axis=0)
    fig = plt.figure()
    spl = fig.add_subplot(111)
    spl.plot(m, color='C0')
    spl.fill_between(np.linspace(0, 4995, 4995, dtype=int), u, l, facecolor='blue', alpha=0.5)
    spl.set_yscale("log")
    plt.show()

    samples = np.linspace(0, 4995, 4995, dtype=int)
    save_results('SGD-only-0.5_lr=4.5.pkl', samples, m, l, u)