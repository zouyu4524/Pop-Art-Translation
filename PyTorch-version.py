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

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def build_model(learning_rate):
    model = MyModel(16, 10, 1)
    loss_func = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return model, loss_func, optimizer

def training_model(x, y):
    rmse = []
    for i, sample in enumerate(x):
        # load one sample per step
        input = torch.tensor(sample, dtype=torch.float)
        output = torch.tensor([y[i]], dtype=torch.float)

        # predict output with current model
        pred = model(input)

        # record rmse
        with torch.no_grad():
            rmse.append(RMSELoss(pred, output).item())

        loss = loss_func(pred, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rmse


if __name__ == "__main__":

    learning_rate = 10.0 ** (-3.5)

    rmses = []
    for seed in range(50):
        print("Running for seed {:d}.".format(seed))
        start_tic = time.time()
        # build network
        torch.manual_seed(seed)
        model, loss_func, optimizer = build_model(learning_rate)

        # generate dataset
        os.system("python dataset_generator.py -s {seed:d}".format(seed=seed))

        # load dataset
        with open('dataset-with-weired-value.pkl', 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)

        rmse = training_model(x, y)
        rmses.append(rmse)
        print("Time elapsed {:.2f} seconds for run {:d}.".format(time.time()-start_tic, seed))

