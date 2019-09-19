from tensorflow.python.keras.models import Sequential, InputLayer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.initializers import Identity
from tensorflow.python.keras.callbacks import Callback
import tensorflow as tf
from utils import *


class LossHistory(Callback):
    # log losses per batch, rf: https://keras.io/callbacks/#create-a-callback
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('root_mean_squared_error'))


# rmse metric, rf: https://stackoverflow.com/a/43863854/8064227
def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


# build model according to https://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf
def build_model(lr):
    model = Sequential([
        InputLayer((16,), name="input"),
        Dense(10, activation="tanh", name="h0"),
        Dense(10, activation="tanh", name="h1"),
        Dense(10, activation="tanh", name="h2"),
        Dense(1, activation="linear", name="output")
    ])
    model.compile(optimizer=SGD(lr), loss="mse", metrics=[root_mean_squared_error])
    return model


class Basic:
    def __init__(self, lr: float, seed: int):
        self.alpha = lr
        tf.random.set_random_seed(seed)
        self.model = build_model(self.alpha)
        self.mode = None


class SGD_only(Basic):
    def __init__(self, lr: float, seed: int):
        self.mode = "SGD-only"
        super().__init__(lr, seed)

    def training(self, x, y):
        rmses = []
        for i in range(len(x)):
            # rmses.append(self.model.evaluate(x[i].reshape(1, 16), np.asarray([y[i]]), verbose=False)[1])
            rmses.append(self.model.evaluate(x, y, verbose=False, batch_size=len(x))[1]) # test on whole set
            self.model.fit(x[i].reshape(1, 16), np.asarray([y[i]]), batch_size=1, verbose=False)
        return moving_average(rmses, 10)


class Art_only(Basic):
    def __init__(self, lr: float, seed: int):
        self.mu = None
        self.nu = None
        self.beta = 10.0 ** (-4.0)  # step size
        self.mode = "Art-only"
        super().__init__(lr, seed)

    def normalize(self, y):
        normalized_y = (y - self.mu) / np.sqrt(self.nu - self.mu ** 2)
        if hasattr(normalized_y, "__len__"):
            return normalized_y
        else:
            return np.asarray([normalized_y])

    def training(self, x, y):
        rmses = []
        self.mu = y[0]
        self.nu = y[0]**2
        for i in range(1, len(x)):
            # deal with y using Art (adaptive rescaling target)
            self.mu = (1 - self.beta) * self.mu + self.beta * y[i]
            self.nu = (1 - self.beta) * self.nu + self.beta * (y[i] ** 2)

            # # dump mu and nu
            # with open("record_mu_nu_seed={:d}.pkl".format(seed), "ab") as f:
            #     pickle.dump((self.mu, self.nu, y[i]), f)

            normalized_y = self.normalize(y[:i+1])
            rmses.append(np.sqrt(self.nu - self.mu ** 2) *
                         self.model.evaluate(x[:i+1], normalized_y, verbose=False, batch_size=i+1)[1])
            self.model.fit(x[i].reshape(1, 16), self.normalize(y[i]), batch_size=1, verbose=False)
        return moving_average(rmses, 10)


if __name__ == "__main__":

    rmse_repetition = []
    mode = "Art-only"

    profiles = {'Art-only': (Art_only, 10.0 ** (-2.5)),
                'SGD-only': (SGD_only, 10.0 ** (-3.5))}

    for seed in range(50):
        print("running for seed = {seed:d}".format(seed=seed))
        # tf.random.set_random_seed(seed)
        #
        # model = build_model(10.0**(-3.5))
        agent = profiles[mode][0](profiles[mode][1], seed)

        # sgd_only = SGD_only(10.0**(-3.5), seed)

        # generate dataset first
        os.system("python dataset_generator.py -s {seed:d}".format(seed=seed))
        # load dataset
        with open('dataset-with-weired-value.pkl', 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)

        # his = LossHistory()
        # model.fit(x, y, verbose=True, epochs=1, batch_size=1, callbacks=[his], shuffle=False)

        # evaluate
        # rmses = []
        # for i in range(len(x)):
        #     rmses.append(model.evaluate(x[i].reshape(1, 16), np.asarray([y[i]]), verbose=False)[1])
        #     model.fit(x[i].reshape(1, 16), np.asarray([y[i]]), batch_size=1, verbose=False)
        #
        # avg_rmses = moving_average(rmses, 10)
        avg_rmses = agent.training(x, y)
        rmse_repetition.append(avg_rmses)

        # plt.semilogy(avg_rmses)
        # plt.show()
    # plt.semilogy(his.losses)
    # plt.show()
    samples = np.linspace(0, 4994, rmse_repetition[0].shape[0], dtype=int)
    rmse_repetition = np.asarray(rmse_repetition)

    m, l, u = median_and_percentile(rmse_repetition, axis=0)

    # plot
    fig = plt.figure()
    spl = fig.add_subplot(111)
    spl.plot(samples, m, color='C0')
    spl.fill_between(samples, u, l, facecolor='blue', alpha=0.5)
    spl.set_yscale("log")
    plt.show()

    # save results
    save_results('{mode:s}-to-now.pkl'.format(mode=mode), samples, m, l, u)
