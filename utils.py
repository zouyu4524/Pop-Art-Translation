import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


# moving average of results
def moving_average(x, window=10):
    n = len(x)
    assert n >= window
    return np.asarray([
        np.mean(x[i:i + window]) for i in range(n - window + 1)
    ])


# calculate median and given percentiles of a sequence
def median_and_percentile(x, axis, lower=10, upper=90):
    assert (lower >= 0 and upper <= 100)
    median = np.median(x, axis)
    low_per = np.percentile(x, lower, axis)
    up_per = np.percentile(x, upper, axis)
    return median, low_per, up_per


def save_results(filename, samples, m, l, u):
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)
        pickle.dump(m, f)
        pickle.dump(l, f)
        pickle.dump(u, f)


if __name__ == "__main__":
    pass
    # rmse_repetition = []
    # mode = "Art-only"
    #
    # profiles = {'Art-only': (Art_only, 10.0 ** (-2.5)),
    #             'SGD-only': (SGD_only, 10.0 ** (-3.5))}
    #
    # for seed in range(50):
    #     print("running for seed = {seed:d}".format(seed=seed))
    #     # tf.random.set_random_seed(seed)
    #     #
    #     # model = build_model(10.0**(-3.5))
    #     agent = profiles[mode][0](profiles[mode][1], seed)
    #
    #     # sgd_only = SGD_only(10.0**(-3.5), seed)
    #
    #     # generate dataset first
    #     os.system("python dataset_generator.py -s {seed:d}".format(seed=seed))
    #     # load dataset
    #     with open('dataset-with-weired-value.pkl', 'rb') as f:
    #         x = pickle.load(f)
    #         y = pickle.load(f)
    #
    #     # his = LossHistory()
    #     # model.fit(x, y, verbose=True, epochs=1, batch_size=1, callbacks=[his], shuffle=False)
    #
    #     # evaluate
    #     # rmses = []
    #     # for i in range(len(x)):
    #     #     rmses.append(model.evaluate(x[i].reshape(1, 16), np.asarray([y[i]]), verbose=False)[1])
    #     #     model.fit(x[i].reshape(1, 16), np.asarray([y[i]]), batch_size=1, verbose=False)
    #     #
    #     # avg_rmses = moving_average(rmses, 10)
    #     avg_rmses = agent.training(x, y)
    #     rmse_repetition.append(avg_rmses)
    #
    #     # plt.semilogy(avg_rmses)
    #     # plt.show()
    # # plt.semilogy(his.losses)
    # # plt.show()
    # samples = np.linspace(0, 4994, rmse_repetition[0].shape[0], dtype=int)
    # rmse_repetition = np.asarray(rmse_repetition)
    #
    # m, l, u = median_and_percentile(rmse_repetition, axis=0)
    #
    # # plot
    # fig = plt.figure()
    # spl = fig.add_subplot(111)
    # spl.plot(samples, m, color='C0')
    # spl.fill_between(samples, u, l, facecolor='blue', alpha=0.5)
    # spl.set_yscale("log")
    # plt.show()
    #
    # # save results
    # save_results('{mode:s}-to-now.pkl'.format(mode=mode), samples, m, l, u)
