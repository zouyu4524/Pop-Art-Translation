import pickle
import matplotlib.pyplot as plt

with open('results/SGD.pkl', 'rb') as f:
    samples = pickle.load(f)
    m = pickle.load(f)
    l = pickle.load(f)
    u = pickle.load(f)

with open('results/ART.pkl', 'rb') as f:
    samples_ = pickle.load(f)
    m_ = pickle.load(f)
    l_ = pickle.load(f)
    u_ = pickle.load(f)

with open('results/PopArt.pkl', 'rb') as f:
    samples__ = pickle.load(f)
    m__ = pickle.load(f)
    l__ = pickle.load(f)
    u__ = pickle.load(f)


fig = plt.figure()
spl = fig.add_subplot(111)

spl.plot(samples__, m__, color='C2', label='PopArt')
spl.fill_between(samples__, u__, l__, facecolor='C2', alpha=0.5)

spl.plot(samples_, m_, color='C0', label='ART')
spl.fill_between(samples_, u_, l_, facecolor='C0', alpha=0.5)

spl.plot(samples, m, color='C3', label='SGD')
spl.fill_between(samples, u, l, facecolor='C3', alpha=0.5)

spl.set_yscale("log")

plt.xlabel('# samples')
plt.ylabel('RMSE (log scale)')
plt.xlim((0, 5000))

spl.legend(loc="lower left", frameon=False)

plt.show()