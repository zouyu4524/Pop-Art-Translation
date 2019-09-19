import pickle
import matplotlib.pyplot as plt


with open('SGD-only-whole-set.pkl', 'rb') as f:
    samples = pickle.load(f)
    m = pickle.load(f)
    l = pickle.load(f)
    u = pickle.load(f)

with open('Art-only-whole-set.pkl', 'rb') as f:
    samples_ = pickle.load(f)
    m_ = pickle.load(f)
    l_ = pickle.load(f)
    u_ = pickle.load(f)


fig = plt.figure()
spl = fig.add_subplot(111)
spl.plot(samples, m, color='C0')
spl.fill_between(samples, u, l, facecolor='C0', alpha=0.5)
spl.plot(samples_, m_, color='C1')
spl.fill_between(samples_, u_, l_, facecolor='C1', alpha=0.5)
spl.set_yscale("log")
plt.show()