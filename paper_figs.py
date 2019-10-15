import numpy as np
import matplotlib.pyplot as plt
from blockchain import *

def plotter(states, *args, show=False, save=None, log=False):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    for a in args:
        ax.plot(states[a], label=a)
    if log:
        ax.set_yscale('log')
    plt.legend()
    if save:
        plt.savefig(save, format='pdf')
    if show:
        plt.show()

def data_merge(data_dicts):
    lists = lambda k: np.array([d[k] for d in data_dicts])
    return {k:np.mean(lists(k), axis=0) for k in data_dicts[0].keys()}

def multi_run(runs, height, **kwargs):
    sim_dicts = []
    for _ in range(runs):
        data = simulation(height, **kwargs)
        sim_dicts.append(data)
    return data_merge(sim_dicts)
    pass

if __name__ == "__main__":
    d = multi_run(2, 20)
    plotter(d, 'db', 'dr', show=True)
    pass
