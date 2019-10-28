import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from blockchain import *

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'text.usetex': True}

plt.rc('font', family='serif')
plt.rcParams.update(params)

# matplotlib.rcParams['figure.figsize'] = (10,10)

def plotter(states, *args, xlabel="", ylabel="", diagonal=False, show=False, save=None, log=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['blue', 'red', 'green']
    for i,a in enumerate(args):
        ax.plot(states[a], label=a, color=colors[i])
    if diagonal:
        ax.plot([i+1 for i in range(len(states[a]))], '--')
    if log:
        ax.set_yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save:
        plt.savefig(save, format='pdf')
    if show:
        plt.show()

def data_merge(data_dicts):
    lists = lambda k: np.array([d[k] for d in data_dicts])
    return {k:np.mean(lists(k), axis=0) for k in data_dicts[0].keys()},\
           {k:np.mean(lists(k), axis=0) for k in data_dicts[0].keys()}

def multi_run(runs, height, **kwargs):
    sim_dicts = []
    for i in range(runs):
        data = simulation(height, **kwargs)
        sim_dicts.append(data)
        print(f"done run {i+1} of {runs}")
    return data_merge(sim_dicts)

def eta_plot():
    etas = []
    sols_v1 = []
    sols_v2 = []
    height = 100
    num_etas = 15
    for eta in (1/(2**i) for i in range(num_etas)):
        print(eta)
        for v in ['v1', 'v2']:
            print(v)
            if v == 'v1':
                d_avg,d_std = multi_run(1, height, eta=eta, mode=v, update_freq=10)
                sols_v1.append(d_avg['sol_blocks'][-1] / height)
            if v == 'v2':
                d_avg, d_std = multi_run(1, height, eta=eta, mode=v, v2_update_freq=5, update_freq=10)
                sols_v2.append(d_avg['sol_blocks'][-1] / height)

        etas.append(eta)
        pickle.dump(zip(etas, sols_v1), open('v1_etas_sol_2.p', 'wb'))
        pickle.dump(zip(etas, sols_v2), open('v2_etas_sol_2.p', 'wb'))

    # etas, sols_v1 = list(zip(*pickle.load(open('v1_etas_sol.p', 'rb'))))
    # etas, sols_v2 = list(zip(*pickle.load(open('v2_etas_sol.p', 'rb'))))

    xs = np.arange(num_etas)
    fit_v1 = np.polyfit(xs, sols_v1, 1)
    fit_v1_fn = np.poly1d(fit_v1)

    fit_v2 = np.polyfit(xs, sols_v2, 1)
    fit_v2_fn = np.poly1d(fit_v2)

    plt.plot(xs, sols_v1, 'o',  color='blue', label='$v_1$')
    # plt.plot(xs, fit_v1_fn(xs), '--', color='blue')

    plt.plot(xs, sols_v2, 'o', color='red', label='$v_2$')
    plt.plot(xs, fit_v2_fn(xs), '--', color='red')

    labels = [r"$1$"] + ["$\\frac{{1}}{{{0}}}$".format(2**i) for i in range(1, len(etas))]
    plt.xticks(xs, labels)
    plt.xlabel("$\eta$")
    plt.ylabel("Fraction of solution blocks")
    plt.legend(loc='lower right')
    plt.savefig('eta_fig_2.pdf', format='pdf')
    # plt.show()

def sol_plot():
    d_mean, _ = multi_run(1, 200,eta=1/200, mode = 'v1', bounce=False, update_freq=10)
    # d_2 = multi_run(2, 200,eta=1/200, mode = 'v2', bounce=False, update_freq=10)
    plotter(d_mean, 'btc_blocks', 'sol_blocks', log=False, diagonal=True, save="hi.pdf")
    pass
def diff_plot():
    d_mean, _ = multi_run(1, 200,eta=1/200, mode = 'v1', bounce=False, update_freq=10)
    # d_2 = multi_run(2, 200,eta=1/200, mode = 'v2', bounce=False, update_freq=10)
    plotter(d_mean, 'db', 'dr', log=True, diagonal=False, save="d.pdf")
    d_mean, _ = multi_run(1, 200,eta=1/200, mode = 'v2', bounce=False, update_freq=10)
    # d_2 = multi_run(2, 200,eta=1/200, mode = 'v2', bounce=False, update_freq=10)
    plotter(d_mean, 'db', 'dr', log=True, diagonal=False, save="d2.pdf")
    pass
if __name__ == "__main__":
    # d = multi_run(25, 200,eta=1/200, mode = 'v1', bounce=False, update_freq=10)
    # pickle.dump(d, open('../data/v1_eta200_nobounce.p', 'wb'))
    # d = multi_run(25, 200,eta=1/200, mode = 'v2', bounce=False, update_freq=10)
    # pickle.dump(d, open('../data/v2_eta200_nobounce.p', 'wb'))
    # d = pickle.load(open('../data/v2_eta200_nobounce.p', 'rb'))
    # plotter(d, 'sol_blocks', 'btc_blocks', log=False, diagonal=True, save='sol_btc_blocks_v2.pdf')
    eta_plot()
    # diff_plot()
    # sol_plot()
    pass
