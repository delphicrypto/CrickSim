import pickle
import time

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

def NN_train(hidden_layer_sizes=(50,50), max_iter=1000, alpha=1e-4):
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha,
                        solver='adam', verbose=0, tol=1e-10,
                        learning_rate_init=.1) 
    mlp.fit(data, digits.target)
    return mlp

def NN_check(model):
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return model.score(data, digits.target)


def NN_optimize(params, param_update):
    """
    Try different parameters until `current_best` is beaten.
    Written as generator which yields the current score after one NN training.

    Arguments:
     `params`: dictionary of neural network parameters by keyword
     `param_update`: function that takes param dictionary and returns new one for next iteration.
    """

    score = 0
    model = None

    # while score < current_best:
    while True:
        new_params = param_update(params)
        start = time.time()
        print(new_params)
        model = NN_train(**new_params)
        t_train = time.time() - start
        score = NN_check(model)
        params = new_params
        yield (score, t_train)

def param_update(params):
    # params['alpha'] = params['alpha'] / 2
    return params

if __name__ == "__main__":
    num_miners = 3
    times_list = []
    for i in range(num_miners):
        print(f"miner {i}")
        miner_times = []
        opter = NN_optimize({'alpha': 0.1}, param_update)
        for _ in range(1000):
            score, t = next(opter)
            miner_times.append((score, t))
            print(score)
        times_list.append(miner_times)
    print(times_list)
    pickle.dump(times_list, open("miner_times_v1.pickle", "wb"))
    pass
