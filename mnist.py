import time

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

def NN_train(hidden_layer_sizes=(50,50), max_iter=10, alpha=1e-4):
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
        model = NN_train(**new_params)
        t_train = time.time() - start
        score = NN_check(model)
        params = new_params
        yield (score, t_train)

def param_update(params):
    params['max_iter'] += 100
    return params

if __name__ == "__main__":
    opter = NN_optimize({'max_iter': 10}, param_update)
    for _ in range(10):
        print(next(opter))
    pass
