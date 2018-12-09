from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

def NN_train(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4):
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)

    mlp.fit(data, digits.target)
    return mlp

def NN_check(model):
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return model.score(data, digits.target)


def NN_optimize(current_best):
    score = 0
    size = 1
    max_iter = 1
    model = None

    while score < current_best:
        model = NN_train(max_iter=max_iter)
        max_iter += 1
        score = NN_check(model)
    print(f"best score: {score}")
    print(f"n_iter: {max_iter}")
    return model

if __name__ == "__main__":
    NN_optimize(0.2)
    pass
