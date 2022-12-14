import tensorflow as tf 
import numpy as np
import config
import random


def roulette(functions, probs):
    """ Implements roulette wheel selection. """
    
    r = random.random()
    for func, prob in zip(functions, probs):
        if r < prob:
            return func
        else:
            r -= prob
    return None


def mean_sq_error(y1, y2):
    """ Mean square error. """ 
    diff = y1 - y2
    E = 100 * sum(diff*diff) / len(y1)
    return E


def accuracy_score(y1, y2):
    """ Accuracy score including argmax. """ 
    assert y1.shape == y2.shape

    y1_argmax = np.argmax(y1, axis=1)
    y2_argmax = np.argmax(y2, axis=1)
    score = sum(y1_argmax == y2_argmax)
    return score / len(y1)


def error(y1, y2):
    """ Return either accuracy score for classification task 
         or mean square error fo regression. """ 
    if isinstance(y1, tf.data.Dataset):
        if config.global_config["main_alg"]["task_type"] in ("classification", "binary_classification"):
            raise NotImplementedError()

        test_dataset = y1
        loss = 0
        start = 0
        for _, labels in test_dataset:
            d = labels - y2[start:start+labels.shape[0]]
            start += labels.shape[0]
            norms = tf.norm(d, axis=1, ord=2)
            loss += tf.math.reduce_mean(norms)
        return tf.math.reduce_mean(norms)

    else:
        if config.global_config["main_alg"]["task_type"] in ("classification", "binary_classification"):
            return accuracy_score(y1, y2)
        else:
            return mean_sq_error(y1, y2)


def print_stat(E, name):
    """ Outputs statistics """
    print(f"E_{name:6} avg={np.mean(E):.4f} std={np.std(E):.4f}"
          f"min={np.min(E):.4f} max={np.max(E):.4f}")
