import numpy as np


def identity(x: float) -> float:
    return x


def differential_of_identity(_: float) -> float:
    return 1


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(x))


def differential_of_sigmoid(x: float) -> float:
    return 1 / (1 + 1 / np.exp(np.square(x)))
