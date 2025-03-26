import numpy as np

class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        epsilon = 1e-15
        clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(-y_true * np.log(clipped_y_pred) - (1 - y_true) * np.log(1 - clipped_y_pred))

    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        epsilon = 1e-15
        clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(clipped_y_pred))