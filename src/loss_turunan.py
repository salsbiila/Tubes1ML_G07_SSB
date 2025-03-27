import numpy as np

class LossDerivative:
    @staticmethod
    def mse(y_true, y_pred):
        n = y_true.shape[0]
        return -2 * (y_true - y_pred) / n

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1 * (y_true - y_pred) / (y_pred * (1 - y_pred) * n)
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -1 * y_true / (y_pred * n)
    
    @staticmethod
    def softmax_categorical_cross_entropy(y_true, y_pred):
        n = y_true.shape[0]
        return (y_pred - y_true) / n