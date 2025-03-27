import numpy as np

class ActivationDerivative:
    @staticmethod
    def linear(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def tanh(x):
        tanh_x = np.tanh(x)
        return 1 - np.power(tanh_x, 2)
    
    @staticmethod
    def softmax(x, y_pred=None):
        if y_pred is None:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        return y_pred
    
    @staticmethod
    def softplus_derivative(x):
        return 1 / (1 + np.exp(-x))
    
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def selu_derivative(x):
        alpha = 1.67326
        scale = 1.0507
        return np.where(x > 0, scale, scale * alpha * np.exp(x))
    
    @staticmethod
    def prelu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def swish_derivative(x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)