import numpy as np

class ActivationFunction:
    @staticmethod
    def linear(x):
        return np.ones_like(x)

    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # @staticmethod
    # def softplus(x):
    #     return np.log(1 + np.exp(x))
    
    # @staticmethod
    # def elu(x, alpha=1.0):
    #     return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    # @staticmethod
    # def selu(x):
    #     return np.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (np.exp(x) - 1))
    
    # @staticmethod
    # def prelu(x, alpha=0.01):
    #     return np.where(x > 0, x, alpha * x)
    
    # @staticmethod
    # def swish(x):
    #     return x * 1 / (1 + np.exp(-x))