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
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x, y_pred=None):
        if y_pred is None:
            shifted_x = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(shifted_x)
            y_pred = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        s = y_pred.reshape(-1, 1)
        jacobian = np.diagflat(y_pred) - np.dot(s, s.T)
        
        return jacobian