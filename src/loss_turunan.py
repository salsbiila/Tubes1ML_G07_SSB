import numpy as np

class LossDerivative:
    @staticmethod
    def mse(y_true, y_pred):
        n = y_true.shape[0]
        return (2/n) * (y_pred - y_true)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
        
        return -(1/n) * (y_true / y_pred - (1 - y_true) / (1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1.0) 
        
        return -(1/n) * (y_true / y_pred)
    
    @staticmethod
    def softmax_categorical_cross_entropy(y_true, y_pred):
        n = y_true.shape[0]
        return (1/n) * (y_pred - y_true)