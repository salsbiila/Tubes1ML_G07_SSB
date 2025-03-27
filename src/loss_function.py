import numpy as np

class LossFunction:
    
    @staticmethod
    def mse(y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        loss = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
        
        n = y_true.shape[0]        
        grad = 2 * (y_pred - y_true) / n
        
        return loss, grad
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        n = y_true.shape[0]
        grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n
        
        return loss, grad
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        
        y_pred = np.clip(y_pred, epsilon, 1.0)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        n = y_true.shape[0]
        grad = -y_true / y_pred / n
        
        return loss, grad