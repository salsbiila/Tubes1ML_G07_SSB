import numpy as np

class LossFunction:
    
    @staticmethod
    def mse(y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        loss = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
        return loss
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes not match. y_true= {y_true.shape}. y_pred= {y_pred.shape}")
        
        
        y_pred = np.clip(y_pred, epsilon, 1.0)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    @staticmethod
    def compute_l1_regularization(weights, lambda_param):
        l1_reg = 0
        for w in weights.values():
            l1_reg += np.sum(np.abs(w))
        return lambda_param * l1_reg
    
    @staticmethod
    def compute_l2_regularization(weights, lambda_param):
        l2_reg = 0
        for w in weights.values():
            l2_reg += np.sum(w**2)
        return 0.5 * lambda_param * l2_reg
    
    @staticmethod
    def compute_regularized_loss(loss, weights, reg_type, lambda_param):
        if reg_type is None or lambda_param == 0:
            return loss
        
        if reg_type.lower() == 'l1':
            reg_term = LossFunction.compute_l1_regularization(weights, lambda_param)
        elif reg_type.lower() == 'l2':
            reg_term = LossFunction.compute_l2_regularization(weights, lambda_param)
        else:
            raise ValueError(f"Regularization type '{reg_type}' not recognized. Use 'l1', 'l2', or None.")
        
        return loss + reg_term