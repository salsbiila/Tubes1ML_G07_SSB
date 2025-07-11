import numpy as np
from activation_turunan import ActivationDerivative
from loss_turunan import LossDerivative

class BackPropagation:
    @staticmethod
    def compute_gradients(model, X, y):
        gradients = {
            "weights": {},
            "biases": {}
        }
        
        activations = model.activations
        pre_activations = model.pre_activations
        n_layers = len(model.layer_sizes) - 1
        
        y_pred = activations[n_layers]
        
        if model.loss_function == "mse":
            dL_dy_pred = LossDerivative.mse(y, y_pred)
        elif model.loss_function == "bce":
            dL_dy_pred = LossDerivative.binary_cross_entropy(y, y_pred)
        elif model.loss_function == "cce":
            if model.activation_funcs[n_layers-1] == "softmax":
                dL_dy_pred = LossDerivative.softmax_categorical_cross_entropy(y, y_pred)
            else:
                dL_dy_pred = LossDerivative.categorical_cross_entropy(y, y_pred)
        else:
            raise ValueError(f"Loss function '{model.loss_function}' not recognized")
        
        # Iterasi mundur
        delta = dL_dy_pred
        for layer in range(n_layers, 0, -1):

            if layer == n_layers and model.activation_funcs[layer-1] == "softmax" and model.loss_function == "cce":
                pass  # Gunakan delta yang sudah dihitung
            else:
                activation_name = model.activation_funcs[layer-1]
                activation_derivative_method = getattr(ActivationDerivative, activation_name)
                
                df_dz = activation_derivative_method(pre_activations[layer])
                delta = delta * df_dz
            
            # dL/dW = dL/dy_pred * df/dz * dz/dW = delta * aktivasi layer sebelumnya
            gradients["weights"][layer] = activations[layer-1].T @ delta
            
            if model.reg_type is not None and model.lambda_param > 0:
                if model.reg_type == "l1":
                    reg_grad = model.lambda_param * LossDerivative.l1_regularization(model.weights[layer])
                    gradients["weights"][layer] += reg_grad
                elif model.reg_type == "l2":
                    reg_grad = model.lambda_param * LossDerivative.l2_regularization(model.weights[layer])
                    gradients["weights"][layer] += reg_grad
            
            # dL/db = dL/dy_pred * df/dz * dz/db = delta
            gradients["biases"][layer] = np.sum(delta, axis=0, keepdims=True)
            
            if layer > 1:
                # delta untuk layer sebelumnya = delta sekarang * weights sekarang
                delta = delta @ model.weights[layer].T
        
        return gradients
    
    @staticmethod
    def update_weights(model, gradients, learning_rate):
        for layer in gradients["weights"]:
            model.weights[layer] -= learning_rate * gradients["weights"][layer]
            model.biases[layer] -= learning_rate * gradients["biases"][layer]
    
    @staticmethod
    def backward(model, X, y, learning_rate):

        gradients = BackPropagation.compute_gradients(model, X, y)
        BackPropagation.update_weights(model, gradients, learning_rate)
        
        return gradients