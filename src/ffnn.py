import numpy as np
import json, os
from weight_initializer import WeightInitializer
from activation_function import ActivationFunction
from loss_function import LossFunction
from backpropagation import BackPropagation
from interactive_visualizer import InteractiveVisualizer

class FFNN:
    def __init__(self, layer_sizes, activation_funcs=None, weight_init="xavier", loss_function="mse", mean=0, variance=0.1, lower_bound=-0.5, upper_bound=0.5, type="uniform", reg_type=None, lambda_param=0.01, seed=None):
        if not all(isinstance(n, int) and n > 0 for n in layer_sizes):
            raise ValueError("Semua elements di layer_sizes harus positive integers.")
        
        if len(layer_sizes) < 3:
            raise ValueError("The model setidaknya punya 3 layers (input, satu hidden layer, dan output).")
        self.layer_sizes = layer_sizes

        valid_activation_funcs = ["linear", "relu", "sigmoid", "tanh", "softmax", "softplus", "elu", "selu", "prelu", "swish"]
        valid_loss_funcs = ["mse", "bce", "cce"]
        valid_weight_inits = ["zero", "uniform", "normal", "xavier", "he"]
        valid_reg = [None, "l1", "l2"]

        if isinstance(activation_funcs, str):
            if activation_funcs not in valid_activation_funcs:
                raise ValueError(f"activation_funcs harus salah satu dari: {valid_activation_funcs}")
            
            if activation_funcs == "softmax" and loss_function != "cce":
                raise ValueError("Softmax hanya bisa dengan cross-entropy loss function.")
            
            if activation_funcs == "softmax" and layer_sizes[-1] == 1:
                raise ValueError("Cannot use softmax activation function with single output neuron.")
            
            self.activation_funcs = [activation_funcs.lower()] * (len(layer_sizes) - 1)

        elif activation_funcs is None:
            self.activation_funcs = ["relu"] * (len(layer_sizes) - 1)

        elif isinstance(activation_funcs, list):
            if len(activation_funcs) != len(layer_sizes) - 1:
                raise ValueError(f"activation_funcs must have exactly {len(layer_sizes) - 1} elements, activation funcs : {len(activation_funcs)}.")
            
            for i, af in enumerate(activation_funcs):
                if af not in valid_activation_funcs:
                    raise ValueError(f"Invalid activation function '{af}'. Must be one of: {sorted(valid_activation_funcs)}")
                
                if af == "softmax" and i != len(activation_funcs) - 1:
                    raise ValueError("Softmax activation function hanya bisa digunakan di output layer.")
            
            if activation_funcs[-1] == "softmax" and loss_function != "cce":
                raise ValueError("Softmax hanya bisa dengan cross-entropy loss function.")
            
            self.activation_funcs = [af.lower() for af in activation_funcs]
        else:
            raise TypeError("activation_funcs harus string, list of strings, atau None.")
        
        if loss_function.lower() not in valid_loss_funcs:
             raise ValueError(f"loss_function harus salah satu dari: {sorted(valid_loss_funcs)}")
        
        if loss_function.lower() == "cce" and self.layer_sizes[-1] <= 2:
            raise ValueError("Categorical Cross-Entropy is intended for multi-class classification (more than 2 output classes). For binary classification, use Binary Cross-Entropy (bce) instead.")
        self.loss_function = loss_function.lower()

        if weight_init.lower() not in valid_weight_inits:
            raise ValueError(f"weight_init harus salah satu dari: {sorted(valid_weight_inits)}")
        self.weight_init = weight_init.lower()

        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed harus integer atau None.")
        self.seed = seed
        
        if reg_type not in valid_reg:
            raise ValueError(f"reg_type tidak ada: {valid_reg}")
        self.reg_type = reg_type
        
        if not isinstance(lambda_param, (int, float)) or lambda_param < 0:
            raise ValueError("lambda_param harus non-negative number")
        self.lambda_param = lambda_param

        self.mean = mean
        self.variance = variance
        self.lower_bound= lower_bound
        self.upper_bound = upper_bound
        self.type = type

        self.weights = {}
        self.biases = {}
        self.weight_gradients = {}
        self.bias_gradients = {}
        self.pre_activations = {}  
        self.activations = {} 
        

        self.weights, self.biases = self.setup_weights()
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def setup_weights(self):
        weights = {}
        biases = {}
        
        # Parameter untuk inisialisasi
        params = {}
        
        if self.weight_init == "uniform":
            params["lower_bound"] = float(input("Masukkan batas bawah (default = -0.5) = ") or "-0.5")
            params["upper_bound"] = float(input("Masukkan batas atas (default = 0.5) = ") or "0.5")
        elif self.weight_init == "normal":
            params["mean"] = float(input("Masukkan mean (default = 0) = ") or "0")
            params["variance"] = float(input("Masukkan variance (default = 0.1) = ") or "0.1")
            
        # Loop dari input layer sampai output layer
        for i in range(1, len(self.layer_sizes)):
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]

            if self.weight_init == "zero":
                W, b = WeightInitializer.zero_initializer(input_size, output_size)
            elif self.weight_init == "uniform":
                W, b = WeightInitializer.random_uniform_initializer(
                    input_size, output_size, 
                    self.lower_bound, 
                    self.upper_bound, 
                    seed=self.seed
                )
            elif self.weight_init == "normal":
                W, b = WeightInitializer.random_normal_initializer(
                    input_size, output_size, 
                    self.mean, 
                    self.variance, 
                    seed=self.seed
                )
            elif self.weight_init == "xavier":
                W, b = WeightInitializer.xavier_initializer(input_size, output_size, self.type, seed=self.seed)
            elif self.weight_init == "he":
                W, b = WeightInitializer.he_initializer(input_size, output_size, self.type, seed=self.seed)
            else:
                raise ValueError(f"Weight initialization method '{self.weight_init}' not recognized.")

            weights[i] = W
            biases[i] = b

            self.weight_gradients[i] = np.zeros_like(W)
            self.bias_gradients[i] = np.zeros_like(b)
        
        return weights, biases
    
    def compute_loss(self, y_true, y_pred):
        if self.loss_function == "mse":
            base_loss = LossFunction.mse(y_true, y_pred)
        elif self.loss_function == "bce":
            base_loss = LossFunction.binary_cross_entropy(y_true, y_pred)
        elif self.loss_function == "cce":
            base_loss = LossFunction.categorical_cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Loss function '{self.loss_function}' not found.")

        regularized_loss = LossFunction.compute_regularized_loss(
            base_loss, self.weights, self.reg_type, self.lambda_param
        )
        
        return regularized_loss
        
    def forward(self, x):
        activation = x
        self.activations[0] =  activation

        for i in range(1, len(self.layer_sizes)):
            W = self.weights[i]
            b = self.biases[i]

            pre_activation = np.dot(activation, W) + b
            
            # terus diaktivasi
            activation_name = self.activation_funcs[i-1]
            if hasattr(ActivationFunction, activation_name):
                activation_func = getattr(ActivationFunction, activation_name)
                activation = activation_func(pre_activation)
            else:
                raise ValueError(f"Activation function '{activation_name}' not found.")

            self.pre_activations[i] = pre_activation
            self.activations[i] = activation
            
        return activation
    
    def backward(self, X, y, learning_rate):
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        
        gradients = BackPropagation.backward(self, X, y, learning_rate)
    
        for i in gradients["weights"]:
            self.weight_gradients[i] = gradients["weights"][i]
            self.bias_gradients[i] = gradients["biases"][i]
        return loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, learning_rate=0.01, epochs=100, verbose=1):
        n_samples = X_train.shape[0]
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
        for epoch in range(epochs):
            if self.seed is not None:
                np.random.seed(self.seed + epoch)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = self.backward(X_batch, y_batch, learning_rate)
                epoch_losses.append(batch_loss)
            
            avg_train_loss = np.mean(epoch_losses)
            self.history["train_loss"].append(avg_train_loss)
 
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.history["val_loss"].append(val_loss)

            if verbose == 1:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return self.history
    
    def predict(self, X):
        return self.forward(X)
    
    def print_model(self):
        print("Model Structure:")
        print(f"Input Layer: {self.layer_sizes[0]} neurons")
        for i in range(1, len(self.layer_sizes)):
            print(f"Layer {i}:")
            print(f"  - Neurons: {self.layer_sizes[i]}")
            print(f"  - Activation: {self.activation_funcs[i-1]}")
            print(f"  - Weights shape: {self.weights[i].shape}")
            print(f"  - Bias shape: {self.biases[i].shape}")

            if self.weights[i].size > 10:
                print(f"  - Sample weights: {self.weights[i].flatten()[:5]} ...")
            else:
                print(f"  - Weights: {self.weights[i]}")
            if self.biases[i].size > 10:
                print(f"  - Sample biases: {self.biases[i].flatten()[:5]} ...")
            else:
                print(f"  - Biases: {self.biases[i]}")
        
        if self.reg_type is not None:
            print(f"Regularization: {self.reg_type.upper()}, lambda={self.lambda_param}")
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            "layer_sizes": self.layer_sizes,
            "activation_funcs": self.activation_funcs,
            "weight_init": self.weight_init,
            "loss_function": self.loss_function,
            "reg_type": self.reg_type,
            "lambda_param": self.lambda_param,
            "seed": self.seed,
            "weights": {str(k): self.weights[k].tolist() for k in self.weights},
            "biases": {str(k): self.biases[k].tolist() for k in self.biases}
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            model_data = json.load(f)

        model = cls(
            layer_sizes=model_data["layer_sizes"],
            activation_funcs=model_data.get("activation_funcs", model_data.get("activation")),
            weight_init=model_data["weight_init"],
            loss_function=model_data["loss_function"],
            reg_type=model_data.get("reg_type", None),
            lambda_param=model_data.get("lambda_param", 0.0),
            seed=model_data["seed"]
        )

        model.weights = {int(k): np.array(v) for k, v in model_data["weights"].items()}
        model.biases = {int(k): np.array(v) for k, v in model_data["biases"].items()}

        print(f"Model loaded from {filepath} 💾")
        return model
    
    def visualize_model(self, show_gradients=True):
        return InteractiveVisualizer.visualize_network(self, show_gradients=show_gradients)
    
    def visualize_weight_distribution(self, layers=None, include_bias=True):
        return InteractiveVisualizer.plot_weight_distribution(self, layers, include_bias)
    
    def visualize_gradient_weight_distribution(self, layers=None, include_bias=True):
        return InteractiveVisualizer.plot_gradient_weight_distribution(self, layers, include_bias)
    
    def visualize_loss_curve(self):
        return InteractiveVisualizer.plot_loss_curves(self.history)
    
    @staticmethod
    def visualize_weight_distribution_sklearn(model, layers=None):
        return InteractiveVisualizer.plot_weight_distribution_sklearn(model, layers)