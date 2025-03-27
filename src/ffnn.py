import numpy as np
import json, os
from weight_initializer import WeightInitializer
from activation_function import ActivationFunction
from loss_function import LossFunction
from backpropagation import BackPropagation

class FFNN:
    def __init__(self, layer_sizes, activation=None, weight_init="xavier", loss_function="mse", seed=None):
        self.layer_sizes = layer_sizes


        if isinstance(activation, str):
            self.activation = [activation] * (len(layer_sizes) - 1)
        elif activation is None:
            self.activation = ["relu"] * (len(layer_sizes) - 1)
        else:
            if len(activation) != len(layer_sizes) - 1:
                raise ValueError(f"Activation list length ({len(activation)}) must match number of layers - 1 ({len(layer_sizes) - 1})")
            self.activation = activation
            
        self.loss_function = loss_function
        self.weight_init = weight_init
        self.seed = seed


        self.weights = {}
        self.biases = {}
        self.weight_gradients = {}
        self.bias_gradients = {}
        self.sigma = {}  
        self.o = {} 
        

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
                    params.get("lower_bound", -0.5), 
                    params.get("upper_bound", 0.5), 
                    seed=self.seed
                )
            elif self.weight_init == "normal":
                W, b = WeightInitializer.random_normal_initializer(
                    input_size, output_size, 
                    params.get("mean", 0), 
                    params.get("variance", 0.1), 
                    seed=self.seed
                )
            elif self.weight_init == "xavier":
                W, b = WeightInitializer.xavier_initializer(input_size, output_size, seed=self.seed)
            elif self.weight_init == "he":
                W, b = WeightInitializer.he_initializer(input_size, output_size, seed=self.seed)
            else:
                raise ValueError(f"Weight initialization method '{self.weight_init}' not recognized.")

            weights[i] = W
            biases[i] = b
            
            # Initialize gradient storage
            self.weight_gradients[i] = np.zeros_like(W)
            self.bias_gradients[i] = np.zeros_like(b)
        
        return weights, biases
    
    def compute_loss(self, y_true, y_pred):
        if self.loss_function == "mse":
            return LossFunction.mse(y_true, y_pred)
        elif self.loss_function == "binary_cross_entropy":
            return LossFunction.binary_cross_entropy(y_true, y_pred)
        elif self.loss_function == "categorical_cross_entropy":
            return LossFunction.categorical_cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Loss function '{self.loss_function}' not found.")
        
    def forward(self, x):
        o = x
        self.o[0] =  o

        for i in range(1, len(self.layer_sizes)):
            W = self.weights[i]
            b = self.biases[i]

            sigma = np.dot(o, W) + b
            
            # terus diaktivasi
            activation_name = self.activation[i-1]
            if hasattr(ActivationFunction, activation_name):
                activation_func = getattr(ActivationFunction, activation_name)
                o = activation_func(sigma)
            else:
                raise ValueError(f"Activation function '{activation_name}' not found.")

            self.sigma[i] = sigma
            self.o[i] = o
            
        return o
    
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
            # Shuffle data
            if self.seed is not None:
                np.random.seed(self.seed + epoch)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = self.backward(X_batch, y_batch, learning_rate)
                epoch_losses.append(batch_loss)
            
            avg_train_loss = np.mean(epoch_losses)
            self.history["train_loss"].append(avg_train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.history["val_loss"].append(val_loss)
            
            # # Print progress
            # if verbose == 1:
            #     if X_val is not None and y_val is not None:
            #         print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
            #     else:
            #         print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return self.history
    
    def predict(self, X):
        return self.forward(X)
    
    def print_model(self):
        """
        Print model structure with weights and biases
        """
        print("Model Structure:")
        print(f"Input Layer: {self.layer_sizes[0]} neurons")
        for i in range(1, len(self.layer_sizes)):
            print(f"Layer {i}:")
            print(f"  - Neurons: {self.layer_sizes[i]}")
            print(f"  - Activation: {self.activation[i-1]}")
            print(f"  - Weights shape: {self.weights[i].shape}")
            print(f"  - Bias shape: {self.biases[i].shape}")
            
            # Print a small sample of weights and gradients if they are large
            if self.weights[i].size > 10:
                print(f"  - Sample weights: {self.weights[i].flatten()[:5]} ...")
            else:
                print(f"  - Weights: {self.weights[i]}")
            if self.biases[i].size > 10:
                print(f"  - Sample biases: {self.biases[i].flatten()[:5]} ...")
            else:
                print(f"  - Biases: {self.biases[i]}")
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "weight_init": self.weight_init,
            "loss_function": self.loss_function,
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
            activation=model_data["activation"],
            weight_init=model_data["weight_init"],
            loss_function=model_data["loss_function"],
            seed=model_data["seed"]
        )

        model.weights = {int(k): np.array(v) for k, v in model_data["weights"].items()}
        model.biases = {int(k): np.array(v) for k, v in model_data["biases"].items()}

        print(f"Model loaded from {filepath} 💾")
        return model

if __name__ == "__main__":
    # Testing
    layer_sizes = [2, 20,1]
    activations = ["relu", "sigmoid"]
    
    # Sample data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    model = FFNN(layer_sizes, activation=activations, loss_function="mse", weight_init="he", seed=42)
    model.print_model()
    history = model.train(X, y, epochs=10000, learning_rate=0.1, batch_size=4, verbose=1)
    model.print_model()
    
    predictions = model.predict(X)
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i]} -> Predicted: {pred[0]:.4f}, Actual: {y[i][0]}")

    model.save("model/model.json")
    print("======================================== loaded model ========================================")
    loaded_model = FFNN.load("model/model.json")
    loaded_model.print_model()
    new_predictions = loaded_model.predict(X)
    print("\nPredictions:")
    for i, pred in enumerate(new_predictions):
        print(f"Input: {X[i]} -> Predicted: {pred[0]:.4f}, Actual: {y[i][0]}")