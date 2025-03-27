import numpy as np
from weight_initializer import WeightInitializer
from activation_function import ActivationFunction
from loss_function import LossFunction
from network_visualizer import NetworkVisualizer

class FFNN:
    def __init__(self, layer_sizes, activation="relu", weight_init="xavier", loss_function="mse", seed=None):
        self.layer_sizes = layer_sizes # ukuran layer, cth: [input_size, hidden1, hidden2, output_size]
        self.activation = activation # fungsi aktivasi
        self.loss_function = loss_function  # fungsi loss
        self.weight_init = weight_init # metode inisialisasi bobot ('zero', 'uniform', 'normal', 'xavier', 'he')
        self.seed = seed

        # inisialisasi
        self.weights = {}
        self.biases = {}
        self.sigma = {}  # untuk menyimpan nilai-nilai untuk backward propagation
        self.o = {}  # untuk menyimpan nilai-nilai untuk backward propagation
        
        self.weights, self.biases = self.setup_weights()

        # Inisialisasi gradien bobot dan bias
        self.weight_gradient = {}
        self.bias_gradient = {}

        for i in range(1, len(self.layer_sizes)):
            self.weight_gradient[i] = np.zeros_like(self.weights[i])  # Gradien bobot
            self.bias_gradient[i] = np.zeros_like(self.biases[i])  # Gradien bias
    
    def setup_weights(self):
        weights = {}
        biases = {}

        if self.weight_init == "uniform":
            upper_bound = float(input("Masukkan batas bawah (default = -0.5) = "))
            lower_bound = float(input("Masukkan batas atas (default = 0.5) = "))
        elif self.weight_init == "normal":
            mean = float(input("Masukkan mean (default = 0) = "))
            var = float(input("Masukkan batas atas (default = 0.1) = "))
            
        # loop dari input layer sampau output layer
        for i in range(1, len(self.layer_sizes)):  
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]

            if self.weight_init == "zero":
                W, b = WeightInitializer.zero_initializer(input_size, output_size)
            elif self.weight_init == "uniform":
                W, b = WeightInitializer.random_uniform_initializer(input_size, output_size, lower_bound, upper_bound, seed=self.seed)
            elif self.weight_init == "normal":
                W, b = WeightInitializer.random_normal_initializer(input_size, output_size, mean, var, seed=self.seed)
            elif self.weight_init == "xavier":
                W, b = WeightInitializer.xavier_initializer(input_size, output_size, seed=self.seed)
            elif self.weight_init == "he":
                W, b = WeightInitializer.he_initializer(input_size, output_size, seed=self.seed)
            else:
                raise ValueError("Metode inisialisasi tidak dikenal.")

            weights[i] = W
            biases[i] = b
        
        return weights, biases
    
    def compute_loss(self, y_true, y_pred):
        if self.loss_function == "mse":
            return LossFunction.mse(y_true, y_pred)
        elif self.loss_function == "binary_cross_entropy":
            return LossFunction.binary_cross_entropy(y_true, y_pred)
        elif self.loss_function == "categorical_cross_entropy":
            return LossFunction.categorical_cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Loss function '{self.loss_function}' not recognized.")
        
    def forward(self, x):
        o = x
        self.o[0] =  o

        for i in range(1, len(self.layer_sizes)):
            W = self.weights[i]
            b = self.biases[i]

            sigma = np.dot(o, W) + b
        
            # Apply activation function
            activation_name = self.activation[i-1]
            activation = getattr(ActivationFunction, activation_name)
            o = activation(sigma)


            self.sigma[i] = sigma
            self.o[i] = o
        print(self.sigma)
        print(self.o)
        return o

    def print_model(self):
        print("Model Structure:")
        print(f"Input Layer: {self.layer_sizes[0]} neurons")
        for i in range(1, len(self.layer_sizes)):
            print(f"Layer {i}:")
            print(f"  - Neurons: {self.layer_sizes[i]}")
            print(f"  - Activation: {self.activation}")
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
    def visualize_model(self, show_weights=True, show_gradients=True, figsize=(12, 10), enable_zoom=True):
        return NetworkVisualizer.visualize_network(self, show_weights, show_gradients, figsize, enable_zoom)


# # print bobot
# print(ffnn.weights)  
# print(ffnn.biases)

if __name__ == "__main__":
    # Create a simple network with 2 inputs, 3 neurons in hidden layer, and 1 output
    # layer_sizes = [2, 3,3, 1]
    # model = FFNN(layer_sizes, activation=["linear"], loss_function="mse", weight_init="zero", seed=42)
    
    # # Print model structure
    # model.print_model()


    layer_sizes = [3, 2, 2, 3, 1, 2, 2, 3, 1, 2, 3, 1]
    activations= ["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"]
    batch = np.random.rand(1, 3)
    print(batch)
    ffnn = FFNN(layer_sizes, activation= activations, loss_function="mse", weight_init="zero", seed=42)
    ffnn.print_model()
    print("result:")
    print(ffnn.forward(batch))
    ffnn.visualize_model()