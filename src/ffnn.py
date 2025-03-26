import numpy as np
from weight_initializer import WeightInitializer

class FFNN:
    def __init__(self, layer_sizes, activation="relu", weight_init="xavier", seed=None):
        self.layer_sizes = layer_sizes # ukuran layer, cth: [input_size, hidden1, hidden2, output_size]
        self.activation = activation # fungsi aktivasi
        self.weight_init = weight_init # metode inisialisasi bobot ('zero', 'uniform', 'normal', 'xavier', 'he')
        self.seed = seed

        # inisialisasi bobot dan bias
        self.weights, self.biases = self.setup_weights()
    
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

            weights[f"W{i}"] = W
            biases[f"b{i}"] = b
        
        return weights, biases

layer_sizes = [3, 5, 2]  # 3 input, 5 hidden, 2 output
ffnn = FFNN(layer_sizes, weight_init="uniform", seed=42)

# print bobot
print(ffnn.weights)  
print(ffnn.biases)