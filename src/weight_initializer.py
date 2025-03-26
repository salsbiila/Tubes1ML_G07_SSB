import numpy as np

class WeightInitializer :
    @staticmethod
    def zero_initializer(input_size, number_of_neuron) :
        weights = np.zeros((input_size, number_of_neuron))
        biases = np.zeros((1, number_of_neuron))
        return weights, biases

    @staticmethod
    def random_uniform_initializer(input_size, number_of_neuron, lower_bound=-0.5, upper_bound=0.5, seed=None) :
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.random.uniform(lower_bound, upper_bound, (input_size, number_of_neuron))
        biases = np.zeros((1, number_of_neuron))
        return weights, biases
    
    @staticmethod
    def random_normal_initializer(input_size, number_of_neuron, mean=0, variance=0.1, seed=None) :
        if seed is not None:
            np.random.seed(seed)
        
        standard_deviation = np.sqrt(variance)

        weights = np.random.normal(mean, standard_deviation, (input_size, number_of_neuron))
        biases = np.zeros((1, number_of_neuron))
        return weights, biases
    
    @staticmethod
    def xavier_initializer(input_size, number_of_neuron, seed=None) :
        if seed is not None:
            np.random.seed(seed)

        bound = np.sqrt(6.0 / (input_size + number_of_neuron))

        weights = np.random.uniform(-bound, bound, (input_size, number_of_neuron))
        biases = np.zeros((1, number_of_neuron))
        return weights, biases
    
    @staticmethod
    def he_initializer(input_size, number_of_neuron, seed=None) :
        if seed is not None:
            np.random.seed(seed)

        standard_deviation = np.sqrt(2.0 / input_size)

        weights = np.random.normal(0.0, standard_deviation, (input_size, number_of_neuron))
        biases = np.zeros((1, number_of_neuron))
        return weights, biases
    

# input_size = 5
# number_of_neuron = 3

# # Zero Initializer
# weights, biases = WeightInitializer.zero_initializer(input_size, number_of_neuron)
# print("Zero Initializer:\n", "Weights:\n", weights, "\nBiases:\n", biases, "\n")

# # Random Uniform Initializer
# weights, biases = WeightInitializer.random_uniform_initializer(input_size, number_of_neuron, seed=42)
# print("Random Uniform Initializer:\n", "Weights:\n", weights, "\nBiases:\n", biases, "\n")

# # Random Normal Initializer
# weights, biases = WeightInitializer.random_normal_initializer(input_size, number_of_neuron, seed=42)
# print("Random Normal Initializer:\n", "Weights:\n", weights, "\nBiases:\n", biases, "\n")

# # Xavier Initializer
# weights, biases = WeightInitializer.xavier_initializer(input_size, number_of_neuron, seed=42)
# print("Xavier Initializer:\n", "Weights:\n", weights, "\nBiases:\n", biases, "\n")

# # He Initializer
# weights, biases = WeightInitializer.he_initializer(input_size, number_of_neuron, seed=42)
# print("He Initializer:\n", "Weights:\n", weights, "\nBiases:\n", biases, "\n")
