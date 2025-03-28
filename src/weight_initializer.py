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
        biases = np.random.uniform(lower_bound, upper_bound, (1, number_of_neuron))
        return weights, biases
    
    @staticmethod
    def random_normal_initializer(input_size, number_of_neuron, mean=0, variance=0.1, seed=None) :
        if seed is not None:
            np.random.seed(seed)
        
        standard_deviation = np.sqrt(variance)

        weights = np.random.normal(mean, standard_deviation, (input_size, number_of_neuron))
        biases = np.random.normal(mean, standard_deviation, (1, number_of_neuron))
        return weights, biases
    
    @staticmethod
    def xavier_initializer(input_size, number_of_neuron, type="normal", seed=None) :
        if seed is not None:
            np.random.seed(seed)

        if (type == "normal") :
            standard_deviation = np.sqrt(2.0 / input_size + number_of_neuron)
            weights = np.random.normal(0.0, standard_deviation, (input_size, number_of_neuron))
            biases = np.random.normal(0.0, standard_deviation, (1, number_of_neuron))
        elif (type == "uniform") :
            bound = np.sqrt(6.0 / (input_size + number_of_neuron))
            weights = np.random.uniform(-bound, bound, (input_size, number_of_neuron))
            biases = np.random.uniform(-bound, bound, (1, number_of_neuron))
        else : 
            raise ValueError("Type of Xavier Initialization isn't valid. It must be between 'normal' or 'uniform'")

        return weights, biases
    
    @staticmethod
    def he_initializer(input_size, number_of_neuron, type="normal",seed=None) :
        if seed is not None:
            np.random.seed(seed)

        if (type == "normal") :
            standard_deviation = np.sqrt(2.0 / input_size)
            weights = np.random.normal(0.0, standard_deviation, (input_size, number_of_neuron))
            biases = np.random.normal(0.0, standard_deviation, (1, number_of_neuron))
        elif (type == "uniform") :
            bound = np.sqrt(6.0 / input_size)
            weights = np.random.uniform(-bound, bound, (input_size, number_of_neuron))
            biases = np.random.uniform(-bound, bound, (1, number_of_neuron))
        else : 
            raise ValueError("Type of He Initialization isn't valid. It must be between 'normal' or 'uniform'")

        return weights, biases