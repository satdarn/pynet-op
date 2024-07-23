from network import *
class optimizer_settings:
    def __init__(self, algorithm, learning_rate = False, batch_size = False, activation = False):
        self.algorithm = algorithm
        self.optimization_params = []
        if learning_rate:
            self.learning_rates = []
            self.optimization_params.append("learning_rate")
        if batch_size:
            self.batch_sizes = []
            self.optimization_params.append("batch_size")
        if activation:
            self.activations = []
            self.optimization_params.append("activation")
    def set_learning_rates(self, learning_rates):
        try:
            self.learning_rates = learning_rates
        except:
            print("Learning Rate is not chosen as a Optimizable Parameter")
    def set_batch_sizes(self, batch_sizes):
        try:
            self.batch_sizes = batch_sizes
        except:
            print("Batch Size is not chosen as a Optimizable Parameter")
    def set_activations(self, activations):
        try:
            self.activations = activations
        except:
            print("Activations is not chosen as a Optimizable Parameter")
        

def change_network_activations(network, activation):
    for i, layer in enumerate(network.network):
        if isinstance(layer, Activation):
            network.network[i] = activation

class optimizer_settings:
    def __init__(self, algorithm, learning_rate = False, batch_size = False, activation = False):
        self.algorithm = algorithm
        self.optimization_params = []
        if learning_rate:
            self.learning_rates = []
            self.optimization_params.append("learning_rate")
        if batch_size:
            self.batch_sizes = []
            self.optimization_params.append("batch_size")
        if activation:
            self.activations = []
            self.optimization_params.append("activation")
    def set_learning_rates(self, learning_rates):
        try:
            self.learning_rates = learning_rates
        except:
            print("Learning Rate is not chosen as a Optimizable Parameter")
    def set_batch_sizes(self, batch_sizes):
        try:
            self.batch_sizes = batch_sizes
        except:
            print("Batch Size is not chosen as a Optimizable Parameter")
    def set_activations(self, activations):
        try:
            self.activations = activations
        except:
            print("Activations is not chosen as a Optimizable Parameter")
        

def change_network_activations(network, activation):
    for i, layer in enumerate(network.network):
        if isinstance(layer, Activation):
            network.network[i] = activation

def grid_optimizer(network, x_train, y_train, epoachs, learning_rates, batch_sizes, activations):
    error_matrix = np.zeros((len(activations), len(learning_rates), len(batch_sizes)))
    for i, activ in enumerate(activations):
        for j, rate in enumerate(learning_rates):
            for k, batch in enumerate(batch_sizes):
                change_network_activations(network, activ)
                error_matrix[i, j, k] = network.train(mse, mse_prime, x_train, y_train, epochs = 10, learning_rate=rate, batch_size=batch, verbose=False)[-1]
    min_index_flat = np.argmin(error_matrix)
    print(min_index_flat)
