from network import *
import numpy as np
from random import randint

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
    lowest_error = 10000
    optimal_parameters = {}
    for i, activ in enumerate(activations):
        for j, rate in enumerate(learning_rates):
            for k, batch in enumerate(batch_sizes):
                change_network_activations(network, activ)
                error = network.train(mse, mse_prime, x_train, y_train, epoachs = epoachs, learning_rate=rate, batch_size=batch, verbose=False)[-1]
                if error < lowest_error:
                    lowest_error = error
                    optimal_parameters = {"activation": activ, "learning_rate": rate, "batch_size": batch}
    return optimal_parameters

def create_params(learning_rates, batch_sizes, activations):
    param_list=[]
    for i, rate in enumerate(learning_rates):
        for j, activ in enumerate(activations):
            for k, batch in enumerate(batch_sizes):
                param_list.append( (rate, activ, batch))
    return param_list


def random_optimizer(network, x_train, y_train, epoachs, n_random_checks, learning_rates, batch_sizes, activations):
    param_list = create_params(learning_rates, batch_sizes, activations)
    lowest_error = 10000
    print(len(param_list))
    for n in range(0,n_random_checks):
        print(n)
        n_param = len(param_list)
        rate, activ, batch = param_list.pop(randint(0, n_param-1))
        change_network_activations(network, activ)
        error = network.train(mse, mse_prime, x_train, y_train, epoachs = epoachs, learning_rate=rate, batch_size=batch, verbose=False)[-1]
        if error < lowest_error:
            lowest_error = error
            optimal_parameters = {"activation": activ, "learning_rate": rate, "batch_size": batch}
    return optimal_parameters, lowest_error