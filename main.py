import numpy as np
import pandas as pd
from network import * 
from data import *
from optimizer import *
import matplotlib.pyplot as plt

inputs = 10
outputs = 1

# Hyperparameters, hidden layer shape, activation, learning rate

network = Network([Dense(inputs, 20), Sigmoid(), Dense(20, outputs), Sigmoid() ], inputs, outputs)

# Parameters for the simulation
initial_price = 100     # Initial stock price
drift = 0.1             # Drift coefficient (10% annual return)
volatility = 0.2        # Volatility (20% annual volatility
time_period = 1         # Total time period (1 year)
time_step = 1 / 252     # Time step (daily)

time_data, stock_data = simulate_stock_prices(initial_price, drift, volatility, time_period, time_step)

stock_data, min_max_dict = normalization_dataframe(stock_data)

stock_data_list, x_train, y_train = format_data_windows(stock_data, inputs)

grid_optimizer(network, x_train, y_train, 10, [0.01,0.1,0.001], [10,20,40,80], [Sigmoid(), Tanh()])
