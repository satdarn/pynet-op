import numpy as np
import pandas as pd
from network import * 
from data import *
from optimizer import *
import matplotlib.pyplot as plt

inputs = 10
outputs = 1

# Hyperparameters, hidden layer shape, activation, learning rate, batch size

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

batch_sizes = [i for i in range(75, 85) ]
learning_rate = [i/1000 for i in range(700,900)]

errors = network.train(mse, mse_prime, x_train, y_train, epoachs=3000, batch_size=82, learning_rate=0.84, verbose=False)
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(stock_data_list, marker='o', linestyle='-', color='b', label='Error Values')
plt.xlabel('Index')  # Label for x-axis
plt.ylabel('Error Value')  # Label for y-axis
plt.title('Plot of Error Values')  # Title of the plot
plt.legend()  # Display legend
plt.grid(True)  # Show grid lines
plt.show()  # Display the plot