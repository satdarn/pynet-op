import datetime as dt
import numpy as np
import pandas as pd

def normalization_dataframe(
        df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_normalized = df.copy()
    min_max_values = {}

    for column in df_normalized.columns:
        min_value = df_normalized[column].min()
        max_value = df_normalized[column].max()

        min_max_values[column] = (min_value, max_value)

        # Avoid division by zero if all values are the same
        if min_value == max_value:
            df_normalized[column] = 0.0
        else:
            df_normalized[column] = (df_normalized[column] -
                                     min_value) / (max_value - min_value)

    return df_normalized, min_max_values


def denormalize_dataframe(df_normalized: pd.DataFrame,
                          min_max_values: pd.DataFrame) -> pd.DataFrame:
    df_denormalized = df_normalized.copy()

    for column in df_denormalized.columns:
        min_value, max_value = min_max_values[column]
        df_denormalized[column] = df_denormalized[column] * (
            max_value - min_value) + min_value

    return df_denormalized

def format_data_windows(data: pd.DataFrame, prediction_window: int) -> tuple[list, list, list]:
    x_train = []
    y_train = []
    data_np = np.array(data)
    for i in range(len(data_np) - prediction_window - 2):
        x_train.append(data_np[i:i + prediction_window].flatten().T)
        y_train.append(data_np[i + prediction_window + 1])
    return data_np.tolist(), x_train, y_train

def simulate_stock_prices(initial_price, drift, volatility, time_period, time_step):

    # Calculate the number of time steps
    num_steps = int(time_period / time_step)
    
    # Initialize arrays to store time points and stock prices
    time_points = np.linspace(0, time_period, num_steps)
    stock_prices = np.zeros(num_steps)
    
    # Set initial stock price
    stock_prices[0] = initial_price
    
    # Generate random standard normal variables for the Brownian motion
    random_shocks = np.random.randn(num_steps)
    
    # Simulate stock prices using the GBM model
    for step in range(1, num_steps):
        stock_prices[step] = stock_prices[step-1] * np.exp(
            (drift - 0.5 * volatility**2) * time_step +
            volatility * np.sqrt(time_step) * random_shocks[step]
        )
    stock_prices = pd.DataFrame(stock_prices)
    
    return time_points, stock_prices



