
import pandas as pd
df = pd.read_csv("not_final3.csv")
df = df.rename(columns={'Sport_-1': 'Sport'})
print("Dataset loaded successfully.\n")
print("Columns in the dataset:")
print(df.columns)
print("\nPreview of the dataset:")
print(df.head())
print("\nData types of each column:")
print(df.dtypes)
print("\nMissing values summary:")
print(df.isnull().sum())
parameter = 'drug'
sport_name = 'Athletics'  # Example sport
raw_data = df[df['Sport'] == 'Athletics'].filter(regex=f'{parameter}_\d+')
raw_melted = raw_data.melt(var_name='Year', value_name='Value')
raw_melted['Year'] = raw_melted['Year'].str.extract('(\d+)').astype(int)
print("\nRaw data for Athletics (drug):")
print(raw_melted.sort_values('Year'))
ts_data_check = raw_melted.set_index('Year')['Value'].sort_index()
ts_data_check = ts_data_check.fillna(0)
print("\nRechecked time series data for Athletics (drug):")
print(ts_data_check)
parameter = 'equity'
sport_name = 'Athletics'
param_columns = [col for col in df.columns if col.startswith(f"{parameter}_")]
melted_data_equity = df[['Sport'] + param_columns].melt(id_vars='Sport', var_name='Year', value_name='Value')
melted_data_equity['Year'] = melted_data_equity['Year'].str.extract('(\d+)').astype(int)
sport_data_equity = melted_data_equity[melted_data_equity['Sport'] == sport_name].sort_values('Year')
print(f"Raw data for {sport_name} ({parameter}):")
print(sport_data_equity)
import pandas as pd
import numpy as np
df = pd.read_csv("not_final3.csv")
df = df.rename(columns={'Sport_-1': 'Sport'})
excluded_sports = ['climbing', 'Fitness', 'Headis']
df = df[~df['Sport'].isin(excluded_sports)]
parameters = ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']
for parameter in parameters:
    print(f"\nProcessing parameter: {parameter}")
    param_columns = [col for col in df.columns if col.startswith(f"{parameter}_")]
    melted_data = df[['Sport'] + param_columns].melt(id_vars='Sport', var_name='Year', value_name='Value')
    melted_data['Year'] = melted_data['Year'].str.extract(r'(\d+)').astype(int)  # Use raw string
    for sport_name in melted_data['Sport'].unique():
        print(f"\nProcessing sport: {sport_name} ({parameter})")
        sport_data = melted_data[melted_data['Sport'] == sport_name].sort_values('Year')
        if parameter == 'drug':
            sport_data['Value'] = sport_data['Value'].fillna(0)
        elif parameter in ['CV', 'popularity']:
            sport_data = sport_data[sport_data['Value'] != 0].dropna(subset=['Value'])
        else:
            sport_data = sport_data.dropna(subset=['Value'])
        ts_data = sport_data.set_index('Year')['Value']
        print(f"Raw data for {sport_name} ({parameter}):\n", sport_data)
        print(f"\nPrepared time series data for {sport_name} ({parameter}):\n", ts_data)
melted_data
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def incremental_backtest(ts_data, model_func, order, seasonal_order=None, sport_name="", parameter=""):
    predictions = []
    actuals = []
    indices = []
    for t in range(len(ts_data) - 1):
        train = ts_data.iloc[: t + 1]
        if len(train) <= max(order):
            continue
        if model_func == ARIMA:
            model = model_func(train, order=order)
        elif model_func == SARIMAX:
            model = model_func(train, order=order, seasonal_order=seasonal_order)
        else:
            raise ValueError("Unsupported model function")
        try:
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            forecast_value = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
        except ValueError as e:
            print(f"Error for {sport_name} ({parameter}): {e}")
            continue
        predictions.append(forecast_value)
        actuals.append(ts_data.iloc[t + 1])
        indices.append(ts_data.index[t + 1])
    mse = mean_squared_error(actuals, predictions)
    print(f"\nMSE for {sport_name} ({parameter}): {mse:.4f}")
    plt.figure(figsize=(12, 6))
    plt.plot(indices, actuals, label="Actual", marker="o")
    plt.plot(indices, predictions, label=f"{model_func.__name__} Predicted", linestyle="--", marker="x")
    plt.title(f"{model_func.__name__} Predictions for {sport_name} ({parameter})")
    plt.xlabel("Year")
    plt.ylabel(f"{parameter} Values")
    plt.legend()
    plt.grid(True)
    plt.show()
    return mse
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima.arima import auto_arima
import warnings
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import logging
warnings.filterwarnings('ignore')
log_file = "model_processing.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
output_dir = "model_plots"
os.makedirs(output_dir, exist_ok=True)
MIN_DATA_POINTS = 5  # Minimum data points to fit the model
metrics = []
def log_message(message):
    print(message)  # For console viewing (optional)
    logging.info(message)  # For saving into the log file
def incremental_backtest(ts_data, model_func, order, seasonal_order=None, sport_name="", parameter=""):
    predictions, actuals, indices = [], [], []
    for t in range(len(ts_data) - 1):
        train = ts_data.iloc[:t + 1]
        if len(train) <= max(order):
            continue
        if np.all(train == train.iloc[0]):  # Constant series detected
            constant_value = train.iloc[0]
            log_message(f"Constant series detected for {sport_name} ({parameter}). Predicting constant value: {constant_value}")
            predictions.append(constant_value)
            actuals.append(ts_data.iloc[t + 1])
            indices.append(ts_data.index[t + 1])
            continue
        if model_func == ARIMA:
            model = model_func(train, order=order)
        elif model_func == SARIMAX:
            model = model_func(train, order=order, seasonal_order=seasonal_order)
        else:
            raise ValueError("Unsupported model function")
        try:
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            forecast_value = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
            predictions.append(forecast_value)
            actuals.append(ts_data.iloc[t + 1])
            indices.append(ts_data.index[t + 1])
        except Exception as e:
            log_message(f"Fitting error for {sport_name} ({parameter}): {e}")
            break
    if predictions:
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        metrics.append({
            'Sport': sport_name,
            'Parameter': parameter,
            'Model': model_func.__name__,
            'MSE': mse,
            'R2': r2
        })
        log_message(f"Metrics for {sport_name} ({parameter}) - MSE: {mse}, R2: {r2}")
        plt.figure(figsize=(10, 6))
        plt.plot(indices, actuals, label='Actual', marker='o', linestyle='-')
        plt.plot(indices, predictions, label=f'{model_func.__name__} Predicted', marker='x', linestyle='--')
        plt.title(f'Actual vs Predicted for {sport_name} ({parameter})')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{sport_name}_{parameter}_{model_func.__name__}.png")
        plt.close()
    else:
        log_message(f"No predictions made for {sport_name} ({parameter}).")
def check_stationarity(ts_data):
    if np.all(ts_data == ts_data.iloc[0]):
        log_message("Series is constant, skipping stationarity test.")
        return False
    result = adfuller(ts_data)
    log_message(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    return result[1] <= 0.05  # True if stationary
df = pd.read_csv("not_final3.csv")
df = df.rename(columns={'Sport_-1': 'Sport'})
parameters = ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']
for parameter in parameters:
    log_message(f"\nProcessing parameter: {parameter}")
    param_columns = [col for col in df.columns if col.startswith(f"{parameter}_")]
    melted_data = df[['Sport'] + param_columns].melt(id_vars='Sport', var_name='Year', value_name='Value')
    melted_data['Year'] = melted_data['Year'].str.extract(r'(\d+)').astype(int)
    for sport_name in melted_data['Sport'].unique():
        sport_data = melted_data[melted_data['Sport'] == sport_name].sort_values('Year')
        if parameter == 'drug':
            sport_data['Value'] = sport_data['Value'].fillna(0)
        elif parameter in ['CV', 'popularity']:
            sport_data = sport_data[sport_data['Value'] != 0].dropna(subset=['Value'])
        else:
            sport_data = sport_data.dropna(subset=['Value'])
        if sport_data['Value'].sum() == 0:
            log_message(f"All zero values for {sport_name} ({parameter}), skipping.")
            continue
        ts_data = sport_data.set_index('Year')['Value']
        if len(ts_data) < MIN_DATA_POINTS:
            log_message(f"Insufficient data for {sport_name} ({parameter})")
            continue
        log_message(f"Checking stationarity for {sport_name} ({parameter}):")
        is_stationary = check_stationarity(ts_data)
        if not is_stationary:
            log_message(f"Skipping ARIMA fitting for non-stationary series: {sport_name} ({parameter})")
            continue
        auto_arima_model = auto_arima(ts_data, seasonal=False, trace=False)
        arima_order = auto_arima_model.order
        auto_sarima_model = auto_arima(ts_data, seasonal=True, m=1, trace=False)
        sarima_order = auto_sarima_model.order
        sarima_seasonal_order = auto_sarima_model.seasonal_order
        log_message(f"Optimal ARIMA order: {arima_order}")
        log_message(f"Optimal SARIMA order: {sarima_order}, Seasonal: {sarima_seasonal_order}")
        incremental_backtest(ts_data, ARIMA, arima_order, sport_name=sport_name, parameter=parameter)
        incremental_backtest(ts_data, SARIMAX, sarima_order, seasonal_order=sarima_seasonal_order, sport_name=sport_name, parameter=parameter)
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics.csv', index=False)
log_message("\nMetrics saved to 'model_metrics.csv'")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
metrics_df = pd.read_csv('model_metrics.csv')
def create_model_matrix(metrics_df, metric, model_type):
    pivot_df = metrics_df[metrics_df['Model'] == model_type].pivot(
        index='Sport', columns='Parameter', values=metric
    )
    return pivot_df
def plot_model_quality(matrix, title):
    if matrix.empty:
        print(f"No data available to plot for {title}.")
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': title})
    plt.title(f'{title} Heatmap')
    plt.xlabel('Parameter')
    plt.ylabel('Sport')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
arima_r2_matrix = create_model_matrix(metrics_df, 'R2', 'ARIMA')
plot_model_quality(arima_r2_matrix, 'ARIMA R²')
arima_mse_matrix = create_model_matrix(metrics_df, 'MSE', 'ARIMA')
plot_model_quality(arima_mse_matrix, 'ARIMA MSE')
sarima_r2_matrix = create_model_matrix(metrics_df, 'R2', 'SARIMAX')
plot_model_quality(sarima_r2_matrix, 'SARIMA R²')
sarima_mse_matrix = create_model_matrix(metrics_df, 'MSE', 'SARIMAX')
plot_model_quality(sarima_mse_matrix, 'SARIMA MSE')
