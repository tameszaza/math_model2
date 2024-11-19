
import pandas as pd
import numpy as np
df = pd.read_csv("not_final3.csv")
df = df.rename(columns={'Sport_-1': 'Sport'})
excluded_sports = ['climbing', 'Fitness', 'Headis']
df = df[~df['Sport'].isin(excluded_sports)]
parameters = ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV', 'label']
sport_data_dict = {}
for sport_name in df['Sport'].unique():
    sport_dict = {'Year': []}
    all_years = pd.Series(dtype=int)
    for parameter in parameters:
        param_columns = [col for col in df.columns if col.startswith(f"{parameter}_")]
        if not param_columns:
            continue
        melted_data = df[df['Sport'] == sport_name][['Sport'] + param_columns].melt(
            id_vars='Sport', var_name='Year', value_name=f'{parameter}_Value'
        )
        melted_data['Year'] = melted_data['Year'].str.extract(r'(\d+)').astype(int)  # Extract year as int
        melted_data = melted_data.sort_values('Year')
        if all_years.empty:
            all_years = melted_data['Year']
        else:
            all_years = pd.concat([all_years, melted_data['Year']]).drop_duplicates().sort_values()
        melted_data = melted_data.set_index('Year').reindex(all_years).reset_index()
        if parameter == 'drug':
            melted_data[f'{parameter}_Value'] = melted_data[f'{parameter}_Value'].fillna(0)  # For drug, fill missing with 0
        elif parameter in ['popularity', 'CV']:
            melted_data[f'{parameter}_Value'] = melted_data[f'{parameter}_Value'].replace(0, np.nan)
        else:
            melted_data[f'{parameter}_Value'] = melted_data[f'{parameter}_Value']
        sport_dict[parameter] = melted_data[f'{parameter}_Value'].values
    sport_dict['Year'] = all_years.values
    sport_data_dict[sport_name] = pd.DataFrame(sport_dict)
for sport, data in sport_data_dict.items():
    data.set_index('Year', inplace=True)  # Ensure Year is the index
    print(f"\nData for {sport}:")
    print(data.head())
sport_data_dict['Athletics']
import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("model_plots", exist_ok=True)
os.makedirs("model_metrics", exist_ok=True)
os.makedirs("model_weights", exist_ok=True)
metrics_data = []
def moving_average_predict(train_data, window_size):
    return train_data[-window_size:].mean()
def train_and_evaluate_ma_model(train_data, test_data, parameter, save_path, window_sizes):
    best_r2 = -float('inf')
    best_mse = float('inf')
    best_window_size = None
    best_predictions = None
    for window_size in window_sizes:
        predictions = []
        actuals = []  # Initialize actuals for each window_size iteration
        extended_train_data = train_data.copy()
        for i in range(len(test_data)):
            if len(extended_train_data) >= window_size:
                prediction = moving_average_predict(extended_train_data[parameter], window_size)
            else:
                prediction = extended_train_data[parameter].mean()
            predictions.append(prediction)
            actuals.append(test_data[parameter].iloc[i])
            extended_train_data = pd.concat([extended_train_data, test_data.iloc[[i]]])
        r2 = r2_score(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        if r2 > best_r2:
            best_r2, best_mse = r2, mse
            best_window_size = window_size
            best_predictions = predictions
    model_file = f"{save_path}/MA_{parameter}_best.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump({"window_size": best_window_size}, f)
    return best_r2, best_mse, best_predictions, actuals, best_window_size
for sport, df in sport_data_dict.items():
    for parameter in ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']:
        if parameter not in df.columns:
            continue
        df.index = pd.PeriodIndex(df.index, freq='Y')
        parameter_df = df[[parameter]].dropna()
        if len(parameter_df) < 10:
            continue
        year_data = parameter_df.index
        train_data = parameter_df.iloc[:-1]
        test_data = parameter_df.iloc[1:]
        sport_model_path = f"model_weights/{sport}"
        os.makedirs(sport_model_path, exist_ok=True)
        window_sizes = range(2, min(10, len(train_data)))
        ma_r2, ma_mse, ma_predicted, ma_actuals, best_window_size = train_and_evaluate_ma_model(
            train_data, test_data, parameter, sport_model_path, window_sizes
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(year_data[1:].to_timestamp(), ma_actuals, label='Actuals', color='black')
        ax.plot(year_data[1:].to_timestamp(), ma_predicted, label=f'MA (window={best_window_size})', color='blue')
        ax.set_title(f'{sport} - {parameter}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Parameter Value')
        ax.grid(True)
        plt.legend()
        plt.savefig(f"model_plots/{sport}_{parameter}_MA_comparison.png")
        plt.close()
        metrics_data.append({
            'Sport': sport,
            'Parameter': parameter,
            'Model': 'Moving Average',
            'R2': ma_r2,
            'MSE': ma_mse,
            'Best Window Size': best_window_size
        })
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('model_metrics/metrics_ma_summary.csv', index=False)
print("Training and evaluation completed. Models and metrics saved.")
print(f"Year Data Type: {type(year_data)}")
print(f"Year Data Example: {year_data[:5]}")
print(f"ARIMA Actuals Type: {type(arima_actuals)}")
print(f"ARIMA Actuals Example: {arima_actuals[:5]}")
sport = 'Athletics'
parameter = 'drug'
df = sport_data_dict[sport]
df.index = pd.PeriodIndex(df.index, freq='Y')
parameter_df = df[[parameter]].dropna()
year_data = parameter_df.index
year_data_plot = year_data.to_timestamp()  # Conversion for plotting
train_data = parameter_df.iloc[:-1]
test_data = parameter_df.iloc[1:]
arima_model = sm.tsa.ARIMA(train_data[parameter], order=(1, 1, 1))
model_fit = arima_model.fit()
arima_predicted = model_fit.forecast(steps=len(test_data))
arima_actuals = test_data[parameter]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year_data_plot[1:], arima_actuals, label='Actuals', color='black')
ax.plot(year_data_plot[1:], arima_predicted, label='ARIMA', color='green')
ax.set_title(f'Test Plot: {sport} - {parameter}')
ax.set_xlabel('Year')
ax.set_ylabel('Parameter Value')
plt.legend()
plt.show()
from statsmodels.tsa.stattools import adfuller
result = adfuller(train_data[parameter])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] > 0.05:
    print("Data is non-stationary. Consider differencing.")
else:
    print("Data is stationary.")
import pandas as pd
import matplotlib.pyplot as plt
metrics_df = pd.read_csv('model_metrics/metrics_summary.csv')
print(metrics_df.head())  # Preview the first few rows
r2_scores = metrics_df.pivot_table(values='R2', index=['Sport', 'Parameter'], columns='Model')
r2_scores.plot(kind='bar', figsize=(50, 15))
plt.title('Model Comparison by R² Score')
plt.xlabel('Sport and Parameter')
plt.ylabel('R² Score')
plt.legend(title='Model')
plt.grid(True)
plt.tight_layout()
plt.savefig('model_metrics/r2_comparison.png')
plt.show()
mse_scores = metrics_df.pivot_table(values='MSE', index=['Sport', 'Parameter'], columns='Model')
mse_scores.plot(kind='bar', figsize=(50, 15))
plt.title('Model Comparison by MSE')
plt.xlabel('Sport and Parameter')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend(title='Model')
plt.grid(True)
plt.tight_layout()
plt.savefig('model_metrics/mse_comparison.png')
plt.show()
