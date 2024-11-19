
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
import os
import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("model_plots_comparison", exist_ok=True)
os.makedirs("model_metrics", exist_ok=True)
metrics_data = []
def train_and_evaluate_prophet(train_df, test_df, is_first_iteration=False):
    """
    Train and evaluate a Prophet model.
    Handles special case for first iteration by predicting constant value.
    """
    if is_first_iteration:
        prediction = train_df['y'].iloc[0]  # Constant prediction for first iteration
        actual = test_df['y'].values[0]
        return prediction, actual, None
    model = Prophet()
    model.fit(train_df)
    future = model.make_future_dataframe(periods=1, freq='Y')
    forecast = model.predict(future)
    prediction = forecast.iloc[-1]['yhat']
    actual = test_df['y'].values[0]
    return prediction, actual, forecast
for sport, df in sport_data_dict.items():
    for parameter in ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']:
        if parameter not in df.columns:
            continue
        if 'Year' not in df.columns:
            df = df.reset_index()
        parameter_df = df[['Year', parameter]].dropna().rename(columns={'Year': 'ds', parameter: 'y'})
        parameter_df['ds'] = pd.to_datetime(parameter_df['ds'], format='%Y')
        if len(parameter_df) < 3:
            continue  # Skip if not enough data
        predictions = []
        actuals = []
        for i in range(len(parameter_df) - 1):
            train_subset = parameter_df.iloc[:i+1]
            test_point = parameter_df.iloc[i+1:i+2]  # Next data point for validation
            if test_point.empty:
                break
            is_first_iteration = i == 0
            try:
                pred, actual, _ = train_and_evaluate_prophet(train_subset, test_point, is_first_iteration=is_first_iteration)
                predictions.append(pred)
                actuals.append(actual)
            except ValueError as e:
                print(f"Skipping iteration {i+1} for {sport} - {parameter}: {str(e)}")
                continue
        r2 = r2_score(actuals, predictions) if len(predictions) > 1 else None
        mse = mean_squared_error(actuals, predictions) if len(predictions) > 1 else None
        metrics_data.append({
            'Sport': sport,
            'Parameter': parameter,
            'R2': r2,
            'MSE': mse
        })
        predicted_years = parameter_df['ds'][1:len(predictions)+1]  # Use offset for correct plotting
        plt.figure(figsize=(12, 8))
        plt.plot(parameter_df['ds'], parameter_df['y'], label='Actual', color='black', marker='o')
        plt.scatter(predicted_years, predictions, label='Predicted', color='blue', marker='o')
        plt.title(f'{sport} - {parameter} Prediction')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"model_plots_comparison/{sport}_{parameter}_comparison.png")
        plt.close()
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('model_metrics/prophet_metrics.csv', index=False)
print("Training and evaluation completed. Metrics saved.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
metrics_file = 'model_metrics/prophet_metrics.csv'
metrics_df = pd.read_csv(metrics_file)
os.makedirs("model_metrics_heatmaps", exist_ok=True)
r2_pivot = metrics_df.pivot(index="Sport", columns="Parameter", values="R2")
mse_pivot = metrics_df.pivot(index="Sport", columns="Parameter", values="MSE")
plt.figure(figsize=(50,30))
sns.heatmap(r2_pivot, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('R² Score Heatmap')
plt.xlabel('Parameter')
plt.ylabel('Sport')
plt.xticks(rotation=45)
plt.savefig('model_metrics_heatmaps/R2_heatmap.png')
plt.close()
plt.figure(figsize=(50, 30))
sns.heatmap(mse_pivot, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=.5)
plt.title('MSE Score Heatmap')
plt.xlabel('Parameter')
plt.ylabel('Sport')
plt.xticks(rotation=45)
plt.savefig('model_metrics_heatmaps/MSE_heatmap.png')
plt.close()
print("Heatmaps saved to the 'model_metrics_heatmaps' folder.")
