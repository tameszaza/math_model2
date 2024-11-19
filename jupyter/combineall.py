
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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
            id_vars='Sport', var_name='Year', value_name=f'{parameter}'
        )
        melted_data['Year'] = melted_data['Year'].str.extract(r'(\d+)').astype(int)  # Extract year as int
        melted_data = melted_data.sort_values('Year')
        if all_years.empty:
            all_years = melted_data['Year']
        else:
            all_years = pd.concat([all_years, melted_data['Year']]).drop_duplicates().sort_values()
        melted_data = melted_data.set_index('Year').reindex(all_years).reset_index()
        if parameter == 'drug':
            melted_data[f'{parameter}'] = melted_data[f'{parameter}'].fillna(0)  # For drug, fill missing with 0
        elif parameter in ['popularity', 'CV']:
            melted_data[f'{parameter}'] = melted_data[f'{parameter}'].replace(0, np.nan)
        else:
            melted_data[f'{parameter}'] = melted_data[f'{parameter}']
        sport_dict[parameter] = melted_data[f'{parameter}'].values
    sport_dict['Year'] = all_years.values
    sport_data_dict[sport_name] = pd.DataFrame(sport_dict)
sport_data_dict['Athletics']
predictions_data = []
for sport, df in sport_data_dict.items():
    for parameter in ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']:
        if parameter not in df.columns:
            continue
        df.index = pd.PeriodIndex(df['Year'], freq='Y')
        parameter_df = df[[parameter]].dropna()
        if len(parameter_df) < 3:
            continue  # Not enough data to model
        last_year = parameter_df.index[-1].year
        steps_ahead = 2032 - last_year
        if steps_ahead <= 0:
            continue  # Data already includes 2032 or beyond
        train_data = parameter_df[parameter]
        try:
            arima_model = sm.tsa.ARIMA(train_data, order=(1,1,1))
            arima_model_fit = arima_model.fit()
            arima_forecast = arima_model_fit.forecast(steps=steps_ahead)
            arima_pred_2032 = arima_forecast.iloc[-1]
        except Exception as e:
            arima_pred_2032 = np.nan
        try:
            sarima_model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,0,1,4))
            sarima_model_fit = sarima_model.fit()
            sarima_forecast = sarima_model_fit.forecast(steps=steps_ahead)
            sarima_pred_2032 = sarima_forecast.iloc[-1]
        except Exception as e:
            sarima_pred_2032 = np.nan
        try:
            sarimax_model = SARIMAX(train_data, order=(1,1,1))
            sarimax_model_fit = sarimax_model.fit()
            sarimax_forecast = sarimax_model_fit.forecast(steps=steps_ahead)
            sarimax_pred_2032 = sarimax_forecast.iloc[-1]
        except Exception as e:
            sarimax_pred_2032 = np.nan
        predictions_data.append({
            'Sport': sport,
            'Parameter': parameter,
            'Model': 'ARIMA',
            'Prediction_2032': arima_pred_2032
        })
        predictions_data.append({
            'Sport': sport,
            'Parameter': parameter,
            'Model': 'SARIMA',
            'Prediction_2032': sarima_pred_2032
        })
        predictions_data.append({
            'Sport': sport,
            'Parameter': parameter,
            'Model': 'SARIMAX',
            'Prediction_2032': sarimax_pred_2032
        })
predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('predictions_2032.csv', index=False)
print("Predictions for 2032 saved to 'predictions_2032.csv'.")
import pandas as pd
import matplotlib.pyplot as plt
predictions_df = pd.read_csv('predictions_2032.csv')
athletics_df = predictions_df[predictions_df['Sport'] == 'Athletics']
parameters = athletics_df['Parameter'].unique()
for parameter in parameters:
    plt.figure(figsize=(10, 6))
    original_data = sport_data_dict['Athletics'][['Year', parameter]].dropna()
    plt.plot(original_data['Year'], original_data[parameter], label='Historical Data', marker='o')
    pred_2032 = athletics_df[athletics_df['Parameter'] == parameter]
    for model in pred_2032['Model'].unique():
        prediction_value = pred_2032[pred_2032['Model'] == model]['Prediction_2032'].values[0]
        plt.scatter(2032, prediction_value, label=f'Prediction ({model})', s=100)
    plt.title(f'{parameter} Trend for Athletics with 2032 Predictions')
    plt.xlabel('Year')
    plt.ylabel(parameter)
    plt.legend()
    plt.grid(True)
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import os
predictions_df = pd.read_csv('predictions_2032.csv')
output_dir = 'sport_plots'
os.makedirs(output_dir, exist_ok=True)
predictions_df['Plot_Path'] = ''
for sport in predictions_df['Sport'].unique():
    sport_predictions = predictions_df[predictions_df['Sport'] == sport]
    sport_data = sport_data_dict[sport]
    for parameter in sport_predictions['Parameter'].unique():
        plt.figure(figsize=(10, 6))
        historical_data = sport_data[['Year', parameter]].dropna()
        if not historical_data.empty:
            plt.plot(historical_data['Year'], historical_data[parameter], label='Historical Data', marker='o')
        param_predictions = sport_predictions[sport_predictions['Parameter'] == parameter]
        for model in param_predictions['Model'].unique():
            prediction_value = param_predictions[param_predictions['Model'] == model]['Prediction_2032'].values[0]
            plt.scatter(2032, prediction_value, label=f'Prediction ({model})', s=100)
        plt.title(f'{parameter} Trend for {sport} with 2032 Predictions')
        plt.xlabel('Year')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{sport}_{parameter}_trend.png')
        plt.savefig(plot_path)
        plt.close()
        predictions_df.loc[
            (predictions_df['Sport'] == sport) & (predictions_df['Parameter'] == parameter),
            'Plot_Path'
        ] = plot_path
print("All plots saved and DataFrame updated with plot paths.")
predictions_df
