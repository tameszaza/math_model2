
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
print("Is TensorFlow using GPU?", tf.config.list_physical_devices('GPU'))
df = pd.read_csv('filtered.csv')
df.head()
data_dict = {}
for sport in df['Sport_-1'].unique():
    sport_data = df[df['Sport_-1'] == sport]
    sport_dict = {}
    for variable in ['drug', 'equity', 'popularity', 'normalized_country', 'CV']:
        variable_cols = [col for col in df.columns if col.startswith(variable)]
        year_variable_df = sport_data[variable_cols].melt(var_name='Year_Var', value_name=variable)
        year_variable_df['Year'] = year_variable_df['Year_Var'].str.extract('(\d{4})').astype(int)
        year_variable_df = year_variable_df[['Year', variable]].dropna()
        sport_dict[variable] = year_variable_df.reset_index(drop=True)
    data_dict[sport] = sport_dict
df_Athletes_CV = data_dict['Athletics']['CV']
df_Athletes_CV.head()
print("Number of missing values in 'CV':", df_Athletes_CV['CV'].isnull().sum())
df_Athletes_CV['CV'] = df_Athletes_CV['CV'].interpolate(method='linear')
df_Athletes_CV = df_Athletes_CV.sort_values('Year').reset_index(drop=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_Athletes_CV['CV_scaled'] = scaler.fit_transform(df_Athletes_CV['CV'].values.reshape(-1, 1))
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
window_size = 3  # You can adjust this value
data = df_Athletes_CV['CV_scaled'].values
X, y = create_sequences(data, window_size)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_test
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
def plot_training_history(history, metrics=('mse', 'r2_keras')):
    """
    Plot training and validation metrics over epochs.
    Args:
        history: Keras History object after model training.
        metrics: Tuple of metrics to plot. Default is ('mse', 'r2_keras').
    """
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.plot(history.history[metric], label=f'Training {metric.upper()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        plt.title(f'{metric.upper()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.show()
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and compute MSE and R².
    Args:
        model: Trained model to evaluate.
        X_test: Test input data.
        y_test: Test target data.
    Returns:
        test_mse: Mean Squared Error on the test set.
        test_r2: R² score on the test set.
    """
    y_pred = model.predict(X_test)
    if y_test.ndim > 1:
        y_test = y_test.reshape(-1)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {test_mse}")
    print(f"Test R²: {test_r2}")
    return test_mse, test_r2
def plot_predictions(model, data, window_size, scaler, years, target_column='CV_scaled', title='Actual vs Predicted CV Values Over Years'):
    """
    Plot actual vs predicted values for a given model.
    Args:
        model: Trained model for prediction.
        data: DataFrame or array containing the input data.
        window_size: Size of the sequence window used during training.
        scaler: Scaler object used for scaling the data.
        years: Array of years corresponding to the data.
        target_column: Column name of the scaled target variable in `data`.
        title: Title for the plot.
    """
    data_values = data[target_column].values
    X_all, y_all = create_sequences(data_values, window_size)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    y_pred_scaled = model.predict(X_all)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_all.reshape(-1, 1))
    adjusted_years = years[window_size:]
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_years, y_actual, label='Actual', marker='o')
    plt.plot(adjusted_years, y_pred, label='Predicted', marker='x')
    plt.xlabel('Year')
    plt.ylabel('CV')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
df = data_dict['Athletics']['CV']
df.head()
time_series_data = df['CV']
get_ipython().system('pip install pmdarima')
import pmdarima as pm
model = pm.auto_arima(
    time_series_data,
    start_p=0, max_p=5,      # Range for p
    start_q=0, max_q=5,      # Range for q
    d=None,                  # Automatically determine d
    start_d=0, max_d=2,      # Differencing range
    seasonal=False,          # Use seasonal=True for SARIMA
    trace=True,              # Print search progress
    error_action='ignore',   # Ignore errors during search
    suppress_warnings=True,  # Suppress warnings
    stepwise=True            # Use stepwise algorithm for efficiency
)
print(model.summary())
from statsmodels.tsa.arima.model import ARIMA
best_order = model.order  # (p, d, q)
arima_model = ARIMA(time_series_data, order=best_order).fit()
print(arima_model.summary())
model_arima = ARIMA(df['CV'], order=(1,0,0))
model_arima_fit = model_arima.fit()
df['CV_arima_pred'] = model_arima_fit.predict(start=df.index[1], end=df.index[-1])
df['CV_residual'] = df['CV'] - df['CV_arima_pred']
from sklearn.preprocessing import MinMaxScaler
df.reset_index(inplace=True)
df['CV_residual'].fillna(method='bfill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
df['CV_residual_scaled'] = scaler.fit_transform(df['CV_residual'].values.reshape(-1,1))
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
window_size = 3  # Adjust as needed
data = df['CV_residual_scaled'].values
X, y = create_sequences(data, window_size)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
X_train = X_train.reshape((X_train.shape[0], window_size, 1))
X_test = X_test.reshape((X_test.shape[0], window_size, 1))
optimizer5  = Adam(learning_rate=0.0001)
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu', input_shape=(window_size, 1)))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer=optimizer5, loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history5 = model_lstm.fit(
    X_train, y_train,
    epochs=500,
    batch_size=1,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)
y_pred_residual_scaled = model_lstm.predict(X_test)
y_pred_residual = scaler.inverse_transform(y_pred_residual_scaled)
test_indices = df.index[window_size + split_index:]
arima_pred_test = df.loc[test_indices, 'CV_arima_pred'].values
final_pred = arima_pred_test + y_pred_residual.flatten()
from sklearn.metrics import mean_squared_error, r2_score
y_actual = df.loc[test_indices, 'CV'].values
mse = mean_squared_error(y_actual, final_pred)
r2 = r2_score(y_actual, final_pred)
print(f"Hybrid Model Test MSE: {mse}")
print(f"Hybrid Model Test R²: {r2}")
import matplotlib.pyplot as plt
test_years = df.loc[test_indices, 'Year'].values
plt.figure(figsize=(12,6))
plt.plot(test_years, y_actual, label='Actual CV', marker='o')
plt.plot(test_years, arima_pred_test, label='ARIMA Predictions', marker='x')
plt.plot(test_years, final_pred, label='Hybrid ARIMA-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs ARIMA and Hybrid ARIMA-LSTM Predictions')
plt.legend()
plt.grid(True)
plt.show()
arima_all_pred = model_arima_fit.predict(start=df.index[0], end=df.index[-1])
df['CV_residual'] = df['CV'] - arima_all_pred
df['CV_residual'].fillna(method='bfill', inplace=True)
df['CV_residual_scaled'] = scaler.transform(df['CV_residual'].values.reshape(-1, 1))
data_all = df['CV_residual_scaled'].values
X_all, _ = create_sequences(data_all, window_size)
X_all = X_all.reshape((X_all.shape[0], window_size, 1))
lstm_all_residual_scaled = model_lstm.predict(X_all)
lstm_all_residual = scaler.inverse_transform(lstm_all_residual_scaled)
hybrid_all_pred = arima_all_pred[window_size:] + lstm_all_residual.flatten()
actual_values = df['CV'].values[window_size:]
import matplotlib.pyplot as plt
years = df['Year'].values[window_size:]
plt.figure(figsize=(12, 6))
plt.plot(years, actual_values, label='Actual CV', marker='o')
plt.plot(years, hybrid_all_pred, label='Hybrid ARIMA-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs Hybrid ARIMA-LSTM Predictions (All Years)')
plt.legend()
plt.grid(True)
plt.show()
r2 = r2_score(actual_values, hybrid_all_pred)
print("R² value:", r2)
