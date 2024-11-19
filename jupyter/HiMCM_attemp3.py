
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
df = pd.read_csv('/content/drive/MyDrive/filtered.csv')
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
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score, mean_squared_error
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - SS_res / (SS_tot + K.epsilon())
    return r2
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', r2_keras])
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=15,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the weights from the epoch with the best value of the monitored quantity
)
history = model.fit(
    X_train, y_train,
    epochs=500,                # Maximum number of epochs
    batch_size=1,              # You can adjust this based on your dataset size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1                  # Set to 1 to print progress for each epoch
)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Mean Squared Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['r2_keras'], label='Training R²')
plt.plot(history.history['val_r2_keras'], label='Validation R²')
plt.title('R² Score Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.legend()
plt.tight_layout()
plt.show()
y_pred = model.predict(X_test)
if y_test.ndim > 1:
    y_test = y_test.reshape(-1)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")
window_size = 3
data = df_Athletes_CV['CV_scaled'].values
X_all, y_all = create_sequences(data, window_size)
X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
y_pred_scaled = model.predict(X_all)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_all.reshape(-1, 1))
years = df_Athletes_CV['Year'].values
years = years[window_size:]
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(years, y_actual, label='Actual CV', marker='o')
plt.plot(years, y_pred, label='Predicted CV', marker='x')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs Predicted CV Values Over Years')
plt.legend()
plt.grid(True)
plt.show()
model2 = Sequential()
model2.add(LSTM(units=128, activation='relu', input_shape=(X_train.shape[1], 1)))
model2.add(Dense(units=1))
optimizer2 = Adam(learning_rate=0.0001)
model2.compile(optimizer=optimizer2, loss='mean_squared_error', metrics=['mse', r2_keras])
early_stopping2 = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=15,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the weights from the epoch with the best value of the monitored quantity
)
history2 = model2.fit(
    X_train, y_train,
    epochs=500,                # Maximum number of epochs
    batch_size=16,              # You can adjust this based on your dataset size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping2],
    verbose=1                  # Set to 1 to print progress for each epoch
)
plot_training_history(history2)
plot_predictions(model=model2, data=df_Athletes_CV, window_size=3, scaler=scaler, years=df_Athletes_CV['Year'].values)
y_pred = model2.predict(X_test)
if y_test.ndim > 1:
    y_test = y_test.reshape(-1)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
model3 = Sequential()
model3.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())
model3.add(Bidirectional(LSTM(units=64)))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())
model3.add(Dense(units=1))
optimizer3  = Adam(learning_rate=0.00005)  # Reduced learning rate for finer updates
model3.compile(optimizer=optimizer3, loss='mean_squared_error', metrics=['mse', r2_keras])
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)
history3 = model3.fit(
    X_train, y_train,
    epochs=500,  # Increased epochs due to the complexity of the model
    batch_size=2,  # Adjusted batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)
plot_training_history(history3)
plot_predictions(model=model3, data=df_Athletes_CV, window_size=3, scaler=scaler, years=df_Athletes_CV['Year'].values)
y_pred = model3.predict(X_test)
if y_test.ndim > 1:
    y_test = y_test.reshape(-1)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")
model4 = Sequential()
model4.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model4.add(Dropout(0.1))
model4.add(BatchNormalization())
model4.add(Bidirectional(LSTM(units=128)))
model4.add(Dropout(0.1))
model4.add(BatchNormalization())
model4.add(Dense(units=1))
optimizer4  = Adam(learning_rate=0.00005)  # Reduced learning rate for finer updates
model4.compile(optimizer=optimizer4, loss='mean_squared_error', metrics=['mse', r2_keras])
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)
history4 = model4.fit(
    X_train, y_train,
    epochs=700,  # Increased epochs due to the complexity of the model
    batch_size=4,  # Adjusted batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)
plot_training_history(history4)
plot_predictions(model=model4, data=df_Athletes_CV, window_size=3, scaler=scaler, years=df_Athletes_CV['Year'].values)
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
df = data_dict['Athletics']['CV']
ts = df['CV']
df.head()
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Time Series Plot')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: {:.6f}'.format(result[0]))
    print('p-value: {:.6f}'.format(result[1]))
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
adf_test(ts)
ts_diff = ts.diff().dropna()
adf_test(ts_diff)
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(ts_diff, ax=axes[0], lags=10)
axes[0].set_title('ACF Plot')
plot_pacf(ts_diff, ax=axes[1], lags=10, method='ywm')
axes[1].set_title('PACF Plot')
plt.show()
import warnings
warnings.filterwarnings("ignore")  # To ignore convergence warnings
import itertools
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
aic_values = []
parameters = []
for param in pdq:
    try:
        model = ARIMA(ts, order=param)
        results = model.fit()
        aic_values.append(results.aic)
        parameters.append(param)
    except:
        continue
lowest_aic = min(aic_values)
best_params = parameters[aic_values.index(lowest_aic)]
print(f'Best ARIMA order: {best_params} with AIC: {lowest_aic}')
get_ipython().system('pip install pmdarima')
import pmdarima as pm
model_auto = pm.auto_arima(ts, start_p=0, start_q=0,
                           max_p=5, max_q=5, d=None,
                           seasonal=False, stepwise=True,
                           suppress_warnings=True, information_criterion='aic')
print(model_auto.summary())
p, d, q = model_auto.order
arima_model = ARIMA(ts, order=(p, d, q)).fit()
arima_pred = arima_model.predict(start=ts.index[0], end=ts.index[-1], typ='levels')
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Actual')
plt.plot(arima_pred, label='ARIMA Prediction')
plt.legend()
plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: {:.6f}'.format(result[0]))
    print('p-value: {:.6f}'.format(result[1]))
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
adf_test(ts)
ts_diff = ts.diff().dropna()
adf_test(ts_diff)
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(ts, model='additive', period=4)
decompose_result.plot()
plt.show()
max_lag = int(len(ts) / 2) - 1  # Ensure lags < 50% of sample size
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts_diff, ax=axes[0], lags=max_lag)
axes[0].set_title('ACF Plot')
plot_pacf(ts_diff, ax=axes[1], lags=max_lag, method='ywm')
axes[1].set_title('PACF Plot')
plt.show()
model_auto = pm.auto_arima(ts, start_p=0, start_q=0,
                           max_p=3, max_q=3, d=None,
                           start_P=0, start_Q=0, max_P=2, max_Q=2,
                           m=4,  # Seasonal period
                           seasonal=True, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True, information_criterion='aic')
print(model_auto.summary())
order = model_auto.order
seasonal_order = model_auto.seasonal_order
sarimax_model = SARIMAX(ts, order=order, seasonal_order=seasonal_order).fit(disp=False)
sarimax_pred = sarimax_model.predict(start=ts.index[0], end=ts.index[-1])
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Actual')
plt.plot(sarimax_pred, label='SARIMAX Prediction')
plt.legend()
plt.title('SARIMAX Model Prediction')
plt.show()
from keras.layers import GRU, Dense, Dropout
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Time Series Plot')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()
train_size = int(len(ts) * 0.8)  # Use 80% for training
test_size = len(ts) - train_size
train_ts = ts.iloc[:train_size]
test_ts = ts.iloc[train_size:]
print(f'Training data points: {len(train_ts)}')
print(f'Testing data points: {len(test_ts)}')
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_ts.values.reshape(-1, 1))
test_scaled = scaler.transform(test_ts.values.reshape(-1, 1))
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
seq_length = 1  # Using a sequence length of 1 due to limited data
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)
X_train = X_train.reshape((X_train.shape[0], seq_length, 1))
X_test = X_test.reshape((X_test.shape[0], seq_length, 1))
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
from keras.regularizers import l2
model_gru = Sequential()
model_gru.add(GRU(units=64, activation='relu', input_shape=(seq_length, 1),
              kernel_regularizer=l2(0.0005)))
model_gru.add(Dropout(0.2))  # Dropout to prevent overfitting
model_gru.add(Dense(units=1))
optimizer_gru = Adam(learning_rate=0.0001)
model_gru.compile(optimizer=optimizer_gru , loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history_gru = model_gru.fit(X_train, y_train, epochs=500, batch_size=1,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=0)
plt.figure(figsize=(10, 4))
plt.plot(history_gru.history['loss'], label='Training Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.title('GRU Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
train_pred_scaled = model_gru.predict(X_train)
train_pred = scaler.inverse_transform(train_pred_scaled)
y_train_actual = scaler.inverse_transform(y_train)
test_pred_scaled = model_gru.predict(X_test)
test_pred = scaler.inverse_transform(test_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
print(f'Training RMSE: {train_rmse:.4f}')
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
print(f'Testing RMSE: {test_rmse:.4f}')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_ts.index[seq_length:], y_train_actual, label='Actual', marker='o')
plt.plot(train_ts.index[seq_length:], train_pred, label='Predicted', marker='o')
plt.title('Training Data')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(test_ts.index[seq_length:], y_test_actual, label='Actual', marker='o')
plt.plot(test_ts.index[seq_length:], test_pred, label='Predicted', marker='o')
plt.title('Testing Data')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pmdarima as pm
import matplotlib.pyplot as plt
df = data_dict['Athletics']['CV']
df.head()
time_series_data = df['CV']
model_grid = pm.auto_arima(
    time_series_data,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,
    start_d=0, max_d=2,
    seasonal=True,
    m=1,  # Set to appropriate seasonal period
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(model_grid.summary())
best_order = model_grid.order
best_seasonal_order = model_grid.seasonal_order
print(f"Best order: {best_order}")
print(f"Best seasonal order: {best_seasonal_order}")
sarimax2_model = SARIMAX(time_series_data, order=best_order, seasonal_order=best_seasonal_order).fit()
print(sarimax2_model.summary())
df['CV_sarimax_pred'] = sarimax2_model.predict(start=df.index[0], end=df.index[-1])
df['CV_residual'] = df['CV'] - df['CV_sarimax_pred']
df['CV_residual'].fillna(method='bfill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
df['CV_residual_scaled'] = scaler.fit_transform(df['CV_residual'].values.reshape(-1, 1))
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
optimizer = Adam(learning_rate=0.0001)
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu', input_shape=(window_size, 1)))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer=optimizer, loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model_lstm.fit(
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
sarimax_pred_test = df.loc[test_indices, 'CV_sarimax_pred'].values
final_pred = sarimax_pred_test + y_pred_residual.flatten()
y_actual = df.loc[test_indices, 'CV'].values
mse = mean_squared_error(y_actual, final_pred)
r2 = r2_score(y_actual, final_pred)
print(f"Hybrid Model Test MSE: {mse}")
print(f"Hybrid Model Test R²: {r2}")
test_years = df.loc[test_indices, 'Year'].values
plt.figure(figsize=(12, 6))
plt.plot(test_years, y_actual, label='Actual CV', marker='o')
plt.plot(test_years, sarimax_pred_test, label='SARIMAX Predictions', marker='x')
plt.plot(test_years, final_pred, label='Hybrid SARIMAX-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs SARIMAX and Hybrid SARIMAX-LSTM Predictions')
plt.legend()
plt.grid(True)
plt.show()
sarimax_all_pred = sarimax2_model.predict(start=df.index[0], end=df.index[-1])
df['CV_residual'] = df['CV'] - sarimax_all_pred
df['CV_residual'].fillna(method='bfill', inplace=True)
df['CV_residual_scaled'] = scaler.transform(df['CV_residual'].values.reshape(-1, 1))
data_all = df['CV_residual_scaled'].values
X_all, _ = create_sequences(data_all, window_size)
X_all = X_all.reshape((X_all.shape[0], window_size, 1))
lstm_all_residual_scaled = model_lstm.predict(X_all)
lstm_all_residual = scaler.inverse_transform(lstm_all_residual_scaled)
hybrid_all_pred = sarimax_all_pred[window_size:] + lstm_all_residual.flatten()
actual_values = df['CV'].values[window_size:]
years = df['Year'].values[window_size:]
plt.figure(figsize=(12, 6))
plt.plot(years, actual_values, label='Actual CV', marker='o')
plt.plot(years, hybrid_all_pred, label='Hybrid SARIMAX-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs Hybrid SARIMAX-LSTM Predictions (All Years)')
plt.legend()
plt.grid(True)
plt.show()
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
model_dict = {}
predictions_dict = {}
window_size = 3
lstm_units = 50
epochs = 500
batch_size = 1
learning_rate = 0.0001
future_steps = 16  # Number of future predictions
year_interval = 4  # Interval for future years
for sport, variables in data_dict.items():
    model_dict[sport] = {}
    predictions_dict[sport] = {}
    for variable, df in variables.items():
        print(f"Training models for Sport: {sport}, Variable: {variable}")
        df = df.sort_values(by='Year').reset_index(drop=True)  # Ensure chronological order
        time_series_data = df[variable].values
        if np.all(np.isnan(time_series_data)):
            print(f"All values are NaN for {sport} - {variable}. Skipping.")
            continue
        filled_data = df[variable].fillna(method='bfill')  # Backfill for ARIMA
        df['Residual'] = filled_data
        try:
            arima_model = pm.auto_arima(
                filled_data,
                start_p=0, max_p=5,
                start_q=0, max_q=5,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            arima_order = arima_model.order
            arima_fit = ARIMA(filled_data, order=arima_order).fit()
            df['ARIMA_Pred'] = arima_fit.predict(start=df.index[0], end=df.index[-1])
        except Exception as e:
            print(f"Error training ARIMA for {sport} - {variable}: {e}")
            continue
        df['Residual'] = df[variable] - df['ARIMA_Pred']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['Residual_Scaled'] = scaler.fit_transform(
            df['Residual'].fillna(0).values.reshape(-1, 1)  # Handle NaN residuals for LSTM
        )
        data = df['Residual_Scaled'].values
        X, y = create_sequences(data, window_size)
        if len(X) == 0:  # Skip if insufficient data for LSTM
            print(f"Insufficient data for LSTM for {sport} - {variable}. Skipping.")
            continue
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train = X_train.reshape((X_train.shape[0], window_size, 1))
        X_test = X_test.reshape((X_test.shape[0], window_size, 1))
        model_lstm = Sequential()
        model_lstm.add(LSTM(units=lstm_units, activation='relu', input_shape=(window_size, 1)))
        model_lstm.add(Dense(units=1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
        X_all = create_sequences(data, window_size)[0]
        X_all = X_all.reshape((X_all.shape[0], window_size, 1))
        lstm_residual_pred_scaled = model_lstm.predict(X_all)
        lstm_residual_pred = scaler.inverse_transform(lstm_residual_pred_scaled)
        hybrid_pred = df['ARIMA_Pred'].iloc[window_size:].values + lstm_residual_pred.flatten()
        future_arima_pred = arima_fit.forecast(steps=future_steps)
        future_residuals_input = np.zeros((future_steps, window_size, 1))
        if len(data) >= window_size:
            future_residuals_input[0] = data[-window_size:].reshape(-1, 1)
        future_residual_scaled = []
        for i in range(future_steps):
            next_residual = model_lstm.predict(future_residuals_input[i:i + 1])
            future_residual_scaled.append(next_residual[0][0])
            if i < future_steps - 1:
                future_residuals_input[i + 1, 1:] = future_residuals_input[i, :-1]
                future_residuals_input[i + 1, -1] = next_residual
        future_residuals = scaler.inverse_transform(np.array(future_residual_scaled).reshape(-1, 1))
        future_hybrid_pred = future_arima_pred + future_residuals.flatten()
        future_years = np.arange(df['Year'].iloc[-1] + year_interval,
                                 df['Year'].iloc[-1] + year_interval * (future_steps + 1),
                                 year_interval)
        model_dict[sport][variable] = {
            'ARIMA': arima_fit,
            'LSTM': model_lstm
        }
        predictions_dict[sport][variable] = {
            'Actual': df[variable].iloc[window_size:].values,
            'Hybrid': hybrid_pred,
            'Future Hybrid': future_hybrid_pred,
            'ARIMA': df['ARIMA_Pred'].iloc[window_size:].values
        }
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], df[variable], label='Actual', marker='o')
        plt.plot(df['Year'].iloc[window_size:], hybrid_pred, label='Hybrid Prediction (Train)', marker='^')
        plt.plot(future_years, future_hybrid_pred, label='Hybrid Prediction (Future)', marker='s')
        plt.plot(future_years, future_arima_pred, label='ARIMA Prediction (Future)', linestyle='--', marker='x')
        plt.title(f'{sport} - {variable} (Future Predictions)')
        plt.xlabel('Year')
        plt.ylabel(variable)
        plt.legend()
        plt.grid()
        plt.show()
print("Model training and future prediction visualization complete!")
print(df.columns)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
def build_multi_head_lstm(input_shapes, lstm_units=50, dense_units=64):
    inputs = []
    lstm_outputs = []
    for shape in input_shapes:
        inp = Input(shape=(shape[0], 1))  # Input shape (window_size, 1)
        lstm_out = LSTM(units=lstm_units, activation='relu')(inp)
        inputs.append(inp)
        lstm_outputs.append(lstm_out)
    concatenated = Concatenate()(lstm_outputs)
    dense_out = Dense(dense_units, activation='relu')(concatenated)
    output = Dense(1)(dense_out)  # Final single output
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
window_size = 3
lstm_units = 50
dense_units = 64
epochs = 500
batch_size = 32
future_steps = 5  # Number of future points to predict
model_dict = {}
predictions_dict = {}
for sport, variables in data_dict.items():
    print(f"Training Multi-head LSTM for sport: {sport}")
    X_data = []
    y_data = None
    variable_names = []
    target_variable = 'CV'  # Specify the target variable
    for variable, df in variables.items():
        print(f"Processing variable: {variable}")
        df = df.sort_values(by='Year').reset_index(drop=True)  # Ensure chronological order
        time_series = df[variable].fillna(method='bfill').values  # Handle missing values
        X, y = create_sequences(time_series, window_size)
        X_data.append(X)
        variable_names.append(variable)
        if variable == target_variable:
            y_data = y  # Set target variable
    if y_data is None:
        raise ValueError(f"Target variable '{target_variable}' not found in variables: {variable_names}")
    min_len = min(len(x) for x in X_data)
    X_data = [x[:min_len] for x in X_data]
    y_data = y_data[:min_len]
    X_data = [x.reshape((x.shape[0], x.shape[1], 1)) for x in X_data]
    input_shapes = [(window_size,) for _ in X_data]
    model = build_multi_head_lstm(input_shapes, lstm_units=lstm_units, dense_units=dense_units)
    split_idx = int(len(y_data) * 0.8)
    X_train = [x[:split_idx] for x in X_data]
    y_train = y_data[:split_idx]
    X_test = [x[split_idx:] for x in X_data]
    y_test = y_data[split_idx:]
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    y_pred = model.predict(X_test)
    model_dict[sport] = model
    predictions_dict[sport] = {
        'Actual': y_test,
        'Predicted': y_pred.flatten(),
        'Variable Names': variable_names
    }
    backtest_predictions = []
    for i in range(window_size, len(y_data)):
        input_sequence = [x[i - window_size:i] for x in X_data]
        input_sequence = [seq.reshape((1, window_size, 1)) for seq in input_sequence]
        pred = model.predict(input_sequence)
        backtest_predictions.append(pred[0][0])
    future_predictions = []
    last_sequences = [x[-window_size:] for x in X_data]  # Get the last sequences for each variable
    for _ in range(future_steps):
        input_sequence = [seq.reshape((1, window_size, 1)) for seq in last_sequences]
        future_pred = model.predict(input_sequence)
        future_predictions.append(future_pred[0][0])
        for j in range(len(last_sequences)):
            last_sequences[j] = np.roll(last_sequences[j], -1, axis=0)
            if variable_names[j] == target_variable:
                last_sequences[j][-1] = future_pred[0][0]
            else:
                last_sequences[j][-1] = last_sequences[j][-2]
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual (Test)', marker='o')
    plt.plot(range(len(y_pred)), y_pred.flatten(), label='Predicted (Test)', marker='x')
    plt.plot(range(window_size, len(backtest_predictions) + window_size), backtest_predictions, label='Backtesting', linestyle='--')
    plt.plot(range(len(y_test), len(y_test) + future_steps), future_predictions, label='Future Predictions', marker='s')
    plt.title(f"Multi-head LSTM Predictions for {sport}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
print("Training, backtesting, and future prediction visualization complete!")
get_ipython().system('pip install keras-tuner -q')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
import pmdarima as pm
import matplotlib.pyplot as plt
import keras_tuner as kt
df = data_dict['Athletics']['CV']
time_series_data = df['CV']
model_grid = pm.auto_arima(
    time_series_data,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,
    start_d=0, max_d=2,
    seasonal=True,
    m=1,  # Set to appropriate seasonal period
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
best_order = model_grid.order
best_seasonal_order = model_grid.seasonal_order
sarimax2_model = SARIMAX(time_series_data, order=best_order, seasonal_order=best_seasonal_order).fit()
df['CV_sarimax_pred'] = sarimax2_model.predict(start=df.index[0], end=df.index[-1])
df['CV_residual'] = df['CV'] - df['CV_sarimax_pred']
df['CV_residual'].fillna(method='bfill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
df['CV_residual_scaled'] = scaler.fit_transform(df['CV_residual'].values.reshape(-1, 1))
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
def prepare_data(window_size, return_X_y=False):
    data = df['CV_residual_scaled'].values
    X, y = create_sequences(data, window_size)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_val = X[split_index:]
    y_val = y[split_index:]
    X_train = X_train.reshape((X_train.shape[0], window_size, 1))
    X_val = X_val.reshape((X_val.shape[0], window_size, 1))
    if return_X_y:
        return X, y, X_train, y_train, X_val, y_val
    else:
        return X_train, y_train, X_val, y_val
def build_model(hp):
    window_size = hp.Int('window_size', min_value=2, max_value=10, step=1)
    X_train, y_train, X_val, y_val = prepare_data(window_size)
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=16, max_value=128, step=16),
        activation=hp.Choice('activation', values=['relu', 'tanh']),
        input_shape=(window_size, 1)
    ))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=1))
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG'))
    else:
        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='lstm_sarimax_hybrid'
)
class CustomTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        window_size = hp.get('window_size')
        X_train, y_train, X_val, y_val = prepare_data(window_size)
        model = self.hypermodel.build(hp)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=hp.Int('batch_size', min_value=16, max_value=128, step=16),
            callbacks=[early_stopping],
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        self.oracle.update_trial(trial.trial_id, {'val_loss': val_loss})
tuner = CustomTuner(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='lstm_sarimax_hybrid'
)
tuner.search()
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
window_size = best_hps.get('window_size')
units = best_hps.get('units')
activation = best_hps.get('activation')
dropout = best_hps.get('dropout')
optimizer_choice = best_hps.get('optimizer')
learning_rate = best_hps.get('learning_rate')
batch_size = best_hps.get('batch_size')
print(f"""
The hyperparameter search is complete. The optimal hyperparameters are:
- Window Size: {window_size}
- Units: {units}
- Activation: {activation}
- Dropout Rate: {dropout}
- Optimizer: {optimizer_choice}
- Learning Rate: {learning_rate}
- Batch Size: {batch_size}
""")
X, y, X_train, y_train, X_test, y_test = prepare_data(window_size, return_X_y=True)
model_lstm = tuner.hypermodel.build(best_hps)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_lstm.fit(
    X_train, y_train,
    epochs=100,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)
y_pred_residual_scaled = model_lstm.predict(X_test)
y_pred_residual = scaler.inverse_transform(y_pred_residual_scaled)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
test_start_index = window_size + split_index
test_end_index = window_size + len(X)
test_indices = df.index[test_start_index : test_end_index]
sarimax_pred_test = df.loc[test_indices, 'CV_sarimax_pred'].values
test_indices = df.index[test_start_index : test_end_index]
sarimax_pred_test = df.loc[test_indices, 'CV_sarimax_pred'].values
final_pred = sarimax_pred_test + y_pred_residual.flatten()
y_actual = df.loc[test_indices, 'CV'].values
mse = mean_squared_error(y_actual, final_pred)
r2 = r2_score(y_actual, final_pred)
print(f"Hybrid Model Test MSE: {mse}")
print(f"Hybrid Model Test R²: {r2}")
test_years = df.loc[test_indices, 'Year'].values
plt.figure(figsize=(12, 6))
plt.plot(test_years, y_actual, label='Actual CV', marker='o')
plt.plot(test_years, sarimax_pred_test, label='SARIMAX Predictions', marker='x')
plt.plot(test_years, final_pred, label='Hybrid SARIMAX-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs SARIMAX and Hybrid SARIMAX-LSTM Predictions')
plt.legend()
plt.grid(True)
plt.show()
sarimax_all_pred = sarimax2_model.predict(start=df.index[0], end=df.index[-1])
df['CV_residual'] = df['CV'] - sarimax_all_pred
df['CV_residual'].fillna(method='bfill', inplace=True)
df['CV_residual_scaled'] = scaler.transform(df['CV_residual'].values.reshape(-1, 1))
data_all = df['CV_residual_scaled'].values
X_all, _ = create_sequences(data_all, window_size)
X_all = X_all.reshape((X_all.shape[0], window_size, 1))
lstm_all_residual_scaled = model_lstm.predict(X_all)
lstm_all_residual = scaler.inverse_transform(lstm_all_residual_scaled)
hybrid_all_pred = sarimax_all_pred[window_size:] + lstm_all_residual.flatten()
actual_values = df['CV'].values[window_size:]
years = df['Year'].values[window_size:]
plt.figure(figsize=(12, 6))
plt.plot(years, actual_values, label='Actual CV', marker='o')
plt.plot(years, hybrid_all_pred, label='Hybrid SARIMAX-LSTM Predictions', marker='^')
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual vs Hybrid SARIMAX-LSTM Predictions (All Years)')
plt.legend()
plt.grid(True)
plt.show()
df_CV=data_dict['Athletics']['CV']
df_popularity = data_dict['Athletics']['popularity']
df_drug = data_dict['Athletics']['drug']
df_normalized_country = data_dict['Athletics']['normalized_country']
df_total = pd.concat([df_CV,df_popularity,df_drug,df_normalized_country], axis=1)
df_total = df_total.loc[:, ~df_total.T.duplicated()]
df_total.head()
print("Column names:", df_total.columns.tolist())
duplicates = df_total.columns.duplicated()
if duplicates.any():
    print("Duplicate columns found:")
    print(df_total.columns[duplicates])
else:
    print("No duplicate columns found.")
df_total = df_total.loc[:, ~df_total.columns.duplicated()]
df_total.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    print('---')
for column in df_total.columns:
    adf_test(df_total[column], title=column)
df_total['drug_diff'] = df_total['drug'].diff()
df_total['normalized_country_diff'] = df_total['normalized_country'].diff()
df_total.drop(['drug', 'normalized_country'], axis=1, inplace=True)
df_total = df_total.dropna()
adf_test(df_total['drug_diff'], title='drug_diff')
adf_test(df_total['normalized_country_diff'], title='normalized_country_diff')
df_stationary = df_total[['CV', 'popularity', 'drug_diff', 'normalized_country_diff']]
data = df_stationary.copy()
from statsmodels.tsa.api import VAR
model = VAR(data)
maxlags = 1
lag_order_results = model.select_order(maxlags=maxlags)
print(lag_order_results.summary())
optimal_lag = lag_order_results.aic
if optimal_lag is None:
    optimal_lag = 1  # Default to 1 if AIC is not available
var_model = model.fit(optimal_lag)
print(var_model.summary())
forecast_steps = 1  # Adjust as needed
forecast = var_model.forecast(y=data.values[-optimal_lag:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=[data.index[-1] + 1], columns=data.columns)
print(forecast_df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
last_observations = df_stationary.values[-optimal_lag:]
forecast_steps = 10  # Number of future time points to forecast
forecast = var_model.forecast(y=last_observations, steps=forecast_steps)
last_index = df_stationary.index[-1]
forecast_index = [last_index + i for i in range(1, forecast_steps + 1)]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df_stationary.columns)
differenced_vars = ['drug_diff', 'normalized_country_diff']
non_differenced_vars = ['CV', 'popularity']
last_actual_values = df_total.iloc[-1]
forecast_df_inverted = forecast_df.copy()
for var in differenced_vars:
    forecast_df_inverted[var] = last_actual_values[var] + forecast_df[var].cumsum()
for var in non_differenced_vars:
    forecast_df_inverted[var] = forecast_df[var]
df_total_reset = df_total.reset_index(drop=True)
forecast_df_inverted_reset = forecast_df_inverted.reset_index(drop=True)
df_combined = pd.concat([df_total_reset, forecast_df_inverted_reset], ignore_index=True)
plt.figure(figsize=(12,6))
plt.plot(df_combined.index[:len(df_total)], df_combined['CV'].iloc[:len(df_total)], label='Actual CV')
plt.plot(df_combined.index[len(df_total)-1:], df_combined['CV'].iloc[len(df_total)-1:], label='Forecasted CV', linestyle='--', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('CV')
plt.title('Actual and Forecasted CV')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_total_reset = df_total.reset_index()
df_total_reset.set_index('Year', inplace=True)
df_stationary_reset = df_stationary.copy()
df_stationary_reset['Year'] = df_total_reset.index
df_stationary_reset.set_index('Year', inplace=True)
last_observations = df_stationary_reset.values[-optimal_lag:]
forecast_steps = 10  # Number of future time points to forecast
forecast = var_model.forecast(y=last_observations, steps=forecast_steps)
last_year = df_total_reset.index[-1]
future_years = [last_year + 4 * (i + 1) for i in range(forecast_steps)]
forecast_df = pd.DataFrame(forecast, index=future_years, columns=df_stationary_reset.columns)
differenced_vars = ['drug_diff', 'normalized_country_diff']
non_differenced_vars = ['CV', 'popularity']
last_actual_values = df_total_reset.iloc[-1]
forecast_df_inverted = forecast_df.copy()
for var in differenced_vars:
    forecast_df_inverted[var] = last_actual_values[var] + forecast_df[var].cumsum()
for var in non_differenced_vars:
    forecast_df_inverted[var] = forecast_df[var]
df_combined = pd.concat([df_total_reset, forecast_df_inverted], axis=0)
plt.figure(figsize=(12,6))
plt.plot(df_combined.index[:len(df_total_reset)], df_combined['CV'].iloc[:len(df_total_reset)], label='Actual CV')
plt.plot(df_combined.index[len(df_total_reset)-1:], df_combined['CV'].iloc[len(df_total_reset)-1:], label='Forecasted CV', linestyle='--', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual and Forecasted CV')
plt.show()
df_total_reset = df_total.reset_index()
df_total_reset = df_total_reset.sort_values('Year')
df_total_reset.set_index('Year', inplace=True)
df_total_reset.index = df_total_reset.index.astype(int)
df_stationary_reset = df_stationary.copy()
df_stationary_reset['Year'] = df_total_reset.index
df_stationary_reset.set_index('Year', inplace=True)
fitted_values = var_model.fittedvalues
fitted_values.index = df_stationary_reset.index[optimal_lag:]
df_fitted = pd.DataFrame(fitted_values, index=fitted_values.index, columns=df_stationary_reset.columns)
last_year_in_data = df_total_reset.index[-1]
years_to_forecast = list(range(last_year_in_data + 4, 2025, 4))
forecast_steps = len(years_to_forecast)
last_observations = df_stationary_reset.values[-optimal_lag:]
forecast = var_model.forecast(y=last_observations, steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=years_to_forecast, columns=df_stationary_reset.columns)
differenced_vars = ['drug_diff', 'normalized_country_diff']  # Adjust based on your data
non_differenced_vars = ['CV', 'popularity']
last_actual_values = df_total_reset.iloc[-1]
forecast_df_inverted = forecast_df.copy()
for var in differenced_vars:
    last_value = last_actual_values[var]
    forecast_df_inverted[var] = forecast_df[var].cumsum() + last_value
for var in non_differenced_vars:
    forecast_df_inverted[var] = forecast_df[var]
fitted_values_inverted = df_fitted.copy()
for var in differenced_vars:
    actual_values = df_total_reset[var].loc[fitted_values_inverted.index]
    fitted_values_inverted[var] = actual_values.shift(1) + fitted_values_inverted[var]
for var in non_differenced_vars:
    fitted_values_inverted[var] = df_fitted[var]
df_combined = pd.concat([
    df_total_reset[[*non_differenced_vars, *differenced_vars]].iloc[:optimal_lag],
    fitted_values_inverted,
    forecast_df_inverted
], axis=0)
plt.figure(figsize=(14,7))
plt.plot(df_total_reset.index, df_total_reset['CV'], label='Actual CV', marker='o')
plt.plot(fitted_values_inverted.index, fitted_values_inverted['CV'], label='Fitted CV', linestyle='--')
plt.plot(forecast_df_inverted.index, forecast_df_inverted['CV'], label='Forecasted CV', linestyle='--', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('CV')
plt.title('Actual, Fitted, and Forecasted CV (1896 - 2024)')
plt.xticks(range(df_total_reset.index[0], 2025, 8))  # Adjust ticks as needed
plt.grid(True)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
cv_series=df_CV
plt.figure(figsize=(12,6))
plt.plot(cv_series)
plt.title('Time Series Plot of CV')
plt.xlabel('Time')
plt.ylabel('CV')
plt.show()
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(cv_series)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
if adf_result[1] > 0.05:
    cv_series_diff = cv_series.diff().dropna()
    d = 1  # Order of differencing
else:
    cv_series_diff = cv_series
    d = 0
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(cv_series_diff, ax=axes[0], lags=20)
plot_pacf(cv_series_diff, ax=axes[1], lags=20)
plt.show()
p = 1
q = 1
model = ARIMA(cv_series, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())
forecast_steps = 10  # Number of periods to forecast
forecast = model_fit.forecast(steps=forecast_steps)
last_date = cv_series.index[-1]
forecast_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='4Y')[1:]
plt.figure(figsize=(12,6))
plt.plot(cv_series, label='Actual CV')
plt.plot(forecast_index, forecast, label='Forecasted CV', linestyle='--', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('CV')
plt.title('Actual and Forecasted CV using ARIMA')
plt.show()
get_ipython().system('sudo apt update')
get_ipython().system('sudo apt install libcairo2-dev ffmpeg texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended dvipng cm-super')
get_ipython().system('pip install manimce')
get_ipython().system('pip uninstall manimce')
get_ipython().system('pip install manim')
get_ipython().system('pip install manim')
from manim import *
class CircleToSquare(Scene):
    def construct(self):
        circle = Circle()  # Create a circle
        square = Square()  # Create a square
        self.play(Create(circle))  # Animate creation of the circle
        self.wait(1)
        self.play(Transform(circle, square))  # Transform circle into square
        self.wait(1)
