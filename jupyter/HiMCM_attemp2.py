
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
