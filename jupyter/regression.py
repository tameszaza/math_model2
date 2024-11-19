
import pandas as pd
import numpy as np
df = pd.read_csv("not_final2.csv")
df.head()
allowed_set = {'Sport', 'drug', 'equity', 'label', 'popularity', 'normalized_country', 'normalized', 'CV', 'estimate per event'}
filtered_columns = [col for col in df.columns if col.split('_')[0] in allowed_set]
df_filtered = df[filtered_columns]
df_filtered
df_filtered.to_csv('filtered.csv')
allowed_set = {'Sport','equity',}
filtered_columns = [col for col in df.columns if col.split('_')[0] in allowed_set]
df_filtered2 = df[filtered_columns]
df_filtered2
df_filtered2.to_csv('equity.csv')
equity_data = df_filtered2.drop(columns=[ 'Sport_-1'])
equity_data
print(equity_data.head())  # Ensure years are columns, not part of data
df = equity_data
melted_data = df.reset_index().melt(id_vars='index', var_name='Year', value_name='Target')
melted_data['Year'] = melted_data['Year'].str.extract('(\d+)').astype(int)
melted_data['Target'] = melted_data['Target'].fillna(-1)
melted_data = melted_data.sort_values(by=['index', 'Year']).reset_index(drop=True)
def create_lags(group):
    group['Lag_1'] = group['Target'].shift(1, fill_value=-1)
    group['Lag_2'] = group['Target'].shift(2, fill_value=-1)
    group['Lag_3'] = group['Target'].shift(3, fill_value=-1)
    return group
result = melted_data.groupby('index').apply(create_lags).reset_index(drop=True)
print(result)
def fill_lags(row):
    for lag in ['Lag_3', 'Lag_2', 'Lag_1']:  # Iterate from higher to lower lag
        if row[lag] == -1:
            lower_lags = [l for l in ['Lag_1', 'Lag_2', 'Lag_3'] if row[l] != -1 and l < lag]
            if lower_lags:
                row[lag] = row[lower_lags[-1]]  # Use the most recent lower lag value
    return row
filled_result = result.groupby('index').apply(lambda group: group.apply(fill_lags, axis=1)).reset_index(drop=True)
print(filled_result)
def fill_lags(row):
    for lag in ['Lag_1', 'Lag_2', 'Lag_3']:
        if row[lag] == -1:
            lower_lags = [l for l in ['Lag_1', 'Lag_2', 'Lag_3'] if row[l] != -1 and l > lag]
            if lower_lags:
                row[lag] = row[lower_lags[0]]  # Use the most recent lower lag value
    return row
filled_result = filled_result.groupby('index').apply(lambda group: group.apply(fill_lags, axis=1)).reset_index(drop=True)
print(filled_result)
cleaned_result = filled_result[(filled_result[['Target', 'Lag_1', 'Lag_2', 'Lag_3']] != -1).all(axis=1)].reset_index(drop=True)
print(cleaned_result)
def prepare_regression_data(df):
    regression_data = df.groupby('index').apply(lambda group: group[['index','Year', 'Target']])
    regression_data = regression_data.reset_index(drop=True)
    regression_data.rename(columns={'Target': 'Equity'}, inplace=True)  # Rename Target to Equity
    regression_data = regression_data[regression_data['Equity'] != -1]
    return regression_data
regression_ready_data = prepare_regression_data(filled_result)
print(regression_ready_data)
def prepare_data_for_incremental_training(df):
    """
    Prepare data for incremental training for each sport.
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['index', 'Year', 'Equity']
    Returns:
        dict: Dictionary containing data points for each sport (index), split by train/test sets incrementally.
    """
    prepared_data = {}
    for sport in df['index'].unique():
        sport_data = df[df['index'] == sport].sort_values(by='Year').reset_index(drop=True)
        X = sport_data[['Year']].values
        y = sport_data['Equity'].values
        n = len(X)
        sport_datasets = []
        for i in range(2, n):
            sport_datasets.append({
                'train_X': X[:i],      # First i rows for training
                'train_y': y[:i],      # Corresponding target values
                'test_X': X[i].reshape(1, -1),  # The next point to predict
                'test_y': y[i]         # The true value for evaluation
            })
        prepared_data[sport] = sport_datasets
    return prepared_data
prepared_data = prepare_data_for_incremental_training(regression_ready_data)
before_2004 = cleaned_result[cleaned_result['Year'] < 2004].reset_index(drop=True)
after_or_equal_2004 = cleaned_result[cleaned_result['Year'] >= 2004].reset_index(drop=True)
print("Data before 2004:")
print(before_2004.head())
print("\nData from 2004 onwards:")
print(after_or_equal_2004.head())
before_2004
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
if 'index' in before_2004.columns:
    before_2004 = before_2004.drop('index', axis=1)
if 'index' in after_or_equal_2004.columns:
    after_or_equal_2004 = after_or_equal_2004.drop('index', axis=1)
X_train = before_2004.drop(['Target'], axis=1)
y_train = before_2004['Target']
X_test = after_or_equal_2004.drop(['Target'], axis=1)
y_test = after_or_equal_2004['Target']
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import pandas as pd
def visualize_data(X, y, title=""):
    print(f"\n{title} - Descriptive Statistics")
    print(X.describe())
    print(f"\n{title} Target Variable - Descriptive Statistics")
    print(y.describe())
    plt.figure(figsize=(10, 6))
    plt.plot(y, label="Target")
    plt.title(f"{title} Target Variable")
    plt.legend()
    plt.show()
visualize_data(X_train, y_train, "Training Data")
visualize_data(X_test, y_test, "Test Data")
sns.pairplot(before_2004, diag_kind='kde')
plt.suptitle('Pair Plot for Features (Train Data)', y=1.02)
plt.show()
plt.figure(figsize=(10, 6))
correlation = before_2004.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance using RandomForest')
plt.show()
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.title('PCA of Training Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(label='Target')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
def plot_rolling_stats(series, window=5, title="Rolling Mean and Standard Deviation"):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original', color='blue')
    plt.plot(rolling_mean, label=f'Rolling Mean (window={window})', color='orange')
    plt.plot(rolling_std, label=f'Rolling Std (window={window})', color='green')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
def test_stationarity(series, window=5):
    print(f"Results of Dickey-Fuller Test:")
    result = adfuller(series)
    print(f"Test Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")
    plot_rolling_stats(series, window)
def plot_acf_pacf(series, lags=20):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series, lags=lags, ax=plt.gca(), title="Autocorrelation")
    plt.subplot(122)
    plot_pacf(series, lags=lags, ax=plt.gca(), title="Partial Autocorrelation")
    plt.tight_layout()
    plt.show()
def plot_lag_correlations(data, lags=12):
    plt.figure(figsize=(10, 6))
    lagged_data = pd.concat([data.shift(i) for i in range(lags + 1)], axis=1)
    lagged_data.columns = [f'Lag_{i}' for i in range(lags + 1)]
    sns.heatmap(lagged_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Lagged Feature Correlation Heatmap")
    plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
def seasonal_decompose_analysis(series, model='additive', freq=4):
    decomposition = seasonal_decompose(series, model=model, period=freq)
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Observed')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
print("Stationarity Test and Rolling Statistics for Target Variable:")
test_stationarity(before_2004['Target'])
print("\nAutocorrelation and Partial Autocorrelation for Target Variable:")
plot_acf_pacf(before_2004['Target'])
print("\nLag Feature Correlation Analysis:")
print("\nSeasonal Decomposition Analysis:")
seasonal_decompose_analysis(before_2004['Target'], model='additive', freq=4)
y_train
models = [
    {
        'name': 'CatBoost',
        'estimator': CatBoostRegressor(random_state=42, verbose=0),
        'param_grid': {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200]
        }
    },
    {
        'name': 'Random Forest',
        'estimator': RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'Decision Tree',
        'estimator': DecisionTreeRegressor(random_state=42),
        'param_grid': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'k-NN Regression',
        'estimator': KNeighborsRegressor(),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    {
        'name': 'XGBoost',
        'estimator': XGBRegressor(random_state=42, verbosity=0),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    },
    {
        'name': 'LightGBM',
        'estimator': LGBMRegressor(random_state=42),
        'param_grid': {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }
    },
    {
        'name': 'Bagging',
        'estimator': BaggingRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 1.0],
            'max_features': [0.5, 1.0]
        }
    },
    {
        'name': 'AdaBoost',
        'estimator': AdaBoostRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 1.0]
        }
    },
    {
        'name': 'Gradient Boosting',
        'estimator': GradientBoostingRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    },
    {
        'name': 'SVR',
        'estimator': SVR(),
        'param_grid': {
            'C': [0.5],        # Narrow range for regularization strength
            'epsilon': [0.1],     # Fixed epsilon to avoid unnecessary grid search
            'kernel': ['linear'], # Linear kernel for faster computation
            'tol': [0.03]         # Increase tolerance for quicker convergence
        }
    }
]
results = []
for model_dict in models:
    name = model_dict['name']
    estimator = model_dict['estimator']
    param_grid = model_dict['param_grid']
    print(f"Training {name}...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Best Params: {best_params}")
    print(f"{name} - MSE: {mse}, R2: {r2}")
    results.append({
        'Model': name,
        'Best Params': best_params,
        'MSE': mse,
        'R2': r2
    })
estimators = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=5)),
    ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
]
final_estimator = LinearRegression()
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)
print("Training Stacking Regressor...")
param_grid = {
    'final_estimator__fit_intercept': [True, False]
}
grid_search = GridSearchCV(
    estimator=stacking_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Stacking Regressor - Best Params: {best_params}")
print(f"Stacking Regressor - MSE: {mse}, R2: {r2}")
results.append({
    'Model': 'Stacking Regressor',
    'Best Params': best_params,
    'MSE': mse,
    'R2': r2
})
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df)
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['MSE'], color='skyblue')
plt.xlabel('Mean Squared Error (MSE)')
plt.title('Model Comparison based on MSE')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['R2'], color='salmon')
plt.xlabel('R^2 Score')
plt.title('Model Comparison based on R^2 Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
