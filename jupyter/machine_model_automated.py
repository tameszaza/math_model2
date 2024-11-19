
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
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("not_final3.csv")
parameters = ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']
split_years = {
    'normalized_country': 2000,
    'CV': 2000
}
default_split_year = 2004
fill_missing_with_zero = ['drug']
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
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
all_best_estimators = {}
df = pd.read_csv("not_final3.csv")
parameters = ['drug', 'equity','popularity', 'normalizedcountry', 'CV']
split_years = {
    'normalized_country': 2000,
    'CV': 2000
}
default_split_year = 2004
fill_missing_with_zero = ['drug']
best_estimators = {}
for parameter in parameters:
    print(f"\nProcessing parameter: {parameter}")
    print("=" * 50)
    param_columns = [col for col in df.columns if col.split('_')[0] == parameter]
    if 'Sport_-1' in df.columns:
        param_columns.insert(0, 'Sport_-1')
    df_param = df[param_columns]
    if 'Sport_-1' in df_param.columns:
        df_param = df_param.rename(columns={'Sport_-1': 'Sport'})
    else:
        df_param['Sport'] = df['Sport_-1']
    melted_data = df_param.reset_index().melt(id_vars=['index', 'Sport'], var_name='Year', value_name='Target')
    melted_data['Year'] = melted_data['Year'].str.extract('(\d+)').astype(int)
    if parameter in fill_missing_with_zero:
        melted_data['Target'] = melted_data['Target'].fillna(0)
    else:
        melted_data['Target'] = melted_data['Target'].fillna(-1)
    melted_data = melted_data.sort_values(by=['index', 'Year']).reset_index(drop=True)
    def create_lags(group):
        group['Lag_1'] = group['Target'].shift(1, fill_value=-1)
        group['Lag_2'] = group['Target'].shift(2, fill_value=-1)
        group['Lag_3'] = group['Target'].shift(3, fill_value=-1)
        return group
    result = melted_data.groupby('index').apply(create_lags).reset_index(drop=True)
    def fill_lags(row):
        for lag in ['Lag_1', 'Lag_2', 'Lag_3']:
            if row[lag] == -1:
                lower_lags = [l for l in ['Lag_1', 'Lag_2', 'Lag_3'] if row[l] != -1 and l > lag]
                if lower_lags:
                    row[lag] = row[lower_lags[0]]  # Use the most recent lower lag value
        return row
    result = result.groupby('index').apply(lambda group: group.apply(fill_lags, axis=1)).reset_index(drop=True)
    cleaned_result = result[(result[['Target', 'Lag_1', 'Lag_2', 'Lag_3']] != -1).all(axis=1)].reset_index(drop=True)
    print("\nData after cleaning and before modeling:")
    print(cleaned_result.head())
    split_year = split_years.get(parameter, default_split_year)
    before_split = cleaned_result[cleaned_result['Year'] < split_year].reset_index(drop=True)
    after_or_equal_split = cleaned_result[cleaned_result['Year'] >= split_year].reset_index(drop=True)
    print(f"\nData before {split_year}: {before_split.shape}")
    print(f"Data from {split_year} onwards: {after_or_equal_split.shape}")
    X_train = before_split.drop(['Target', 'index', 'Sport'], axis=1)
    y_train = before_split['Target']
    X_test = after_or_equal_split.drop(['Target', 'index', 'Sport'], axis=1)
    y_test = after_or_equal_split['Target']
    def visualize_data(X, y, title=""):
        print(f"\n{title} - Descriptive Statistics")
        print(X.describe())
        print(f"\n{title} Target Variable - Descriptive Statistics")
        print(y.describe())
        plt.figure(figsize=(10, 6))
        plt.plot(y.values, label="Target")
        plt.title(f"{title} Target Variable for {parameter}")
        plt.legend()
        plt.show()
    visualize_data(X_train, y_train, "Training Data")
    visualize_data(X_test, y_test, "Test Data")
    combined_data = pd.concat([X_train, y_train], axis=1)
    sns.pairplot(combined_data, diag_kind='kde')
    plt.suptitle(f'Pair Plot for Features (Train Data) - {parameter}', y=1.02)
    plt.show()
    plt.figure(figsize=(10, 6))
    correlation = combined_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f"Feature Correlation Heatmap - {parameter}")
    plt.show()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Feature Importance using RandomForest - {parameter}')
    plt.show()
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
    plt.title(f'PCA of Training Data - {parameter}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='Target')
    plt.show()
    def plot_rolling_stats(series, window=5, title="Rolling Mean and Standard Deviation"):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        plt.figure(figsize=(12, 6))
        plt.plot(series.values, label='Original', color='blue')
        plt.plot(rolling_mean.values, label=f'Rolling Mean (window={window})', color='orange')
        plt.plot(rolling_std.values, label=f'Rolling Std (window={window})', color='green')
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
        plot_rolling_stats(series, window, title=f"Rolling Statistics for {parameter}")
    def plot_acf_pacf(series, lags=20):
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(series, lags=lags, ax=plt.gca(), title="Autocorrelation")
        plt.subplot(122)
        plot_pacf(series, lags=lags, ax=plt.gca(), title="Partial Autocorrelation")
        plt.tight_layout()
        plt.show()
    print("\nStationarity Test and Rolling Statistics for Target Variable:")
    test_stationarity(y_train)
    print("\nAutocorrelation and Partial Autocorrelation for Target Variable:")
    plot_acf_pacf(y_train)
    results = []
    models = [
        {
            'name': 'CatBoost',
            'estimator': CatBoostRegressor(random_state=42, verbose=0),
            'param_grid': {
                'depth': [4, 6],
                'learning_rate': [0.01, 0.05],
                'iterations': [100]
            }
        },
        {
            'name': 'Random Forest',
            'estimator': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 5],
                'min_samples_split': [2, 5]
            }
        },
        {
            'name': 'Decision Tree',
            'estimator': DecisionTreeRegressor(random_state=42),
            'param_grid': {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
        },
        {
            'name': 'k-NN Regression',
            'estimator': KNeighborsRegressor(),
            'param_grid': {
                'n_neighbors': [3, 5],
                'weights': ['uniform', 'distance']
            }
        },
        {
            'name': 'XGBoost',
            'estimator': XGBRegressor(random_state=42, verbosity=0),
            'param_grid': {
                'n_estimators': [100],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.05]
            }
        },
        {
            'name': 'LightGBM',
            'estimator': LGBMRegressor(random_state=42),
            'param_grid': {
                'num_leaves': [31, 50],
                'learning_rate': [0.01, 0.05],
                'n_estimators': [100]
            }
        },
        {
            'name': 'Bagging',
            'estimator': BaggingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [10, 50],
                'max_samples': [0.5, 1.0],
                'max_features': [0.5, 1.0]
            }
        },
        {
            'name': 'AdaBoost',
            'estimator': AdaBoostRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.05, 1.0]
            }
        },
        {
            'name': 'Gradient Boosting',
            'estimator': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5]
            }
        },
    ]
    for model_dict in models:
        name = model_dict['name']
        estimator = model_dict['estimator']
        param_grid = model_dict['param_grid']
        print(f"\nTraining {name}...")
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
        if parameter == 'CV':
            best_estimators[name] = best_estimator  # Store the best estimator for 'CV'
    estimators_list = [
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
        ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
    ]
    final_estimator = LinearRegression()
    stacking_model = StackingRegressor(
        estimators=estimators_list,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    print("\nTraining Stacking Regressor...")
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
    if parameter == 'CV':
        best_estimators['Stacking Regressor'] = best_estimator  # Store the best stacking estimator for 'CV'
    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:")
    print(results_df)
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Model'], results_df['MSE'], color='skyblue')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.title(f'Model Comparison based on MSE - {parameter}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Model'], results_df['R2'], color='salmon')
    plt.xlabel('R^2 Score')
    plt.title(f'Model Comparison based on R^2 Score - {parameter}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    print(f"\nBest model for '{parameter}' is: {best_model_name}")
    if parameter not in all_best_estimators:
        all_best_estimators[parameter] = {}
    all_best_estimators[parameter][best_model_name] = best_estimator
    best_model = all_best_estimators[parameter][best_model_name]
    non_zero_counts = cleaned_result.groupby('Sport')['Target'].apply(lambda x: (x != 0).sum())
    most_data_sport = non_zero_counts.idxmax()
    print(f"Sport with the most non-zero '{parameter}' values: {most_data_sport}")
    sport_index = cleaned_result[cleaned_result['Sport'] == most_data_sport]['index'].unique()[0]
    sport_test_data = after_or_equal_split[after_or_equal_split['index'] == sport_index].reset_index(drop=True)
    if sport_test_data.empty:
        print(f"No test data available for sport: {most_data_sport}. Selecting another sport.")
        valid_sports = after_or_equal_split['Sport'].unique()
        non_zero_counts = non_zero_counts[non_zero_counts.index.isin(valid_sports)]
        most_data_sport = non_zero_counts.idxmax()
        print(f"New sport selected for '{parameter}': {most_data_sport}")
        sport_index = cleaned_result[cleaned_result['Sport'] == most_data_sport]['index'].unique()[0]
        sport_test_data = after_or_equal_split[after_or_equal_split['index'] == sport_index].reset_index(drop=True)
    print(f"Final selected sport for '{parameter}': {most_data_sport}")
    X_sport_test = sport_test_data.drop(['Target', 'index', 'Sport'], axis=1)
    y_sport_test = sport_test_data['Target']
    y_sport_pred = best_model.predict(X_sport_test)
    predictions_df = pd.DataFrame({
        'Actual': y_sport_test.values,
        'Predicted': y_sport_pred
    })
import pandas as pd
import matplotlib.pyplot as plt
def analyze_sport(sport_name=None):
    if not sport_name:
        non_zero_counts = cleaned_result.groupby('Sport')['Target'].apply(lambda x: (x != 0).sum())
        sport_name = non_zero_counts.idxmax()
    print(f"Analyzing sport: {sport_name}")
    sport_data = melted_data[melted_data['Sport'] == sport_name].reset_index(drop=True)
    sport_data = sport_data.sort_values(by='Year').reset_index(drop=True)
    sport_data = create_lags(sport_data)
    sport_data = sport_data.apply(fill_lags, axis=1)
    prediction_data = sport_data[(sport_data[['Lag_1', 'Lag_2', 'Lag_3']] != -1).all(axis=1)].reset_index(drop=True)
    X_sport = prediction_data[['Year', 'Lag_1', 'Lag_2', 'Lag_3']]
    y_sport_actual = prediction_data['Target']
    years = prediction_data['Year']
    for model_name, best_model in best_estimators.items():
        print(f"\nUsing Best Model: {model_name}")
        if hasattr(best_model, 'feature_names_'):
            required_features = best_model.feature_names_
        else:
            required_features = X_sport.columns
        X_sport_corrected = X_sport[required_features]
        y_sport_pred = best_model.predict(X_sport_corrected)
        plot_df = pd.DataFrame({
            'Year': years,
            'Actual': y_sport_actual,
            'Predicted': y_sport_pred
        })
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['Year'], plot_df['Actual'], marker='o', label='Actual CV', color='blue')
        plt.plot(plot_df['Year'], plot_df['Predicted'], marker='x', linestyle='--', label=f'Predicted CV ({model_name})', color='orange')
        plt.title(f'Actual vs. Predicted CV Values over Years for {sport_name} ({model_name})')
        plt.xlabel('Year')
        plt.ylabel('CV Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
import pandas as pd
import matplotlib.pyplot as plt
def analyze_sport_combined(sport_name=None):
    if not sport_name:
        non_zero_counts = cleaned_result.groupby('Sport')['Target'].apply(lambda x: (x != 0).sum())
        sport_name = non_zero_counts.idxmax()
    print(f"Analyzing sport: {sport_name}")
    sport_data = melted_data[melted_data['Sport'] == sport_name].reset_index(drop=True)
    sport_data = sport_data.sort_values(by='Year').reset_index(drop=True)
    sport_data = create_lags(sport_data)
    sport_data = sport_data.apply(fill_lags, axis=1)
    prediction_data = sport_data[(sport_data[['Lag_1', 'Lag_2', 'Lag_3']] != -1).all(axis=1)].reset_index(drop=True)
    X_sport = prediction_data[['Year', 'Lag_1', 'Lag_2', 'Lag_3']]
    y_sport_actual = prediction_data['Target']
    years = prediction_data['Year']
    combined_df = pd.DataFrame({'Year': years, 'Actual': y_sport_actual})
    for model_name, best_model in best_estimators.items():
        print(f"\nUsing Best Model: {model_name}")
        if hasattr(best_model, 'feature_names_'):
            required_features = best_model.feature_names_
        else:
            required_features = X_sport.columns
        X_sport_corrected = X_sport[required_features]
        combined_df[f'Predicted ({model_name})'] = best_model.predict(X_sport_corrected)
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['Year'], combined_df['Actual'], marker='o', label='Actual CV', color='blue', linewidth=2)
    for col in combined_df.columns[2:]:
        plt.plot(combined_df['Year'], combined_df[col], linestyle='--', marker='x', label=col)
    plt.title(f'Actual vs. Predicted CV Values over Years for {sport_name} (All Models)')
    plt.xlabel('Year')
    plt.ylabel('CV Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
analyze_sport_combined()
analyze_sport_combined('Gymnastics')
