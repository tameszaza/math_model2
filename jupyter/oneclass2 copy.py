
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
import pandas as pd
all_sports_data = []
for sport, df in sport_data_dict.items():
    df = df.reset_index()  # Ensure 'Year' is a column, not an index
    df['Sport'] = sport  # Add a new column to indicate the sport name
    all_sports_data.append(df)  # Append the DataFrame to the list
combined_sport_data = pd.concat(all_sports_data, ignore_index=True)
combined_sport_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('not_final3.csv')
selected_columns = df[['Sport_-1', 'estimate per event_2024']]
selected_columns = selected_columns.drop([65, 66, 67])
selected_columns = selected_columns.rename(columns={'estimate per event_2024': 'sustain', 'Sport_-1': 'Sport'})
selected_columns['sustain'] = np.log1p(selected_columns['sustain'])
print(selected_columns)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
def fill_drug_column(df):
    df['drug'] = df['drug'].ffill().bfill()  # Forward fill, then backward fill
    return df
def regression_imputation(df, column, degree=2, min_val=None, max_val=None):
    available_data = df[df[column].notna()]
    missing_data = df[df[column].isna()]
    if len(available_data) < 2:
        print(f"Skipping regression for {column}: Not enough data.")
        return df[column]
    if missing_data.empty:
        return df[column]
    X_train = available_data.index.values.reshape(-1, 1)  # Year as feature
    y_train = available_data[column].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    X_missing = missing_data.index.values.reshape(-1, 1)
    X_missing_poly = poly.transform(X_missing)
    predictions = model.predict(X_missing_poly)
    if min_val is not None:
        predictions = np.maximum(predictions, min_val)
    if max_val is not None:
        predictions = np.minimum(predictions, max_val)
    df.loc[missing_data.index, column] = predictions
    return df[column]
def fill_continuous_columns_with_regression(df, degree=1):
    for col in ['equity', 'popularity', 'normalizedcountry', 'CV']:
        min_val, max_val = None, None
        if col == 'normalizedcountry':
            min_val, max_val = 0.0, 1.0
        elif col == 'equity':
            min_val, max_val = 0.0, 0.5
        elif col == 'popularity':
            min_val, max_val = 0.0, 0.4  # Cap popularity
        elif col == 'CV':
            min_val, max_val = 0.0, 1.5  # Cap CV
        if df[col].notna().sum() < 2:
            print(f"Skipping {col} due to insufficient data.")
            continue
        df[col] = regression_imputation(df, col, degree=degree, min_val=min_val, max_val=max_val)
    return df
for sport, data in sport_data_dict.items():
    data = fill_drug_column(data)  # Fill drug column as before
    sport_data_dict[sport] = fill_continuous_columns_with_regression(data)
print(sport_data_dict['Alpine Skiing'].head(30))
import matplotlib.pyplot as plt
def plot_data_with_labels(data):
    """
    Plot data with distinct colors for points where label is 1 or 0.
    Parameters:
    data (pd.DataFrame): DataFrame containing the columns: 'drug', 'equity', 'popularity', 
                         'normalizedcountry', 'CV', 'label' with 'Year' as the index.
    """
    label_0 = data[data['label'] == 0]
    label_1 = data[data['label'] == 1]
    parameters = ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV']
    for param in parameters:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[param], color='gray', label='Trend', linestyle='--')
        plt.scatter(label_0.index, label_0[param], color='blue', label='Label 0', marker='o')
        plt.scatter(label_1.index, label_1[param], color='red', label='Label 1', marker='x')
        plt.title(f"{param.capitalize()} over Years")
        plt.xlabel("Year")
        plt.ylabel(param.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()
plot_data_with_labels(sport_data_dict['Biathlon'])
import pandas as pd
all_sports_data = []
for sport, df in sport_data_dict.items():
    df = df.reset_index()  # Ensure 'Year' is a column, not an index
    df['Sport'] = sport  # Add a new column to indicate the sport name
    all_sports_data.append(df)  # Append the DataFrame to the list
combined_sport_data = pd.concat(all_sports_data, ignore_index=True)
combined_sport_data
merged_df2 = combined_sport_data.merge(selected_columns, on='Sport', how='left').dropna()
merged_df = merged_df2
merged_df2 = merged_df2.drop(columns=['Sport'])
merged_df
normal_class_data = merged_df[merged_df['label'] == 1].drop(columns=['label', 'Sport'])
merged_df = normal_class_data# Check basic information
print(merged_df.info())
print(merged_df.head())
print(merged_df.describe())
merged_df = normal_class_data
import matplotlib.pyplot as plt
import seaborn as sns
merged_df.hist(bins=20, figsize=(15, 10))
plt.show()
for column in merged_df.select_dtypes(include=['float64', 'int64']).columns:
    sns.kdeplot(merged_df[column], label=column)
plt.legend()
plt.show()
corr_matrix = merged_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()
sns.pairplot(merged_df.select_dtypes(include=['float64', 'int64']))
plt.show()
for column in merged_df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=merged_df, x=column)
    plt.title(f'Boxplot of {column}')
    plt.show()
merged_df2.describe()
merged_df2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df2 = merged_df2.copy()
df2.hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df2, x='equity', y='popularity', hue='label', palette='coolwarm')
plt.title("Scatter Plot: Equity vs Popularity (Colored by Label)")
plt.xlabel("Equity")
plt.ylabel("Popularity")
plt.legend(title="Label", labels=["Anomaly", "Normal"])
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(6, 4))
sns.countplot(data=df2, x='label', palette='coolwarm')
plt.title("Class Distribution")
plt.xlabel("Class (1 = Normal, 0 = Anomaly)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Anomaly", "Normal"])
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib  # For saving models and scalers
import os  # For creating directories
os.makedirs("saved_models", exist_ok=True)
X = df2.drop(columns=['label'])
y = df2['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, "saved_models/scaler.pkl")
print("Scaler saved at 'saved_models/scaler.pkl'")
results = {}
def evaluate_classification_model(model, param_grid, model_name, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    model_path = f"saved_models/{model_name}_best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model '{model_name}' saved at '{model_path}'")
    preds = best_model.predict(X_val)
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'classification_report': classification_report(y_val, preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_val, preds)
    }
    print(f"Best Params for {model_name}: {results[model_name]['best_params']}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_val, preds))
    cm = results[model_name]['confusion_matrix']
    labels = np.unique([0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
models_and_params = {
    "RandomForest": (RandomForestClassifier(), {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'random_state': [42]
    }),
    "LogisticRegression": (LogisticRegression(), {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100, 200]
    }),
    "SVC": (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    }),
    "GradientBoosting": (GradientBoostingClassifier(), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 10],
        'random_state': [42]
    }),
    "BaggingClassifier": (BaggingClassifier(), {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
        'random_state': [42]
    }),
    "AdaBoostClassifier": (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1.0],
        'random_state': [42]
    }),
    "XGBoostClassifier": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 10],
        'random_state': [42]
    }),
    "LGBMClassifier": (LGBMClassifier(), {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [-1, 10, 20],
        'random_state': [42]
    }),
    "ExtraTreesClassifier": (ExtraTreesClassifier(), {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'random_state': [42]
    }),
    "CatBoostClassifier": (CatBoostClassifier(verbose=0), {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'depth': [6, 8, 10],
        'random_seed': [42]
    }),
}
for model_name, (model, param_grid) in models_and_params.items():
    evaluate_classification_model(model, param_grid, model_name, X_train_scaled, y_train, X_val_scaled, y_val)
X_val
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
scaler = joblib.load('saved_models/scaler.pkl')
model = joblib.load('saved_models/CatBoostClassifier_best_model.pkl')
if hasattr(model, 'predict_proba'):
    scaled_input = scaler.transform(X_val)  # Scale new input data
    probabilities = model.predict_proba(scaled_input)
    print("Predicted Probabilities:\n", probabilities)
    predicted_classes = np.argmax(probabilities, axis=1)
    print("Predicted Classes:\n", predicted_classes)
else:
    print(f"The model '{type(model).__name__}' does not support probabilistic predictions.")
f1_scores = {model_name: results[model_name]['classification_report']['macro avg']['f1-score'] for model_name in results}
sorted_f1_scores = dict(sorted(f1_scores.items(), key=lambda item: item[1], reverse=True))
plt.figure(figsize=(12, 8))
plt.bar(sorted_f1_scores.keys(), sorted_f1_scores.values(), color='skyblue')
plt.xlabel('Model', fontsize=14)
plt.ylabel('F1 Score (Macro Avg)', fontsize=14)
plt.title('F1 Score Comparison Across Models', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
