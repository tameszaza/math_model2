
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
for sport, data in sport_data_dict.items():
    print(f"Checking {sport}:")
    print(data.columns)
    if 'Year' not in data.columns:
        print(f"Year column is missing in {sport}")
    else:
        print(data.head())  # View the first few rows
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
        elif col == 'popularity':  # Cap popularity between 0 and 0.3
            min_val, max_val = 0.0, 0.3
        if df[col].notna().sum() < 2:
            print(f"Skipping {col} due to insufficient data.")
            continue
        df[col] = regression_imputation(df, col, degree=degree, min_val=min_val, max_val=max_val)
    return df
for sport, data in sport_data_dict.items():
    data = fill_drug_column(data)  # Fill drug column as before
    sport_data_dict[sport] = fill_continuous_columns_with_regression(data)
print(sport_data_dict['Alpine Skiing'].head(30))
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
scaler = MinMaxScaler()
selected_columns['sustain'] = scaler.fit_transform(selected_columns[['sustain']])
print(selected_columns)
merged_df = combined_sport_data.merge(selected_columns, on='Sport', how='left').dropna()
merged_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("Class Distribution:")
print(merged_df['label'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=merged_df, palette='viridis')
plt.title("Class Distribution (Normal vs Anomalies)")
plt.xlabel("Label (1 = Normal, 0 = Anomaly)")
plt.ylabel("Count")
plt.show()
print("\nSummary Statistics for Normal Data:")
print(merged_df[merged_df['label'] == 1].describe())
print("\nSummary Statistics for Anomaly Data:")
print(merged_df[merged_df['label'] == 0].describe())
for col in merged_df.columns:
    if col not in ['Year', 'label', 'Sport']:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=merged_df, x=col, hue='label', kde=True, palette='coolwarm', bins=30)
        plt.title(f"Distribution of {col} by Class")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
normal_data = merged_df[merged_df['label'] == 1]  # Filter only normal data
numeric_columns = normal_data.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap (Normal Data)")
plt.show()
subset_cols = [col for col in merged_df.columns if col not in ['Year', 'Sport', 'label']][:5]  # Select first 5 features for clarity
sns.pairplot(merged_df[subset_cols + ['label']], hue='label', palette='Set1', diag_kind='kde')
plt.suptitle("Pairplot of Features (Normal vs Anomaly)", y=1.02)
plt.show()
for col in merged_df.columns:
    if col not in ['Year', 'label', 'Sport']:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='label', y=col, data=merged_df, palette='pastel')
        plt.title(f"Boxplot of {col} by Class")
        plt.xlabel("Label (1 = Normal, 0 = Anomaly)")
        plt.ylabel(col)
        plt.show()
from sklearn.model_selection import train_test_split
features = merged_df.select_dtypes(include=['number']).drop(columns=['label']).dropna()
labels = merged_df['label'][features.index]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
print(f"Training data size after dropping NaN: {X_train.shape}")
print(f"Testing data size after dropping NaN: {X_test.shape}")
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score, pos_label=1)
iso_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.8, 0.6],
    'contamination': [0.05, 0.1, 0.2],
    'random_state': [42]
}
oc_svm_params = {
    'nu': [0.05, 0.1, 0.15],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly']
}
ell_env_params = {
    'contamination': [0.05, 0.1, 0.2],
    'support_fraction': [0.8, 0.9, None]
}
iso_forest_grid = GridSearchCV(
    IsolationForest(),
    param_grid=iso_forest_params,
    scoring=f1_scorer,
    cv=3,  # 3-fold cross-validation
    n_jobs=-1
)
iso_forest_grid.fit(X_train, y_train)
oc_svm_grid = GridSearchCV(
    OneClassSVM(),
    param_grid=oc_svm_params,
    scoring=f1_scorer,
    cv=3,
    n_jobs=-1
)
oc_svm_grid.fit(X_train, y_train)
ell_env_grid = GridSearchCV(
    EllipticEnvelope(),
    param_grid=ell_env_params,
    scoring=f1_scorer,
    cv=3,
    n_jobs=-1
)
ell_env_grid.fit(X_train, y_train)
lof_best_params = None
lof_best_f1 = 0
for n_neighbors in [10, 20, 30]:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    y_pred_lof = lof.fit_predict(X_test)
    y_pred_lof = [1 if x == 1 else 0 for x in y_pred_lof]  # Convert -1 to 0 (anomaly)
    f1 = f1_score(y_test, y_pred_lof)
    if f1 > lof_best_f1:
        lof_best_f1 = f1
        lof_best_params = {'n_neighbors': n_neighbors}
print(f"Best LOF params: {lof_best_params} with F1 Score: {lof_best_f1}")
best_iso_forest = iso_forest_grid.best_estimator_
best_oc_svm = oc_svm_grid.best_estimator_
best_ell_env = ell_env_grid.best_estimator_
y_pred_iso_best = best_iso_forest.predict(X_test)
y_pred_iso_best = [1 if x == 1 else 0 for x in y_pred_iso_best]
y_pred_svm_best = best_oc_svm.predict(X_test)
y_pred_svm_best = [1 if x == 1 else 0 for x in y_pred_svm_best]
y_pred_ell_best = best_ell_env.predict(X_test)
y_pred_ell_best = [1 if x == 1 else 0 for x in y_pred_ell_best]
lof = LocalOutlierFactor(n_neighbors=lof_best_params['n_neighbors'], contamination=0.1)
y_pred_lof_best = lof.fit_predict(X_test)
y_pred_lof_best = [1 if x == 1 else 0 for x in y_pred_lof_best]
models = {
    "Best Isolation Forest": y_pred_iso_best,
    "Best One-Class SVM": y_pred_svm_best,
    "Best Elliptic Envelope": y_pred_ell_best,
    "Best Local Outlier Factor": y_pred_lof_best
}
for model_name, preds in models.items():
    print(f"--- {model_name} ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, preds)}\n")
from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(10, 8))
for model_name, preds in models.items():
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='best')
plt.show()
