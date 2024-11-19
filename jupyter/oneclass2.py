
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
scaler = MinMaxScaler()
selected_columns['sustain'] = scaler.fit_transform(selected_columns[['sustain']])
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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
features = merged_df.drop(columns=['Year'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
models = {
    "IsolationForest": IsolationForest(),
    "OneClassSVM": OneClassSVM(kernel='rbf', gamma='scale')
}
param_grids = {
    "IsolationForest": {
        'n_estimators': [50, 100, 150],
        'max_samples': [0.8, 1.0],
        'contamination': [0.05, 0.1, 0.2],
        'random_state': [42]
    },
    "OneClassSVM": {
        'nu': [0.01, 0.05, 0.1, 0.2],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }
}
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='f1_macro',  # Change if necessary
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(scaled_features, [1]*len(scaled_features))  # Only normal data as label
    results[model_name] = {
        "best_params": grid_search.best_params_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_
    }
evaluation_scores = {}
evaluation_scores = {}
for model_name, model_info in results.items():
    model = model_info['best_estimator']
    predictions = model.predict(scaled_features)
    evaluation_scores[model_name] = {
        "Proportion of Anomalies": sum(predictions == -1) / len(predictions)
    }
print(f"\n\nOverall Models - Comparison:")
print(evaluation_scores)
plt.figure(figsize=(10, 5))
plt.bar(range(len(evaluation_scores)), [score["Proportion of Anomalies"] for _, score in evaluation_scores.items()])
plt.xticks(range(len(evaluation_scores)), list(evaluation_scores.keys()), rotation=45)
plt.ylabel("Proportion Anomalies Detected")
plt.title("Comparison Anomaly Method Performances")
plt.tight_layout()
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
df2 = merged_df2.copy()
normal_data = df2[df2['label'] == 1].drop(columns=['label'])
all_data = df2.drop(columns=['label'])
scaler = StandardScaler()
scaled_normal_data = scaler.fit_transform(normal_data)
scaled_all_data = scaler.transform(all_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_normal_data)
preds_if = iso_forest.predict(scaled_all_data)
preds_if = [0 if p == -1 else 1 for p in preds_if]
results['IsolationForest'] = classification_report(df2['label'], preds_if, output_dict=True)
oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
oc_svm.fit(scaled_normal_data)
preds_svm = oc_svm.predict(scaled_all_data)
preds_svm = [0 if p == -1 else 1 for p in preds_svm]
results['OneClassSVM'] = classification_report(df2['label'], preds_svm, output_dict=True)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
train_data = torch.tensor(scaled_normal_data, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
autoencoder = Autoencoder(input_dim=scaled_normal_data.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
for epoch in range(50):
    for batch in train_loader:
        batch = batch[0]
        optimizer.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
all_data_tensor = torch.tensor(scaled_all_data, dtype=torch.float32).to(device)
reconstructions = autoencoder(all_data_tensor).cpu().detach().numpy()
mse = np.mean(np.power(scaled_all_data - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 90)
preds_ae = [1 if e <= threshold else 0 for e in mse]
results['Autoencoder'] = classification_report(df2['label'], preds_ae, output_dict=True)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_normal_data)
distances = kmeans.transform(scaled_all_data).min(axis=1)
threshold = np.percentile(distances, 90)
preds_kmeans = [1 if d <= threshold else 0 for d in distances]
results['KMeans'] = classification_report(df2['label'], preds_kmeans, output_dict=True)
ell_env = EllipticEnvelope(contamination=0.1)
ell_env.fit(scaled_normal_data)
preds_ell = ell_env.predict(scaled_all_data)
preds_ell = [0 if p == -1 else 1 for p in preds_ell]
results['EllipticEnvelope'] = classification_report(df2['label'], preds_ell, output_dict=True)
pca = PCA(n_components=scaled_normal_data.shape[1])
pca.fit(scaled_normal_data)
proj = pca.inverse_transform(pca.transform(scaled_all_data))
mse_pca = np.mean(np.power(scaled_all_data - proj, 2), axis=1)
threshold_pca = np.percentile(mse_pca, 90)
preds_pca = [1 if mse <= threshold_pca else 0 for mse in mse_pca]
results['PCA'] = classification_report(df2['label'], preds_pca, output_dict=True)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(scaled_normal_data)
log_probs = gmm.score_samples(scaled_all_data)
threshold_gmm = np.percentile(log_probs, 10)
preds_gmm = [1 if lp >= threshold_gmm else 0 for lp in log_probs]
results['GMM'] = classification_report(df2['label'], preds_gmm, output_dict=True)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
preds_lof = lof.fit_predict(scaled_all_data)
preds_lof = [0 if p == -1 else 1 for p in preds_lof]
results['LocalOutlierFactor'] = classification_report(df2['label'], preds_lof, output_dict=True)
for model_name, metrics in results.items():
    print(f"Classification Report for {model_name}:")
    print(metrics)
    cm = confusion_matrix(df2['label'], [1 if p > 0 else 0 for p in preds_lof])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Anomaly", "Normal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid
df2 = merged_df2.copy()  # Ensure df2 contains your dataset with labels
normal_data = df2[df2['label'] == 1].drop(columns=['label'])
all_data = df2.drop(columns=['label'])
scaler = StandardScaler()
scaled_normal_data = scaler.fit_transform(normal_data)
scaled_all_data = scaler.transform(all_data)
results = {}
def evaluate_model(model, param_grid, model_name, X_train, X_test, y_true, gmm=False, pca=False, lof=False):
    best_score = -np.inf
    best_params = None
    best_preds = None
    for params in ParameterGrid(param_grid):
        if lof:
            preds = model.set_params(**params).fit_predict(X_test)
            preds = [0 if p == -1 else 1 for p in preds]
        elif gmm:
            model.set_params(**params).fit(X_train)
            scores = model.score_samples(X_test)
            threshold = np.percentile(scores, 10)
            preds = [1 if score >= threshold else 0 for score in scores]
        elif pca:
            model.set_params(**params).fit(X_train)
            reconstructions = model.inverse_transform(model.transform(X_test))
            mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
            threshold = np.percentile(mse, 90)
            preds = [1 if error <= threshold else 0 for error in mse]
        else:
            model.set_params(**params).fit(X_train, [1]*len(X_train))
            preds = model.predict(X_test)
            preds = [0 if p == -1 else 1 for p in preds]
        report = classification_report(y_true, preds, output_dict=True)
        f1_score = report['1']['f1-score']  # F1 for normal class (1)
        if f1_score > best_score:
            best_score = f1_score
            best_params = params
            best_preds = preds
    results[model_name] = {
        'best_params': best_params,
        'classification_report': classification_report(y_true, best_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, best_preds)
    }
    print(f"Best Params for {model_name}: {results[model_name]['best_params']}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_true, best_preds))
    cm = results[model_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(cm, display_labels=["Anomaly", "Normal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
param_grid_if = {
    'n_estimators': [100, 150],
    'max_samples': [0.8, 1.0],
    'contamination': [0.1, 0.12],
    'random_state': [42]
}
evaluate_model(IsolationForest(), param_grid_if, "IsolationForest", scaled_normal_data, scaled_all_data, df2['label'])
param_grid_ocsvm = {
    'nu': [0.02, 0.05],
    'kernel': ['rbf'],
    'gamma': ['scale', 0.1]
}
evaluate_model(OneClassSVM(), param_grid_ocsvm, "OneClassSVM", scaled_normal_data, scaled_all_data, df2['label'])
param_grid_ee = {
    'contamination': [0.1, 0.15],
    'random_state': [42]
}
evaluate_model(EllipticEnvelope(), param_grid_ee, "EllipticEnvelope", scaled_normal_data, scaled_all_data, df2['label'])
param_grid_gmm = {
    'n_components': [1, 2],
    'covariance_type': ['full', 'tied'],
    'random_state': [42]
}
evaluate_model(GaussianMixture(), param_grid_gmm, "GMM", scaled_normal_data, scaled_all_data, df2['label'], gmm=True)
param_grid_pca = {
    'n_components': [0.9, 0.95]
}
evaluate_model(PCA(), param_grid_pca, "PCA", scaled_normal_data, scaled_all_data, df2['label'], pca=True)
param_grid_lof = {
    'n_neighbors': [20, 25],
    'contamination': [0.1]
}
evaluate_model(LocalOutlierFactor(), param_grid_lof, "LocalOutlierFactor", scaled_normal_data, scaled_all_data, df2['label'], lof=True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
X = df2.drop(columns=['label'])
y = df2['label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("Unique classes in y_train:", np.unique(y_train))
print("Unique classes in y_val:", np.unique(y_val))
results = {}
def evaluate_pca(param_grid, model_name, X_train, y_train, X_val, y_val):
    best_score = -np.inf
    best_params = None
    best_preds = None
    for params in ParameterGrid(param_grid):
        model = PCA(**params)
        model.fit(X_train)
        X_val_reconstructed = model.inverse_transform(model.transform(X_val))
        reconstruction_errors = np.mean((X_val - X_val_reconstructed) ** 2, axis=1)
        threshold = np.percentile(reconstruction_errors, 90)  # Top 10% most error
        preds = np.where(reconstruction_errors > threshold, 0, 1)  # Anomaly (0), Normal (1)
        report = classification_report(y_val, preds, output_dict=True)
        f1_score = report['1']['f1-score']
        if f1_score > best_score:
            best_score = f1_score
            best_params = params
            best_preds = preds
    results[model_name] = {
        'best_params': best_params,
        'classification_report': classification_report(y_val, best_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_val, best_preds)
    }
    print(f"Best Params for {model_name}: {results[model_name]['best_params']}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_val, best_preds))
    cm = results[model_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(cm, display_labels=["Anomaly", "Normal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
param_grid_if = {
    'n_estimators': [50, 100],
    'max_samples': [0.8, 1.0],
    'contamination': [0.1, 0.2],
    'random_state': [42]
}
evaluate_model(IsolationForest(), param_grid_if, "IsolationForest", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_ocsvm = {
    'nu': [0.05, 0.1],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}
evaluate_model(OneClassSVM(), param_grid_ocsvm, "OneClassSVM", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_ee = {
    'contamination': [0.1, 0.15],
    'random_state': [42]
}
evaluate_model(EllipticEnvelope(), param_grid_ee, "EllipticEnvelope", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_gmm = {
    'n_components': [1, 2],
    'covariance_type': ['full', 'tied'],
    'random_state': [42]
}
evaluate_model(GaussianMixture(), param_grid_gmm, "GMM", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_pca = {
    'n_components': [0.9, 0.95]  # Fraction of variance to retain
}
evaluate_pca(param_grid_pca, "PCA", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_lof = {
    'n_neighbors': [20, 30],
    'contamination': [0.1]
}
evaluate_model(LocalOutlierFactor(novelty=True), param_grid_lof, "LocalOutlierFactor", X_train_scaled, y_train, X_val_scaled, y_val, lof=True)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
X = df2.drop(columns=['label'])
y = df2['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
results = {}
def evaluate_classification_model(model, param_grid, model_name, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
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
    disp = ConfusionMatrixDisplay(cm, display_labels=["Anomaly", "Normal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'random_state': [42]
}
evaluate_classification_model(RandomForestClassifier(), param_grid_rf, "RandomForest", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200]
}
evaluate_classification_model(LogisticRegression(), param_grid_lr, "LogisticRegression", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
evaluate_classification_model(SVC(), param_grid_svc, "SVC", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_knn = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance']
}
evaluate_classification_model(KNeighborsClassifier(), param_grid_knn, "KNN", X_train_scaled, y_train, X_val_scaled, y_val)
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 10],
    'random_state': [42]
}
evaluate_classification_model(GradientBoostingClassifier(), param_grid_gb, "GradientBoosting", X_train_scaled, y_train, X_val_scaled, y_val)
from sklearn.ensemble import BaggingClassifier
param_grid_bagging = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
    'random_state': [42]
}
evaluate_classification_model(BaggingClassifier(), param_grid_bagging, "BaggingClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
from sklearn.ensemble import AdaBoostClassifier
param_grid_adaboost = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    'random_state': [42]
}
evaluate_classification_model(AdaBoostClassifier(), param_grid_adaboost, "AdaBoostClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
from xgboost import XGBClassifier
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'random_state': [42]
}
evaluate_classification_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, "XGBoostClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
from lightgbm import LGBMClassifier
param_grid_lgbm = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [-1, 10, 20],  # -1 means no limit
    'random_state': [42]
}
evaluate_classification_model(LGBMClassifier(), param_grid_lgbm, "LGBMClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
from sklearn.ensemble import ExtraTreesClassifier
param_grid_extra_trees = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'random_state': [42]
}
evaluate_classification_model(ExtraTreesClassifier(), param_grid_extra_trees, "ExtraTreesClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
from catboost import CatBoostClassifier
param_grid_catboost = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [6, 8, 10],
    'random_seed': [42]
}
evaluate_classification_model(CatBoostClassifier(verbose=0), param_grid_catboost, "CatBoostClassifier", X_train_scaled, y_train, X_val_scaled, y_val)
