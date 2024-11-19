
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
sport_data_dict['Alpine Skiing']
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
        if df[col].notna().sum() < 2:
            print(f"Skipping {col} due to insufficient data.")
            continue
        df[col] = regression_imputation(df, col, degree=degree, min_val=min_val, max_val=max_val)
    return df
for sport, data in sport_data_dict.items():
    data = fill_drug_column(data)  # Fill drug column as before
    sport_data_dict[sport] = fill_continuous_columns_with_regression(data)
print(sport_data_dict['Alpine Skiing'].head(30))
sport_data_dict['Alpine Skiing']
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
merged_df = combined_sport_data.merge(selected_columns, on='Sport', how='left')
merged_df
import seaborn as sns
from sklearn.preprocessing import StandardScaler
cleaned_data = merged_df.drop(['Sport'], axis=1).dropna()  # Exclude 'Sport' column
X = cleaned_data.drop(['label'], axis=1)  # Features
y = cleaned_data['label']  # Labels
print(data.isnull().sum())
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x=y, palette='viridis')
plt.title('Class Distribution (0 vs 1)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
correlation_matrix = cleaned_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
for col in X.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(cleaned_data, x=col, hue='label', multiple='stack', kde=True, palette='coolwarm')
    plt.title(f'Distribution of {col} by Label')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
sns.pairplot(cleaned_data, vars=['equity', 'popularity', 'normalizedcountry', 'CV'], hue='label', palette='coolwarm')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()
for col in ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV', 'sustain']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y=col, data=merged_df)
    plt.title(f'Boxplot of {col} by Label')
    plt.xlabel('Label')
    plt.ylabel(col)
    plt.show()
for col in ['drug', 'equity', 'popularity', 'normalizedcountry', 'CV', 'sustain']:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='label', y=col, data=merged_df, split=True, inner='quartile')
    plt.title(f'Violin Plot of {col} by Label')
    plt.xlabel('Label')
    plt.ylabel(col)
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def train_evaluate_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)
print("Training on imbalanced data...")
baseline_results = train_evaluate_model(X_train, y_train, X_test, y_test)
print("Training with SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_results = train_evaluate_model(X_train_smote, y_train_smote, X_test, y_test)
print("Training with Downsampling...")
undersampler = RandomUnderSampler(random_state=42)
X_train_downsample, y_train_downsample = undersampler.fit_resample(X_train, y_train)
downsample_results = train_evaluate_model(X_train_downsample, y_train_downsample, X_test, y_test)
results = {
    "Baseline": baseline_results["weighted avg"],
    "SMOTE": smote_results["weighted avg"],
    "Downsampling": downsample_results["weighted avg"]
}
results_df = pd.DataFrame(results).T
print(results_df)
results_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Resampling Methods on Model Performance")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
os.makedirs("class_results", exist_ok=True)
os.makedirs("class_weights", exist_ok=True)
os.makedirs("class_plots", exist_ok=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, RocCurveDisplay
def train_and_evaluate_model(model, model_name, param_grid=None):
    """
    Train and evaluate a given model.
    If param_grid is provided, use GridSearchCV for hyperparameter tuning.
    """
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = model.fit(X_train, y_train)
        best_params = None
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model: {model_name}\nAccuracy: {acc}")
    model_path = f"class_weights/{model_name}_model.pkl"
    pd.to_pickle(best_model, model_path)
    with open(f"class_results/{model_name}_report.txt", 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nBest Params: {best_params}")
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig(f"class_plots/{model_name}_roc_curve.png")
    plt.close()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"class_plots/{model_name}_confusion_matrix.png")
    plt.close()
    return best_model
from sklearn.linear_model import LogisticRegression
logreg_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
train_and_evaluate_model(LogisticRegression(), "LogisticRegression", logreg_params)
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
class ClassificationPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = self._initialize_models()
        self.results_dir = "class_results"
        self.weights_dir = "class_weights"
        self.plots_dir = "class_plots"
        self._prepare_directories()
    def _initialize_models(self):
        """Initialize models with hyperparameter grids for tuning."""
        return {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=500),
                "params": {"C": [0.01, 0.1, 1, 10]}
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(),
                "params": {"max_depth": [3, 5, 10, None]}
            },
            "RandomForest": {
                "model": RandomForestClassifier(),
                "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(),
                "params": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100]}
            },
            "AdaBoost": {
                "model": AdaBoostClassifier(),
                "params": {"n_estimators": [50, 100]}
            },
            "Bagging": {
                "model": BaggingClassifier(),
                "params": {"n_estimators": [10, 50, 100]}
            },
            "SVM": {
                "model": SVC(probability=True),
                "params": {"C": [0.1, 1], "kernel": ["rbf"]}
            },
            "MLP": {
                "model": MLPClassifier(max_iter=1000),
                "params": {"hidden_layer_sizes": [(50,), (100,), (50, 50)]}
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 7]}
            },
            "GaussianNB": {
                "model": GaussianNB(),
                "params": None  # No hyperparameters for Naive Bayes
            },
            "LDA": {
                "model": LinearDiscriminantAnalysis(),
                "params": None  # No hyperparameters for LDA
            }
        }
    def _prepare_directories(self):
        """Create directories for saving results, weights, and plots."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    def train_and_evaluate_model(self, model_name, model, param_grid=None):
        """Train and evaluate a single model."""
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_model = model.fit(self.X_train, self.y_train)
            best_params = None
        y_pred = best_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Model: {model_name}\nAccuracy: {acc}")
        model_path = os.path.join(self.weights_dir, f"{model_name}_model.pkl")
        pd.to_pickle(best_model, model_path)
        with open(os.path.join(self.results_dir, f"{model_name}_report.txt"), "w") as f:
            f.write(classification_report(self.y_test, y_pred))
            f.write(f"\nBest Params: {best_params}")
        RocCurveDisplay.from_estimator(best_model, self.X_test, self.y_test)
        plt.savefig(os.path.join(self.plots_dir, f"{model_name}_roc_curve.png"))
        plt.close()
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.plots_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
        return best_model, acc
    def run(self):
        """Run training and evaluation for all models."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        results = []
        for model_name, config in self.models.items():
            print(f"Training {model_name}...")
            best_model, acc = self.train_and_evaluate_model(model_name, config["model"], config["params"])
            results.append({"Model": model_name, "Accuracy": acc})
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, "model_comparison.csv"), index=False)
        print("All models trained and evaluated.")
pipeline = ClassificationPipeline(X, y)
pipeline.run()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
results_df = pd.read_csv('class_results/model_comparison.csv')
results_df = results_df.sort_values(by='Accuracy', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('class_plots/model_accuracy_comparison.png')
plt.show()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Using device:", tf.test.gpu_device_name() or "CPU")
print("Is TensorFlow built with GPU support?", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Available GPUs: {gpus}")
    for gpu in gpus:
        print(f"GPU details: {gpu}")
else:
    print("No GPUs available.")
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
import joblib
class AdvancedModelsPipeline:
    def __init__(self, X, y):
        self.X = self._remove_highly_correlated_features(X)
        self.y = y
        self.results = []
        os.makedirs("advanced_model_weights", exist_ok=True)
        os.makedirs("advanced_model_plots", exist_ok=True)
        os.makedirs("advanced_model_results", exist_ok=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=0.2, random_state=42)
        self.X_train_dl = self.X_train.values.reshape(-1, self.X_train.shape[1], 1)
        self.X_test_dl = self.X_test.values.reshape(-1, self.X_test.shape[1], 1)
    def _remove_highly_correlated_features(self, X, threshold=0.95):
        """Remove features with correlation above the threshold."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        print(f"Removing highly correlated features: {to_drop}")
        return X.drop(columns=to_drop)
    def train_and_evaluate(self, model, model_name):
        """Train and evaluate a given model."""
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        model_path = f"advanced_model_weights/{model_name}_model.pkl"
        joblib.dump(model, model_path)
        report_path = f"advanced_model_results/{model_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        self.results.append({'Model': model_name, 'Accuracy': acc})
        print(f"Model: {model_name} | Accuracy: {acc}")
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(self.X_test)
        else:
            print(f"Skipping ROC for {model_name}, no suitable method found.")
            return
        if np.any(np.isnan(y_prob)):
            print(f"Skipping {model_name} due to NaN in probabilities.")
            return
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc(fpr, tpr)))
        plt.title(f'{model_name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f"advanced_model_plots/{model_name}_roc_curve.png")
        plt.close()
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"advanced_model_plots/{model_name}_confusion_matrix.png")
        plt.close()
    def run(self):
        """Run training and evaluation for all models."""
        self.train_and_evaluate(QuadraticDiscriminantAnalysis(), "QDA")
        self.train_and_evaluate(GaussianNB(), "Naive Bayes")
        stack = StackingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('lr', LogisticRegression())])
        self.train_and_evaluate(stack, "Stacking Classifier")
        vote = VotingClassifier(estimators=[
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC(probability=True))])
        self.train_and_evaluate(vote, "Voting Classifier")
        self.train_cnn()
        self.train_rnn(LSTM, "LSTM")
        self.train_rnn(GRU, "GRU")
    def train_cnn(self):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.X_train_dl.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.X_train_dl, self.y_train, epochs=10, validation_data=(self.X_test_dl, self.y_test), verbose=0)
        self._save_deep_model_results(model, "CNN", history)
    def train_rnn(self, rnn_layer, model_name):
        model = Sequential([
            rnn_layer(50, return_sequences=False, input_shape=(self.X_train_dl.shape[1], 1)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.X_train_dl, self.y_train, epochs=10, validation_data=(self.X_test_dl, self.y_test), verbose=0)
        self._save_deep_model_results(model, model_name, history)
    def _save_deep_model_results(self, model, model_name, history):
        acc = history.history['val_accuracy'][-1]
        self.results.append({'Model': model_name, 'Accuracy': acc})
        model.save(f'advanced_model_weights/{model_name}_model.h5')
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"advanced_model_plots/{model_name}_learning_curve.png")
        plt.close()
    def plot_comparison(self):
        """Plot a comparison of model accuracies."""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values(by='Accuracy', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Accuracy')
        plt.ylabel('Model')
        plt.xlim(0, 1)  # Assuming accuracy is between 0 and 1
        plt.savefig("advanced_model_plots/model_comparison.png")
        plt.show()
print(X.isnull().sum())  # Check for NaN in features
print(y.isnull().sum())  # Check for NaN in labels
import tensorflow as tf
print(tf.sysconfig.get_build_info()["cuda_version"])  # TensorFlow's expected CUDA version
print(tf.sysconfig.get_build_info()["cudnn_version"])  # TensorFlow's expected cuDNN version
pipeline = AdvancedModelsPipeline(X, y)
pipeline.run()
pipeline.plot_comparison()
