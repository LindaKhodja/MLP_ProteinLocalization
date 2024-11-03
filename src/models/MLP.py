import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from collections import Counter

# Load data
df = pd.read_csv('feature_matrix.csv')

# Map one-hot encoded columns back to a single 'Type' column
df['Type'] = df[['Is_SP', 'Is_MT', 'Is_CH', 'Is_TH', 'Is_Other']].idxmax(axis=1).str.replace('Is_', '')

# Define features and target
X = df.drop(columns=['Type', 'Is_SP', 'Is_MT', 'Is_CH', 'Is_TH', 'Is_Other'])
y = df['Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only to the training data

# Define a custom sampling strategy for SMOTE
# Set the target number of samples for each minority class relative to the majority class
sampling_strategy = {
    'CH': int(len(y[y == 'Other']) * 0.1),  # Aim for 10% of 'Other' samples
    'MT': int(len(y[y == 'Other']) * 0.2),  # Aim for 20% of 'Other' samples
    'SP': int(len(y[y == 'Other']) * 0.3),  # Aim for 30% of 'Other' samples
    'TH': int(len(y[y == 'Other']) * 0.05), # Aim for 5% of 'Other' samples
    # 'Other' is not included because it is the majority class
}

# Apply SMOTE with custom sampling strategy
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution after resampling for verification
print("Class distribution after SMOTE:", Counter(y_resampled))


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize MLP model with a grid search for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
}

mlp = MLPClassifier(max_iter=300, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)

# Fit the model with the best hyperparameters
grid_search.fit(X_train, y_train)
best_mlp = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_mlp.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
