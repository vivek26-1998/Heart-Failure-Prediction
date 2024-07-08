import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pickle

# Generate a synthetic dataset
np.random.seed(42)

data = {
    'Age': np.random.randint(29, 77, size=1000),
    'Sex': np.random.choice(['M', 'F'], size=1000),
    'ChestPainType': np.random.choice(['TA', 'ATA', 'NAP', 'ASY'], size=1000),
    'RestingBP': np.random.randint(80, 200, size=1000),
    'Cholesterol': np.random.randint(100, 400, size=1000),
    'FastingBS': np.random.choice([0, 1], size=1000),
    'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], size=1000),
    'MaxHR': np.random.randint(60, 202, size=1000),
    'ExerciseAngina': np.random.choice(['Y', 'N'], size=1000),
    'Oldpeak': np.random.uniform(0.0, 6.2, size=1000),
    'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], size=1000),
    'HeartDisease': np.random.choice([0, 1], size=1000)
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# List of numerical and categorical columns
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Preprocessing for numerical data: scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model to disk
model_filename = 'heart_failure_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)