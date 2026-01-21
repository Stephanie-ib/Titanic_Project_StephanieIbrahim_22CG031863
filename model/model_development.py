"""
Titanic Survival Prediction - Model Development
Author: [Your Name]
Matric No: [Your Matric Number]

This script trains a Logistic Regression model to predict Titanic survival.
Features used: Pclass, Sex, Age, Fare, Embarked
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION - MODEL DEVELOPMENT")
print("=" * 60)

# ==================== STEP 1: LOAD DATASET ====================
print("\n[STEP 1] Loading Titanic Dataset...")

# Option 1: Load from seaborn (recommended for local development)
try:
    import seaborn as sns
    df = sns.load_dataset('titanic')
    print("✓ Dataset loaded from seaborn")
except:
    # Option 2: Load from CSV (if you have titanic.csv)
    print("⚠ Seaborn not available. Please download titanic.csv")
    print("Download from: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = pd.read_csv('titanic.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ==================== STEP 2: FEATURE SELECTION ====================
print("\n[STEP 2] Selecting 5 Features...")

# Selected features: Pclass, Sex, Age, Fare, Embarked
selected_features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']

# Keep only selected columns
df = df[selected_features].copy()
print(f"✓ Selected features: {selected_features[:-1]}")
print(f"✓ Target variable: survived")

# ==================== STEP 3: HANDLE MISSING VALUES ====================
print("\n[STEP 3] Handling Missing Values...")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing Age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing Embarked with mode (most common value)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Fill missing Fare with median
df['fare'].fillna(df['fare'].median(), inplace=True)

# Drop rows with missing Survived values
df.dropna(subset=['survived'], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("✓ All missing values handled")

# ==================== STEP 4: ENCODE CATEGORICAL VARIABLES ====================
print("\n[STEP 4] Encoding Categorical Variables...")

# Encode Sex: male=1, female=0
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
print("✓ Sex encoded: male=1, female=0")

# Encode Embarked: C=0, Q=1, S=2
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print("✓ Embarked encoded: C=0, Q=1, S=2")

print("\nDataset after encoding:")
print(df.head())

# ==================== STEP 5: PREPARE FEATURES AND TARGET ====================
print("\n[STEP 5] Preparing Features and Target...")

X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = df['survived']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ==================== STEP 6: SPLIT DATA ====================
print("\n[STEP 6] Splitting Data into Train and Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ==================== STEP 7: FEATURE SCALING ====================
print("\n[STEP 7] Applying Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ==================== STEP 8: TRAIN LOGISTIC REGRESSION MODEL ====================
print("\n[STEP 8] Training Logistic Regression Model...")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("✓ Model training completed")

# ==================== STEP 9: EVALUATE MODEL ====================
print("\n[STEP 9] Evaluating Model Performance...")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# ==================== STEP 10: SAVE MODEL AND SCALER ====================
print("\n[STEP 10] Saving Model and Scaler...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save both model and scaler in one pickle file
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': ['pclass', 'sex', 'age', 'fare', 'embarked']
}

with open('model/titanic_survival_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model and scaler saved to: model/titanic_survival_model.pkl")

# ==================== STEP 11: TEST MODEL PERSISTENCE ====================
print("\n[STEP 11] Testing Model Persistence (Reload & Predict)...")

# Load the saved model
with open('model/titanic_survival_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']

print("✓ Model and scaler loaded successfully")

# Test prediction with a sample passenger
# Example: 3rd class, male, age 22, fare 7.25, embarked from S
test_passenger = [[3, 1, 22, 7.25, 2]]
test_passenger_scaled = loaded_scaler.transform(test_passenger)
prediction = loaded_model.predict(test_passenger_scaled)

print("\nTest Prediction:")
print(f"Passenger details: Pclass=3, Sex=Male, Age=22, Fare=7.25, Embarked=S")
print(f"Prediction: {'SURVIVED' if prediction[0] == 1 else 'DID NOT SURVIVE'}")

print("\n" + "=" * 60)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nNext steps:")
print("1. Run the Flask app: python app.py")
print("2. Open browser at: http://localhost:5001")
print("=" * 60)
