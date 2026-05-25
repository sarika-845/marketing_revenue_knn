import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# -------------------------------
# Step 1: Load datasets
# -------------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Merge both datasets
df = pd.concat([train, test], ignore_index=True)

# Inspect columns
print("Columns in dataset:", df.columns)

# -------------------------------

# Step 2: Handle missing values
# -------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------------

# Step 3: Keep only numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Drop 'id' since it's not a predictor
numeric_df = numeric_df.drop('id', axis=1)

# Split features and target
X = numeric_df.drop('sales_revenue', axis=1)
y = numeric_df['sales_revenue']

# -------------------------------
# Step 4: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 6: Build and train KNN model
# -------------------------------
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# -------------------------------
# Step 7: Predictions
# -------------------------------
y_pred = knn.predict(X_test)

# -------------------------------
# Step 8: Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# -------------------------------
print("Training features:", X.columns.tolist())

# Step 9: Save model & scaler
# -------------------------------
joblib.dump(knn, r"C:\Users\sony\OneDrive\Desktop\marketing_revenue_knn\model.pkl")
joblib.dump(scaler, r"C:\Users\sony\OneDrive\Desktop\marketing_revenue_knn\scaler.pkl")
print("Training features:", X.columns.tolist())


print("✅ Model and scaler saved successfully!") 
