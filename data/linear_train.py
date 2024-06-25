import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load cleaned data
file_path = "sp500_prices_cleaned_2021_to_2024.csv"
data = pd.read_csv(file_path, index_col=0)
data.index = pd.to_datetime(data.index)  # Convert index to datetime if not already

# Assuming 'Close' is our target variable (dependent variable)
X = data.drop(columns=['Close', 'Volume'])  # Features (independent variables)
y = data['Close']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial features
poly = PolynomialFeatures(degree=2)  # Example: degree=2 for quadratic terms
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train polynomial regression model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predictions
y_pred_train = model_poly.predict(X_train_poly)
y_pred_test = model_poly.predict(X_test_poly)

# Evaluate model performance
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Polynomial Regression Model (Degree 2):")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")

# Plotting actual and predicted values over time
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Actual', color='blue')
plt.plot(X_test.index, y_pred_test, label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('S&P 500 Close Price')
plt.title('Polynomial Regression (Degree 2): Actual vs. Predicted over Time')
plt.legend()
plt.grid(True)
plt.show()
