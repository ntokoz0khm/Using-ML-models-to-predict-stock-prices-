import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load cleaned data
file_path = "sp500_prices_cleaned_2000_to_2024.csv"
data = pd.read_csv(file_path, index_col=0)
data.index = pd.to_datetime(data.index)  # Convert index to datetime if not already

# Assuming 'Close' is our target variable (dependent variable)
X = data.drop(columns=['Close', 'Volume'])  # Features (independent variables)
y = data['Close']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Neural Network model
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer with one neuron for regression
])

# Compile the model
model_nn.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model_nn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                       validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Evaluate model performance
train_loss = model_nn.evaluate(X_train_scaled, y_train, verbose=0)
test_loss = model_nn.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Neural Network Model:")
print(f"Training Loss: {train_loss}")
print(f"Testing Loss: {test_loss}")

# Plotting actual vs. predicted values over time
y_pred_train_nn = model_nn.predict(X_train_scaled).flatten()
y_pred_test_nn = model_nn.predict(X_test_scaled).flatten()

plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Actual', color='blue')
plt.plot(X_test.index, y_pred_test_nn, label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('S&P 500 Close Price')
plt.title('Neural Network: Actual vs. Predicted over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training and validation loss over time (epochs)
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network: Training and Validation Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()