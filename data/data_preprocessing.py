import pandas as pd

# Load the collected data from CSV file
raw_data_path = "sp500_prices_yahoo_2000_to_2024.csv"
data = pd.read_csv(raw_data_path, index_col=0)

# Display first few rows to inspect the data
print("Initial data:")
print(data.head())

# Handling missing values
data.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Removing redundant columns (if any)
data.drop(columns=['Adj Close'], inplace=True)

# Handling duplicate records (if any)
data.drop_duplicates(inplace=True)

# Adjusting data types (if needed, assuming Date is already datetime)
data['Close'] = data['Close'].astype(float)

# Handling outliers (if applicable)

# Normalizing or scaling data (if necessary)

# Feature engineering (if applicable)

# Save cleaned data back to CSV file
cleaned_file_path = "sp500_prices_cleaned_2000_to_2024.csv"
data.to_csv(cleaned_file_path)

print(f"Cleaned data saved to: {cleaned_file_path}")