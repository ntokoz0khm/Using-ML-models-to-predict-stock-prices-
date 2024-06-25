import yfinance as yf
import pandas as pd
import os

def collect_stock_data(ticker, start_date, end_date):
    # Fetch historical market data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

if __name__ == "__main__":
    # You can replace 'NPN.JO' with the ticker of the stock you are interested in
    ticker_symbol = "^GSPC"
    
    # Define the date range for the data collection
    start_date = '2000-01-01'
    end_date = '2024-01-01'
    
    # Define the path to save the collected data
    raw_data_path = "sp500_prices_yahoo_2021_to_2024.csv"
    
    # Ensure the raw data directory exists
    #os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # Collect stock data
    stock_data = collect_stock_data(ticker_symbol, start_date, end_date)
    
    # Save the stock data to a CSV file
    stock_data.to_csv(raw_data_path)
    
    print(f"Stock data for {ticker_symbol} saved to {raw_data_path}")