import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine

class StockDataProcessor:

    SCALER_FILE = "/app/models/new_scaler.pkl"
    NORMALISED_DATA_FILE = "/app/data/new_normalized_combined_data.csv"
    POSTGRES_PASSWORD = "NOT_SET"

    def __init__(self, directory="/app/new_stock_data"):
        self.directory = directory
        self.combined_data = pd.DataFrame()

        if self.POSTGRES_PASSWORD != "NOT_SET":
            self.engine = create_engine(f"postgresql+psycopg2://postgres:{self.POSTGRES_PASSWORD}@172.17.0.3:5432/ml_trader")

    def download_data(self, tickers):
        os.makedirs(self.directory, exist_ok=True)

        for ticker in tickers:
            file_path = f"{self.directory}/{ticker}_hourly_data.csv"
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path, skiprows=3, header=None)
                last_date = pd.to_datetime(existing_data[0]).max()
                print("------")
                print(last_date)
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                print("new file for " + ticker)
                start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')

            end_date = datetime.now().strftime('%Y-%m-%d')

            try:
                new_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')

                if new_data.empty:
                    raise ValueError(
                        f"Failed download: {ticker}: YFPricesMissingError('possibly de-listed; no price data found (1h {start_date} -> {end_date})')"
                    )
                if os.path.exists(file_path):
                    new_data.to_csv(file_path, mode='a', header=False)
                else:
                    new_data.to_csv(file_path)
            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")

    def process_data(self, tickers):
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                ticker = filename.split('_')[0]
                if ticker in tickers:
                    print(f"Processing file: {filename}")
                    file_path = os.path.join(self.directory, filename)
                    data = pd.read_csv(file_path, skiprows=2, header=None)
                    try:
                        data.columns = ['Datetime', 'Adj Close', 'High', 'Low', 'Open', 'Volume']
                        print(f"Column names in {filename}: {data.columns.tolist()}")
                        if 'Datetime' in data.columns:
                            filtered_data = data[['Datetime', 'Adj Close', 'Open', 'High', 'Low', 'Volume']].copy()
                            filtered_data.rename(columns={'Datetime': 'Date'}, inplace=True)
                        elif 'Date' in data.columns:
                            filtered_data = data[['Date', 'Adj Close', 'Open', 'High', 'Low', 'Volume']]
                        else:
                            print(f"Neither 'Datetime' nor 'Date' column found in {filename}. Exiting with failure.")
                            sys.exit(1)
                        print(f"Filtered DataFrame shape: {filtered_data.shape}")
                        filtered_data.loc[:, 'Ticker'] = ticker
                        self.combined_data = pd.concat([self.combined_data, filtered_data])

                        if self.POSTGRES_PASSWORD != "NOT_SET":
                            # Replace 'table_name' with the name of the table you want to insert the data into
                            self.combined_data.to_sql(ticker, self.engine, schema='ml', if_exists='replace', index=False)

                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

        self.combined_data.reset_index(drop=True, inplace=True)
        #print (self.combined_data.head())

    def normalize_data(self):
        scaler = MinMaxScaler()
        # Normalize all relevant columns
        columns_to_normalize = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
        normalized_data = pd.DataFrame(scaler.fit_transform(self.combined_data[columns_to_normalize]),
                                       columns=columns_to_normalize)
        normalized_data['Date'] = self.combined_data['Date'].values
        normalized_data['Ticker'] = self.combined_data['Ticker'].values
        normalized_data.dropna(inplace=True)
        with open(self.NORMALISED_DATA_FILE, mode='w', newline='') as file:
            normalized_data.to_csv(file, index=False)
        joblib.dump(scaler, self.SCALER_FILE)
        print("The normalized combined dataset has been saved to 'new_normalized_combined_data.csv'.")

    def plot_data(self, ticker):
        plt.figure(figsize=(15, 7))
        ticker_data = self.combined_data[self.combined_data['Ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Adj Close'])
        plt.title(f'{ticker} Adjusted Close Price', fontsize=16)
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Adjusted Close Price', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(['Adj Close'], prop={'size': 15})
        plt.savefig(f"/app/tmp/{ticker.lower()}_adj_close_price.png")

if __name__ == "__main__":


    '''
        most_traded_stocks = [
        "TSLA", "NVDA", "AAPL", "META", "LLY", "MSFT", "AMZN", "AMD", "GOOG", "NFLX",
        "BABA", "BA", "BAC", "BBBY", "BBY", "BIDU", "BIIB", "BKNG", "BMY", "BRK.B",
        "C", "CAT", "CCL", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
        "CSCO", "CVS", "CVX", "DAL", "DIS", "DISH", "DOW", "DUK", "EA", "EBAY",
        "F", "FDX", "GE", "GM", "GME", "GS", "HAL", "HD", "HON", "IBM",
        "INTC", "JNJ", "JPM", "KO", "LMT", "LOW", "LUV", "MA", "MCD", "MMM",
        "MO", "MRK", "MS", "MU", "NKE"
    ]
    '''


    tickers = [
        "TSLA", "NVDA", "AAPL", "META", "LLY", "MSFT", "AMZN", "AMD", "GOOG", "NFLX",
        "BABA", "BA", "BAC", "BBBY", "BBY", "BIDU", "BIIB", "BKNG", "BMY", "BRK.B",
        "C", "CAT", "CCL", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
        "CSCO", "CVS", "CVX", "DAL", "DIS", "DISH", "DOW", "DUK", "EA", "EBAY",
        "F", "FDX", "GE", "GM", "GME", "GS", "HAL", "HD", "HON", "IBM",
        "INTC", "JNJ", "JPM", "KO", "LMT", "LOW", "LUV", "MA", "MCD", "MMM",
        "MO", "MRK", "MS", "MU", "NKE"
    ]
    processor = StockDataProcessor()
    processor.download_data(tickers)
    processor.process_data(tickers)
    processor.normalize_data()
    # processor.plot_data('TSLA')