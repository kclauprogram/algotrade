import datetime
import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
from ib_insync import IB, Stock, BarData, util, Contract

# Constants
IB_HOST = '127.0.0.1'
IB_PORT = 7496  # or 7496 for the paper trading port
IB_CLIENT_ID = 1  # Choose a unique client ID
CSV_FILE = 'sp500_prices.csv'
output_path = 'D:/project/data/sp500/'
REQUEST_DELAY = 10  # seconds

# BATCH_SIZE = 20 # Alternative batched approach
# BATCH_DELAY = 200

def get_sp500_symbols():
    """
    Retrieves the list of S&P 500 stock symbols from Wikipedia.

    Returns:
        list: A list of S&P 500 stock symbols.  Handles multiple share classes.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    symbols = df_sp500['Symbol']
    
    return symbols


def get_closing_price(ib: IB, symbol: str, time: str) -> BarData | None:
    """
    Fetches the closing price for a given stock symbol.

    Args:
        ib: The IB connection object.
        symbol: The stock symbol.

    Returns:
        BarData | None:  The BarData object containing the closing price, or None on error.
    """
    try:
        contract = Stock(symbol, 'SMART', 'USD')  # SMART routing, USD currency
        # ib.qualifyContracts(contract) #no need since we use historicalData

        # Fetch historical data for the last trading day
        # bars = ib.reqHistoricalData(
        #     contract,
        #     endDateTime='',  # Empty string for the most recent data
        #     durationStr='1 D',
        #     barSizeSetting='1 day',
        #     whatToShow='TRADES',
        #     useRTH=True,
        #     formatDate=1
        # )
        # Fetch historical data using a specific end date (yesterday)
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        end_date_time = datetime.datetime.combine(today, datetime.time(16, 0, 0))  # Market close time (EST)

        bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date_time, #yesterday 4pm
                durationStr= time,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=0 #add a timeout
        )

        if bars:
            return bars  # Return the first (and only) bar
        else:
            print(f"No data found for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    

def save_to_csv(data: list, name: str):
    """Saves the stock data to a CSV file"""

    df = pd.DataFrame(data, columns=['date', 'close', 'open', 'average'])
    #df['symbol'] = name
    print(df)
    path = 'D:/project/data/sp500_tmp/' + name + '.csv'

    # Append to CSV file
    df.to_csv(path, index=False)
   

def main(duration : str):
    """Main function to run the data pipeline."""
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        print("Connected to IB Gateway/TWS.")

        symbols = get_sp500_symbols()
        print(f"Fetched {len(symbols)} S&P 500 symbols.")

        # Sequential requests with delay
        for i, symbol in enumerate(symbols):
            print(f"Fetching data for {symbol} ({i+1}/{len(symbols)})")
            bar_data = get_closing_price(ib, symbol, duration)
            print(bar_data)
            time.sleep(1)
            price_data = bar_data
            """
            price_data = [bar_data.date,
                    symbol,
                    bar_data.close,
                   bar_data.open,
                    bar_data.average]
            
            if bar_data:
                price_data = {
                    'date': bar_data.date,
                    'symbol': symbol,
                    'close': bar_data.close,
                    'open': bar_data.open,
                    'average': bar_data.average
                }
            """
            print(price_data)
            save_to_csv(price_data, symbol)
            time.sleep(REQUEST_DELAY)  # Delay to avoid rate limits


        # # batched requests
        # for i in range(0, len(symbols), BATCH_SIZE):
        #      batch_symbols = symbols[i:i + BATCH_SIZE]
        #      print(f"Fetching data for batch {i // BATCH_SIZE + 1} ({len(batch_symbols)} symbols)")
        #      tasks = [get_closing_price(ib, symbol) for symbol in batch_symbols]
        #      results = ib.run(tasks)

        #      for symbol, bar_data in zip(batch_symbols, results):
        #          if bar_data:
        #              all_data.append({
        #                   'Date': bar_data.date,
        #                   'Symbol': symbol,
        #                   'Close Price': bar_data.close
        #              })
        #      time.sleep(BATCH_DELAY)  # Delay after each batch.

    finally:
        ib.disconnect()
        print("Disconnected from IB Gateway/TWS.")
        
    return 0

#main()