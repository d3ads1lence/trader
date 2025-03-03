import os
import pandas as pd
import requests
import numpy as np
import time

class BinanceDataFetcher:
    def __init__(self, symbol="BTCUSDT", interval="1h", days=365, filename="historical_data.csv"):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.filename = filename
    
    def fetch_data(self):
        # Check if the historical data file exists
        if os.path.exists(self.filename):
            print("Loading data from file...")
            df = pd.read_csv(self.filename)
        else:
            print("Downloading data from Binance...")
            millis_per_day = 24 * 60 * 60 * 1000
            end_time = int(time.time() * 1000)
            start_time = end_time - (self.days * millis_per_day)
            all_data = []
            
            while start_time < end_time:
                params = {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000
                }
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                start_time = data[-1][0] + 1  # Move start time forward
                time.sleep(0.5)  # Prevent rate limiting
            
            df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                                 "close_time", "quote_asset_volume", "num_trades", 
                                                 "taker_buy_base", "taker_buy_quote", "ignore"])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
            
            # Save data to file
            df.to_csv(self.filename, index=False)
            print("Data saved to file.")
        
        return df

class EMACrossoverStrategy:
    def __init__(self, short_window=9, long_window=21):
        self.short_window = short_window
        self.long_window = long_window
        
    def apply_strategy(self, df):
        df['EMA9'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        df['Signal'] = np.where(df['EMA9'] > df['EMA21'], 1, -1)
        df['Trade_Signal'] = df['Signal'].diff()
        return df

class RSIStrategy:
    def __init__(self, rsi_period=14, overbought=70, oversold=30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
    
    def apply_strategy(self, df):
        delta = df['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < self.oversold, 1, np.where(df['RSI'] > self.overbought, -1, 0))
        df['Trade_Signal'] = df['Signal'].diff()
        return df

class BollingerBandsStrategy:
    def __init__(self, window=20, std_dev=2):
        self.window = window
        self.std_dev = std_dev

    def apply_strategy(self, df):
        df['Middle_Band'] = df['close'].rolling(window=self.window).mean()  
        df['Upper_Band'] = df['Middle_Band'] + (df['close'].rolling(window=self.window).std() * self.std_dev)
        df['Lower_Band'] = df['Middle_Band'] - (df['close'].rolling(window=self.window).std() * self.std_dev)

        df['Prev_Close'] = df['close'].shift(1)

        # Fix: Buy when the price touches or goes below the lower band
        buy_condition = df['close'] <= df['Lower_Band']

        # Fix: Sell when the price is at or above the middle Bollinger Band
        sell_condition = df['close'] >= df['Middle_Band']

        df['Signal'] = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        df['Trade_Signal'] = df['Signal'].diff()

        return df

class EMACrossoverRSIStrategy:
    def __init__(self, short_window=9, long_window=21, rsi_period=14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period

    def apply_strategy(self, df):
        df['EMA_Short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_Long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        
        delta = df['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Signal'] = np.where((df['EMA_Short'] > df['EMA_Long']) & (df['RSI'] > 50), 1, 
                                np.where((df['EMA_Short'] < df['EMA_Long']) & (df['RSI'] < 50), -1, 0))
        df['Trade_Signal'] = df['Signal'].diff()
        return df

class BollingerBandsRSIStrategy:
    def __init__(self, window=20, std_dev=2, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.window = window
        self.std_dev = std_dev
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def apply_strategy(self, df):
        df['Middle_Band'] = df['close'].rolling(window=self.window).mean()
        df['Upper_Band'] = df['Middle_Band'] + (df['close'].rolling(window=self.window).std() * self.std_dev)
        df['Lower_Band'] = df['Middle_Band'] - (df['close'].rolling(window=self.window).std() * self.std_dev)
        
        delta = df['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Signal'] = np.where((df['close'] <= df['Lower_Band']) & (df['RSI'] < self.rsi_oversold), 1, 
                                np.where((df['close'] >= df['Middle_Band']), -1, 0))
        df['Trade_Signal'] = df['Signal'].diff()
        return df

class DonchianBreakoutStrategy:
    def __init__(self, window=20):
        self.window = window

    def apply_strategy(self, df):
        df['Upper_Channel'] = df['high'].rolling(window=self.window).max()
        df['Lower_Channel'] = df['low'].rolling(window=self.window).min()
        df['Middle_Channel'] = (df['Upper_Channel'] + df['Lower_Channel']) / 2
        
        df['Prev_Close'] = df['close'].shift(1)
        df['Prev_Volume'] = df['volume'].shift(1)
        avg_volume = df['volume'].rolling(window=self.window).mean()

        df['Signal'] = np.where((df['Prev_Close'] < df['Upper_Channel']) & (df['close'] > df['Upper_Channel']) & (df['volume'] > avg_volume), 1, 
                                np.where((df['Prev_Close'] > df['Lower_Channel']) & (df['close'] < df['Lower_Channel']) & (df['volume'] > avg_volume), -1, 0))
        df['Trade_Signal'] = df['Signal'].diff()
        return df

class TradingBot:
    def __init__(self, strategy, symbol="BTCUSDT", initial_balance=1000):
        self.strategy = strategy
        self.symbol = symbol
        self.base_asset = ''.join([c for c in symbol if not c.isdigit()])[:-4]  # Extract base asset (e.g., BTC, ETH)
        self.balance = initial_balance  # Start with cash
        self.position = 0  # Start with no crypto
        self.trades = 0
        self.first_trade = True  # Ensures first trade is always a buy
        self.buy_price = None  # Track the last buy price

    def run_backtest(self, df):
        df = self.strategy.apply_strategy(df)
        print(f"Initial Balance: ${self.balance:.2f}\n")
        print(f"Base asset: {self.symbol}\n")

        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            
            if df['Trade_Signal'].iloc[i] == 2 and self.first_trade:  # First trade must be a BUY
                self.position = self.balance / price  # Buy all with available balance
                self.balance = 0
                self.first_trade = False
                self.trades += 1
                self.buy_price = price  # Store buy price
                print(f"BUY at {price:.2f}, Amount: {self.position:.6f} {self.base_asset}")

            elif df['Trade_Signal'].iloc[i] == 2 and self.balance > 0:  # Buy only if we have cash
                self.position = self.balance / price
                self.balance = 0
                self.trades += 1
                self.buy_price = price  # Store buy price
                print(f"BUY at {price:.2f}, Amount: {self.position:.6f} {self.base_asset}")

            elif df['Trade_Signal'].iloc[i] == -2 and self.position > 0 and price > self.buy_price:  # Sell only if we hold crypto
                self.balance = self.position * price
                print(f"SELL at {price:.2f}, Amount: {self.position:.6f} {self.base_asset}, Balance: {self.balance}")
                self.position = 0
                self.trades += 1

        final_balance = self.balance + (self.position * df['close'].iloc[-1])
        print(f"\nTotal Trades Executed: {self.trades}")
        print(f"Final Balance after backtest: ${final_balance:.2f}")
        return final_balance



if __name__ == "__main__":
    fetcher = BinanceDataFetcher(symbol="ETHUSDT", days=365)
    df = fetcher.fetch_data()
    strategy = EMACrossoverStrategy()
    bot = TradingBot(strategy, symbol="ETHUSDT")
    bot.run_backtest(df)
