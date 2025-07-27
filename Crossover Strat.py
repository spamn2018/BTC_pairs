import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os

# Kraken valid pair codes (BASE/QUOTE)
# These are ETH/BTC, XRP/BTC, SOL/BTC — we will invert to get BTC/XXX
PAIR_MAP = {
    "BTC/ETH": "XETHXXBT",
    "BTC/XRP": "XXRPXXBT",
    "BTC/SOL": "SOLXBT"
}

INTERVAL = 86400  # minutes
DAYS_BACK = 1000
FEE_RATE = 0.0025

def moving_average_strategy(df):
    df['MA100'] = df['close'].rolling(window=100).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    df['signal'] = 0
    df.loc[(df['MA100'] > df['MA200']) & (df['MA100'].shift(1) <= df['MA200'].shift(1)), 'signal'] = 1
    df.loc[(df['MA100'] < df['MA200']) & (df['MA100'].shift(1) >= df['MA200'].shift(1)), 'signal'] = -1

    entry_price = None
    returns = []
    trades = []

    for i in range(len(df)):
        signal = df.iloc[i]['signal']
        price = df.iloc[i]['close']
        time = df.iloc[i]['time']

        if signal == 1 and entry_price is None:
            entry_price = price * (1 + FEE_RATE)
            trades.append((time, 'BUY', price))
        elif signal == -1 and entry_price is not None:
            exit_price = price * (1 - FEE_RATE)
            profit = (exit_price - entry_price) / entry_price
            returns.append(profit)
            trades.append((time, 'SELL', price))
            entry_price = None

    total_return = np.prod([1 + r for r in returns]) - 1 if returns else 0
    win_rate = np.mean([r > 0 for r in returns]) if returns else 0

    return {
        'total_return_pct': round(total_return * 100, 2),
        'num_trades': len(returns),
        'win_rate_pct': round(win_rate * 100, 2),
        'trades': trades,
        'df': df
    }

async def fetch_ohlc(pair_code, session):
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp())
    url = f'https://api.kraken.com/0/public/OHLC'
    params = {'pair': pair_code, 'interval': INTERVAL, 'since': since}
    async with session.get(url, params=params) as response:
        data = await response.json()
        if data['error']:
            raise Exception(f"Kraken API error: {data['error']}")
        key = list(data['result'].keys())[0]
        raw = data['result'][key]
        df = pd.DataFrame(raw, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        return df

async def run_analysis():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"Running analysis at {timestamp}")
    print(f"{'='*50}")
    
    async with aiohttp.ClientSession() as session:
        for label, kraken_pair in PAIR_MAP.items():
            try:
                # Get fresh data from Kraken
                fresh_df = await fetch_ohlc(kraken_pair, session)
                fresh_df['close'] = 1.0 / fresh_df['close']  # Invert price
                
                safe_name = label.replace("/", "_")
                csv_file = f"{safe_name}_15min_ohlc.csv"
                
                # Load existing data if it exists
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file)
                    existing_df['time'] = pd.to_datetime(existing_df['time'])
                    
                    # Merge with fresh data, avoiding duplicates
                    combined_df = pd.concat([existing_df, fresh_df]).drop_duplicates(subset=['time'], keep='last')
                    combined_df = combined_df.sort_values('time').reset_index(drop=True)
                    print(f"Loaded {len(existing_df)} existing records, added {len(fresh_df)} fresh records")
                else:
                    combined_df = fresh_df
                    print(f"Created new dataset with {len(fresh_df)} records")

                # Run backtest on accumulated data
                result = moving_average_strategy(combined_df.copy())

                # Save updated dataset
                combined_df.to_csv(csv_file, index=False)
                pd.DataFrame(result['trades'], columns=['Time', 'Action', 'Price']).to_csv(f"{safe_name}_trades.csv", index=False)

                # Print results
                print(f"\n=== {label} ===")
                print(f"Dataset size: {len(combined_df)} records")
                print(f"Total Return: {result['total_return_pct']}%")
                print(f"Number of Trades: {result['num_trades']}")
                print(f"Win Rate: {result['win_rate_pct']}%")

            except Exception as e:
                print(f"Error processing {label}: {e}")

async def main():
    print("Starting crypto trading analysis bot...")
    print("Running every 15 minutes. Press Ctrl+C to stop.")
    
    while True:
        try:
            await run_analysis()
            print(f"\nNext run in 15 minutes...")
            await asyncio.sleep(15 * 60)  # Wait 15 minutes
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Retrying in 15 minutes...")
            await asyncio.sleep(15 * 60)

if __name__ == "__main__":
    asyncio.run(run_analysis())
