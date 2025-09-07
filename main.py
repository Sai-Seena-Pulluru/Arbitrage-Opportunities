import os, math, collections
from collections import deque
from datetime import datetime, timedelta, date

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry

import pandas as pd, numpy as np, requests
from binance.client import Client

import matplotlib.pyplot as plt, seaborn as sns

symbol_var = None
interval_var = None
base_var = None
quote_var = None
max_path_length_var = None
capital_var = None
start_cal = None
end_cal = None
type_var = None
selected_base = None
selected_quote = None
pairs_df = pd.read_csv('pairs.csv')
pairs_dict = {row['Coinbase_ID']: row['Binance_ID'] for _, row in pairs_df.iterrows()}
symbol_display_list = list(pairs_dict.keys())
interval_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
intervals = list(interval_seconds.keys())
binance_client = Client()
def get_coinbase_candles(symbol, start_time, end_time, granularity):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {"start": start_time.isoformat(), "end": end_time.isoformat(), "granularity": granularity}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Coinbase API Error: {response.status_code} - {response.text}")
    return response.json()
def validate_inputs():
    if type_var.get() == "direct":
        if not symbol_var.get() or not interval_var.get() or not capital_var.get():
            messagebox.showwarning("Missing Input", "Please fill all required fields.")
            return False
    else:
        if not base_var.get() or not quote_var.get() or not interval_var.get() or not max_path_length_var.get() or not capital_var.get():
            messagebox.showwarning("Missing Input", "Please fill all required fields.")
            return False
    max_row = (1024 * 1024) - 1
    max_pair = len(pairs_df)
    interval = interval_var.get()
    if type_var.get() == "direct":
        if interval == '1m': max_days = math.floor(max_row / (60 * 24))
        elif interval == '5m': max_days = math.floor(max_row / ((60 * 24) / 5))
        elif interval == '15m': max_days = math.floor(max_row / ((60 * 24) / 15))
        elif interval == '1h': max_days = math.floor(max_row / 24)
    else:
        if interval == '1m': max_days = math.floor(max_row / (60 * 24 * max_pair))
        elif interval == '5m': max_days = math.floor(max_row / (((60 * 24) / 5) * max_pair))
        elif interval == '15m': max_days = math.floor(max_row / (((60 * 24) / 15) * max_pair))
        elif interval == '1h': max_days = math.floor(max_row / (24 * max_pair))
    start_dt = start_cal.get_date()
    end_dt = end_cal.get_date()
    selected_days = (end_dt - start_dt).days + 1
    if selected_days > max_days:
        messagebox.showwarning("Date Range Exceeded", f"Maximum allowed days: {max_days}. You selected {selected_days} days.")
        return False
    return True
def ensure_folder_path():
    base_dir = os.path.join(os.getcwd(), "kline-data")
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    type_folder = "direct" if type_var.get() == "direct" else "indirect"
    type_dir = os.path.join(base_dir, type_folder)
    if not os.path.exists(type_dir): os.makedirs(type_dir)
    today_folder = datetime.today().strftime("%Y-%m-%d")
    final_dir = os.path.join(type_dir, today_folder)
    if not os.path.exists(final_dir): os.makedirs(final_dir)
    if type_var.get() == "indirect":
        start_dt = start_cal.get_date()
        end_dt = end_cal.get_date()
        interval_selected = interval_var.get()
        base_selected = base_var.get()
        quote_selected = quote_var.get()
        run_folder_name = f"{base_selected}{quote_selected}_{interval_selected}_{start_dt}_to_{end_dt}"
        final_dir = os.path.join(final_dir, run_folder_name)
        if not os.path.exists(final_dir): os.makedirs(final_dir)
        else:
            for filename in os.listdir(final_dir):
                file_path = os.path.join(final_dir, filename)
                if os.path.isfile(file_path): os.remove(file_path)
    return final_dir
def detect_direct_arbitrage(merged_df, capital_start):
    arb_records = []
    for _, row in merged_df.iterrows():
        binance_close = float(row['binance_close'])
        coinbase_close = float(row['coinbase_close'])
        if binance_close < coinbase_close:
            base_bought = capital_start / binance_close
            final_capital = base_bought * coinbase_close
            path = "buy binance -> sell coinbase"
        elif coinbase_close < binance_close:
            base_bought = capital_start / coinbase_close
            final_capital = base_bought * binance_close
            path = "buy coinbase -> sell binance"
        else:
            continue
        net_profit = final_capital - capital_start
        net_profit_pct = (net_profit / capital_start) * 100 if capital_start else 0
        rec = row.to_dict()
        rec.update({"path": path, "capital_start": capital_start, "capital_end": final_capital, "net_profit": net_profit, "net_profit_percentage": f"{net_profit_pct:.2f}"})
        capital_start = final_capital
        arb_records.append(rec)
    return arb_records
def detect_indirect_arbitrage(save_folder, start_crypto, end_crypto, max_path_length, capital_start):
    all_arb_records = []
    all_csv_files = [os.path.join(save_folder, f) for f in os.listdir(save_folder) if f.endswith(".csv") and f not in ["arbitrage_opportunities.csv"]]
    combined_df = pd.DataFrame()
    for f in all_csv_files:
        temp_df = pd.read_csv(f)
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        combined_df = combined_df[combined_df["best_exchange"] != "no_arbitrage"]
    pairs_df = pd.read_csv("pairs.csv")
    all_cryptos_set = set(pairs_df['Base']).union(set(pairs_df['Quote']))
    rows = []
    for crypto in all_cryptos_set:
        related_pairs = pairs_df[(pairs_df['Base'] == crypto) | (pairs_df['Quote'] == crypto)]
        for pair in related_pairs['Coinbase_ID']: rows.append({"Crypto": crypto, "Possibilities": pair})
    crypto_df = pd.DataFrame(rows)
    enriched_rows = []
    for _, row in crypto_df.iterrows():
        crypto = row["Crypto"]
        pair = row["Possibilities"]
        pair_data = combined_df[combined_df["pair"] == pair]
        if not pair_data.empty:
            max_row = pair_data.loc[pair_data["profit_percentage"].idxmax()].to_dict()
            max_row.update({"Crypto": crypto, "Possibilities": pair})
            enriched_rows.append(max_row)
    enriched_df = pd.DataFrame(enriched_rows)
    df = enriched_df
    graph = collections.defaultdict(list)
    for _, row in df.iterrows():
        base = row['base']
        quote = row['quote']
        edge_data = {'pair': row['pair'], 'exchange': row['best_exchange'], 'profit_percentage': row['profit_percentage']}
        graph[base].append((quote, edge_data))
        graph[quote].append((base, edge_data))
    queue = collections.deque()
    queue.append((start_crypto, [ {'crypto': start_crypto} ], 0, 0.0))
    best_path = None
    while queue:
        node, path, depth, total_profit = queue.popleft()
        if depth >= max_path_length: continue
        for neighbor, edge in graph[node]:
            if any(step['crypto'] == neighbor for step in path): continue
            new_path = path + [{'crypto': neighbor, 'pair': edge['pair'], 'exchange': edge['exchange']}]
            new_profit = total_profit + edge['profit_percentage']
            if neighbor == end_crypto and (best_path is None or new_profit > best_path['total_profit']):
                best_path = {'path': new_path, 'total_profit': new_profit}
            queue.append((neighbor, new_path, depth + 1, new_profit))
    if best_path:
        capital = capital_start
        path_list = best_path['path']
        for i in range(1, len(path_list)):
            prev = path_list[i - 1]
            step = path_list[i]
            row_data = df[df['pair'] == step['pair']]
            if row_data.empty: continue
            row_data = row_data.iloc[0]
            profit_pct = float(row_data['profit_percentage']) / 100
            capital_end = capital * (1 + profit_pct)
            net_profit = capital_end - capital
            net_profit_pct = (net_profit / capital) * 100 if capital else 0
            rec = {'open_time': row_data['open_time'], 'pair': row_data['pair'], 'base': row_data['base'], 'quote': row_data['quote'],
                   'binance_open': row_data['binance_open'], 'binance_high': row_data['binance_high'], 'binance_low': row_data['binance_low'], 'binance_close': row_data['binance_close'], 'binance_volume': row_data['binance_volume'],
                   'coinbase_open': row_data['coinbase_open'], 'coinbase_high': row_data['coinbase_high'], 'coinbase_low': row_data['coinbase_low'], 'coinbase_close': row_data['coinbase_close'], 'coinbase_volume': row_data['coinbase_volume'],
                   'best_exchange': row_data['best_exchange'], 'path_index': i, 'path': " -> ".join([p['crypto'] for p in path_list[:i+1]]),
                   'capital_start': capital, 'capital_end': capital_end, 'net_profit': net_profit, 'net_profit_percentage': f"{net_profit_pct:.2f}"}
            all_arb_records.append(rec)
            capital = capital_end
    return all_arb_records
def download_data():
    if not validate_inputs(): return
    capital_start = float(capital_var.get())
    start_dt = datetime.combine(start_cal.get_date(), datetime.min.time())
    end_dt = datetime.combine(end_cal.get_date(), datetime.max.time())
    interval_selected = interval_var.get()
    if type_var.get() == "direct":
        symbol_selected = symbol_var.get()
    else:
        base_selected = base_var.get()
        quote_selected = quote_var.get()
        max_path_length = int(max_path_length_var.get())
    save_folder = ensure_folder_path()
    interval_sec = interval_seconds[interval_selected]
    if type_var.get() == "direct":
        selected_display_list = [symbol_var.get()]
    else:
        global selected_base, selected_quote
        selected_base = base_var.get()
        selected_quote = quote_var.get()
        selected_display_list = list(pairs_dict.keys())
    total_symbols = len(selected_display_list)
    all_results = []
    for s_idx, selected_display in enumerate(selected_display_list, 1):
        ids = pairs_dict[selected_display]
        coinbase_symbol = selected_display
        binance_symbol = pairs_dict[coinbase_symbol]
        try:
            binance_df = pd.DataFrame()
            chunk_duration = timedelta(seconds=interval_sec * 1000)
            chunks = []
            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + chunk_duration, end_dt)
                chunks.append((current_start, current_end))
                current_start = current_end
            for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                klines = binance_client.get_historical_klines(symbol=binance_symbol, interval=interval_selected,
                                                              start_str=chunk_start.strftime('%d %b, %Y %H:%M:%S'),
                                                              end_str=chunk_end.strftime('%d %b, %Y %H:%M:%S'))
                if klines:
                    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                                       'Close Time', 'Quote Volume', 'Number of Trades', 'Taker Buy Base', 'Taker Buy Quote', 'Ignore'])
                    df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
                    binance_df = pd.concat([binance_df, df], ignore_index=True)
            binance_df.sort_values("Open Time", inplace=True)
            binance_df.drop_duplicates(subset="Open Time", keep="first", inplace=True)
            binance_df.set_index("Open Time", inplace=True)
        except Exception as e: status_label.config(text=f"Binance error: {e}"); continue
        try:
            coinbase_df = pd.DataFrame()
            chunk_duration = timedelta(seconds=interval_sec * 300)
            chunks = []
            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + chunk_duration - timedelta(seconds=interval_sec), end_dt)
                chunks.append((current_start, current_end))
                current_start = current_end + timedelta(seconds=interval_sec)
            for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                candles = get_coinbase_candles(coinbase_symbol, chunk_start, chunk_end, interval_sec)
                if candles:
                    df = pd.DataFrame(candles, columns=["time", "low", "high", "open", "close", "volume"])
                    df["time"] = pd.to_datetime(df["time"], unit='s')
                    df.rename(columns={"time": "Open Time", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
                    df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]]
                    coinbase_df = pd.concat([coinbase_df, df], ignore_index=True)
            coinbase_df.sort_values("Open Time", inplace=True)
            coinbase_df.drop_duplicates(subset="Open Time", keep="first", inplace=True)
            coinbase_df.set_index("Open Time", inplace=True)
        except Exception as e: status_label.config(text=f"Coinbase error: {e}"); continue
        coinbase_df_real = coinbase_df.copy()
        full_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{interval_sec}s')
        coinbase_df = coinbase_df.reindex(full_range)
        first_valid = coinbase_df_real.iloc[0] if not coinbase_df_real.empty else None
        for ts in coinbase_df.index:
            if pd.isna(coinbase_df.loc[ts, "Close"]):
                if ts < coinbase_df_real.index[0]:
                    if first_valid is not None: coinbase_df.loc[ts] = {"Open": first_valid["Open"], "High": first_valid["Open"], "Low": first_valid["Open"], "Close": first_valid["Open"], "Volume": 0}
                else:
                    prev_close = coinbase_df["Close"].loc[:ts].ffill().iloc[-1] if not coinbase_df["Close"].loc[:ts].ffill().empty else None
                    if pd.notna(prev_close): coinbase_df.loc[ts] = {"Open": prev_close, "High": prev_close, "Low": prev_close, "Close": prev_close, "Volume": 0}
        coinbase_df.update(coinbase_df_real)
        coinbase_df.reset_index(inplace=True)
        coinbase_df.rename(columns={"index": "Open Time"}, inplace=True)
        binance_df.reset_index(inplace=True)
        merged_df = pd.merge(binance_df, coinbase_df, on="Open Time", how="outer", suffixes=("_binance", "_coinbase"))
        merged_df.sort_values("Open Time", inplace=True)
        merged_df = merged_df[["Open Time", "Open_binance", "High_binance", "Low_binance", "Close_binance", "Volume_binance", "Open_coinbase", "High_coinbase", "Low_coinbase", "Close_coinbase", "Volume_coinbase"]]
        merged_df.columns = ["open_time", "binance_open", "binance_high", "binance_low", "binance_close", "binance_volume", "coinbase_open", "coinbase_high", "coinbase_low", "coinbase_close", "coinbase_volume"]
        merged_df.insert(1, "pair", coinbase_symbol)
        base, quote = coinbase_symbol.split("-")
        merged_df.insert(2, "base", base)
        merged_df.insert(3, "quote", quote)
        def max_close_decimal_places(series):
            arr_str = series.astype(str).to_numpy(dtype="U")
            decimals = np.char.find(arr_str, '.')
            lengths = np.where(decimals >= 0, np.char.str_len(arr_str) - decimals - 1, 0)
            return lengths.max()
        max_binance = max_close_decimal_places(merged_df['binance_close'])
        max_coinbase = max_close_decimal_places(merged_df['coinbase_close'])
        greatest_precision = max(max_binance, max_coinbase)
        def calc_best_exchange(row):
            binance_close = float(row['binance_close'])
            coinbase_close = float(row['coinbase_close'])
            if binance_close == coinbase_close: return "no_arbitrage"
            return "binance" if binance_close > coinbase_close else "coinbase"
        def calc_profit(row):
            binance_close = float(row['binance_close'])
            coinbase_close = float(row['coinbase_close'])
            if binance_close == coinbase_close: return 0
            return f"{abs(binance_close - coinbase_close):.{greatest_precision}f}"
        def calc_profit_percentage(row):
            binance_close = float(row['binance_close'])
            coinbase_close = float(row['coinbase_close'])
            if binance_close == coinbase_close: return 0
            return ((abs(binance_close - coinbase_close) / (coinbase_close if binance_close > coinbase_close else binance_close)) * 100)
        merged_df["best_exchange"] = merged_df.apply(calc_best_exchange, axis=1)
        merged_df["profit"] = merged_df.apply(calc_profit, axis=1)
        merged_df["profit_percentage"] = merged_df.apply(calc_profit_percentage, axis=1).apply(lambda x: f"{x:.2f}")
        merged_filename = f"{binance_symbol}_{interval_selected}_{start_dt.date()}_to_{end_dt.date()}.csv"
        merged_filepath = os.path.join(save_folder, merged_filename)
        for col in ["coinbase_open", "coinbase_high", "coinbase_low", "coinbase_close"]:
            if col in merged_df.columns: merged_df[col] = merged_df[col].apply(lambda x: ('%.15f' % x).rstrip('0').rstrip('.') if pd.notna(x) else "")
        merged_df.to_csv(merged_filepath, index=False)
        all_results.append(merged_df)
    capital_start = float(capital_var.get().strip())
    arb_records = []
    if type_var.get() == "direct":
        for merged_df in all_results:
            arb_records.extend(detect_direct_arbitrage(merged_df, capital_start))
    else:
        arb_records.extend(detect_indirect_arbitrage(save_folder, start_crypto=base_var.get(), end_crypto=quote_var.get(), max_path_length=int(max_path_length_var.get()), capital_start=capital_start))
    if arb_records:
        arb_df = pd.DataFrame(arb_records)
        arb_file = os.path.join(save_folder, "arbitrage_opportunities.csv")
        arb_df.to_csv(arb_file, index=False)
        visualize_results(arb_file, save_folder)
    progress_bar["value"] = 100
def toggle_type():
    global symbol_var, symbol_menu
    global interval_var, interval_menu
    global base_var, base_menu
    global quote_var, quote_menu
    global max_path_length_var
    global capital_var
    capital_var.set("")
    for widget in dynamic_frame.winfo_children():
        widget.destroy()
    if type_var.get() == "direct":
        tk.Label(dynamic_frame, text="Base-Quote:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        symbol_var = tk.StringVar()
        symbol_menu = ttk.Combobox(dynamic_frame, textvariable=symbol_var, values=symbol_display_list, state="readonly")
        symbol_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tk.Label(dynamic_frame, text="Interval:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        interval_var = tk.StringVar()
        interval_menu = ttk.Combobox(dynamic_frame, textvariable=interval_var, values=intervals, state="readonly")
        interval_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    else:
        unique_assets = sorted(set([s.split("-")[0] for s in symbol_display_list] + [s.split("-")[1] for s in symbol_display_list]))
        tk.Label(dynamic_frame, text="Base:").grid(row=0, column=0, padx=10, pady=5)
        base_var = tk.StringVar()
        base_menu = ttk.Combobox(dynamic_frame, textvariable=base_var, values=unique_assets, state="readonly")
        base_menu.grid(row=0, column=1)
        tk.Label(dynamic_frame, text="Quote:").grid(row=0, column=2, padx=10, pady=5)
        quote_var = tk.StringVar()
        quote_menu = ttk.Combobox(dynamic_frame, textvariable=quote_var, values=unique_assets, state="readonly")
        quote_menu.grid(row=0, column=3)
        tk.Label(dynamic_frame, text="Interval:").grid(row=1, column=0, padx=10, pady=5)
        interval_var = tk.StringVar()
        interval_menu = ttk.Combobox(dynamic_frame, textvariable=interval_var, values=intervals, state="readonly")
        interval_menu.grid(row=1, column=1)
        tk.Label(dynamic_frame, text="Max Path:").grid(row=1, column=2, padx=10, pady=5)
        max_path_length_var = tk.StringVar()
        def validate_max_path(P):
            if P == "":
                return True
            if P.isdigit() and 1 <= int(P) <= 10:
                return True
            return False
        max_path_length_entry = tk.Entry(dynamic_frame, textvariable=max_path_length_var, width=23, validate="key", validatecommand=(root.register(validate_max_path), '%P'))
        max_path_length_entry.grid(row=1, column=3, padx=5, pady=5)
root = tk.Tk()
root.title("Arbitrage Detector")
window_width = 600
window_height = 450
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coord = int((screen_width/2) - (window_width/2))
y_coord = int((screen_height/2) - (window_height/2))
root.geometry(f"{window_width}x{window_height}+{x_coord}+{y_coord}")
root.resizable(False, False)
type_frame = tk.LabelFrame(root, text="Arbitrage Type", padx=10, pady=5)
type_frame.pack(fill="x", padx=10, pady=8)
type_var = tk.StringVar(value="direct")
tk.Radiobutton(type_frame, text="Direct", variable=type_var, value="direct", command=lambda: toggle_type()).pack(side="left", padx=10)
tk.Radiobutton(type_frame, text="Indirect", variable=type_var, value="indirect", command=lambda: toggle_type()).pack(side="left", padx=10)
dynamic_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=5)
dynamic_frame.pack(fill="x", padx=10, pady=8)
date_frame = tk.LabelFrame(root, text="Date Range", padx=10, pady=5)
date_frame.pack(fill="x", padx=10, pady=8)
tk.Label(date_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
start_cal = DateEntry(date_frame, date_pattern='yyyy-mm-dd', maxdate=date.today() - timedelta(days=1))
start_cal.set_date(date.today() - timedelta(days=7))
start_cal.grid(row=0, column=1, padx=5, pady=5, sticky="w")
tk.Label(date_frame, text="End Date:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
end_cal = DateEntry(date_frame, date_pattern='yyyy-mm-dd', maxdate=date.today() - timedelta(days=1))
end_cal.set_date(date.today() - timedelta(days=1))
end_cal.grid(row=0, column=3, padx=5, pady=5, sticky="w")
capital_frame = tk.LabelFrame(root, text="Capital", padx=10, pady=5)
capital_frame.pack(fill="x", padx=10, pady=8)
capital_var = tk.StringVar()
def validate_float(P): return P == "" or P.replace(".", "", 1).isdigit()
vcmd = (root.register(validate_float), '%P')
tk.Label(capital_frame, text="Starting With:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
capital_entry = tk.Entry(capital_frame, textvariable=capital_var, validate="key", validatecommand=vcmd, width=15)
capital_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
tk.Label(capital_frame, text="(in base currency)").grid(row=0, column=2, padx=5, pady=5, sticky="w")
progress_frame = tk.Frame(root)
progress_frame.pack(fill="x", padx=10, pady=2)
progress_bar = ttk.Progressbar(progress_frame, length=500, mode='determinate')
progress_bar.pack(pady=0)
progress_label = tk.Label(progress_frame, text="", fg="blue")
progress_label.pack(pady=0)
status_label = tk.Label(progress_frame, text="", fg="green")
status_label.pack(pady=0)
button_frame = tk.Frame(root)
button_frame.pack(pady=2)
tk.Button(button_frame, text="Download & Detect", command=download_data, width=20).pack()
toggle_type()
def visualize_results(csv_file, save_dir):
    df = pd.read_csv(csv_file)
    direct_df = df[df.get("path", "").str.contains("binance|coinbase", na=False)]
    indirect_df = df[~df.index.isin(direct_df.index)]
    if not direct_df.empty:
        plt.figure(figsize=(8,5))
        direct_df.groupby("best_exchange")["net_profit"].sum().plot(kind="bar", figsize=(6,4), title="Profitable Exchange")
        plt.ylabel("Total Profit")
        plt.xlabel("Exchange")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "direct_profitable_exchange.png"))
        plt.show()
        direct_df['open_time'] = pd.to_datetime(direct_df['open_time'])
        direct_df.set_index('open_time', inplace=True)
        hourly_df = direct_df['capital_end'].resample('h').last().dropna()
        plt.figure(figsize=(8,5))
        plt.plot(hourly_df.index, hourly_df.values)
        plt.title("Capital Growth")
        plt.xlabel("Time")
        plt.ylabel("Capital")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "direct_capital_growth.png"))
        plt.show()
    if not indirect_df.empty:
        path_stats = indirect_df.groupby("path")["net_profit"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8,5))
        sns.barplot(y=path_stats.index[:10], x=path_stats.values[:10])
        plt.title("Profitable Paths")
        plt.xlabel("Avg Profit")
        plt.ylabel("Path")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "indirect_profitable_paths.png"))
        plt.show()
        top_pairs_profit = indirect_df.groupby("pair")["net_profit_percentage"].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8,5))
        sns.barplot(x=top_pairs_profit.values, y=top_pairs_profit.index)
        plt.title("Profitable Pairs")
        plt.xlabel("Profit Percentage")
        plt.ylabel("Pair")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "indirect_profitable_pairs.png"))
        plt.show()
    total_profit = df["net_profit"].sum()
    mean_profit = df["net_profit"].mean()
    std_profit = df["net_profit"].std()
    sharpe_ratio = mean_profit / std_profit if std_profit > 0 else np.nan
    print(f"\nTotal Profit: {total_profit:.2f}")
    print(f"Avg Profit per Trade: {mean_profit:.2f}")
    print(f"Std Dev of Profit: {std_profit:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
root.mainloop()