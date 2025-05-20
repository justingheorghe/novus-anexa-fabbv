import pandas as pd
import numpy as np

def calculate_historical_metrics(file_path, asset_name,
                                 date_col_name='Date',
                                 price_col_name='Close',
                                 date_format=None,
                                 skiprows_config=None,
                                 start_date_str=None,
                                 end_date_str=None):
    """
    Calculează randamentul anual așteptat și volatilitatea anuală
    pe baza datelor istorice de preț dintr-un fișier CSV, filtrând pe un interval de date.

    Args:
        file_path (str): Calea către fișierul CSV.
        asset_name (str): Numele activului.
        date_col_name (str): Numele coloanei cu datele calendaristice.
        price_col_name (str): Numele coloanei cu prețurile de închidere.
        date_format (str, optional): Formatul datei dacă trebuie specificat pentru pd.to_datetime.
        skiprows_config (list, optional): Listă de numere de rânduri de sărit (0-indexat) după rândul de antet.
        start_date_str (str, optional): Data de început pentru filtrare (ex: 'YYYY-MM-DD').
        end_date_str (str, optional): Data de sfârșit pentru filtrare (ex: 'YYYY-MM-DD').

    Returns:
        dict: Un dicționar cu randamentul așteptat și volatilitatea, sau None dacă apare o eroare.
    """
    try:
        df = None
        try:
            # Load CSV, use first row as header, skip specified additional rows
            df = pd.read_csv(file_path, header=0, skiprows=skiprows_config)
        except ValueError as ve:
             print(f"EROARE ({asset_name}): Eroare la citirea CSV '{file_path}' (posibil fișier gol sau format incorect): {ve}.")
             return None
        except Exception as e:
            print(f"EROARE ({asset_name}): Eroare la încărcarea inițială a CSV '{file_path}': {e}.")
            return None

        if df.empty:
            print(f"EROARE ({asset_name}): DataFrame-ul este gol după încărcarea fișierului '{file_path}'.")
            return None

        # Identifică coloana de dată efectivă
        actual_date_col = date_col_name
        if actual_date_col not in df.columns:
            print(f"INFO ({asset_name}): Coloana de dată specificată '{actual_date_col}' nu a fost găsită. Se încearcă prima coloană: '{df.columns[0]}'.")
            actual_date_col = df.columns[0] # Presupunem că prima coloană este data dacă cea specificată nu există
            if actual_date_col not in df.columns: # Verificare suplimentară, deși df.columns[0] ar trebui să existe dacă df nu e gol
                 print(f"EROARE ({asset_name}): Nu s-a putut determina o coloană de dată validă. Coloane disponibile: {df.columns.tolist()}")
                 return None

        # Verificarea existenței coloanei de preț
        actual_price_col = price_col_name
        if actual_price_col not in df.columns:
            print(f"INFO ({asset_name}): Coloana de preț specificată '{actual_price_col}' nu a fost găsită. Se caută alternative.")
            potential_price_cols = [col for col in df.columns if col.lower() in ['close', 'price', 'adj close', 'last']]
            if asset_name == "Vestas Wind Systems" and 'Close' in df.columns: # Specific for Vestas
                 actual_price_col = 'Close'
            elif potential_price_cols:
                actual_price_col = potential_price_cols[0]
            else:
                print(f"EROARE ({asset_name}): Coloana de preț '{price_col_name}' sau alternative comune nu au fost găsite. Coloane disponibile: {df.columns.tolist()}")
                return None
            print(f"INFO ({asset_name}): Se utilizează coloana de preț '{actual_price_col}'.")

        # Conversia coloanei de dată și setarea ca index
        try:
            if date_format:
                df[actual_date_col] = pd.to_datetime(df[actual_date_col], format=date_format, errors='raise')
            else:
                df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors='raise') # Încearcă inferența
            print(f"INFO ({asset_name}): Coloana '{actual_date_col}' convertită în DatetimeIndex cu formatul specificat/inferat.")
        except Exception as e_conv_date:
            print(f"INFO ({asset_name}): Conversia inițială a coloanei '{actual_date_col}' în DatetimeIndex a eșuat: {e_conv_date}. Se încearcă formate comune.")
            common_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%d-%b-%Y", "%Y/%m/%d", "%m/%d/%y"]
            parsed_date_successfully = False
            date_series_as_strings = df[actual_date_col].astype(str)

            for fmt in common_formats:
                if date_format and fmt == date_format: continue
                try:
                    df[actual_date_col] = pd.to_datetime(date_series_as_strings, format=fmt, errors='raise')
                    parsed_date_successfully = True
                    print(f"INFO ({asset_name}): Coloana '{actual_date_col}' a fost parsat cu succes ca dată cu formatul '{fmt}'.")
                    break
                except (ValueError, TypeError):
                    continue
            if not parsed_date_successfully:
                print(f"EROARE ({asset_name}): Eșec la parsarea coloanei '{actual_date_col}' ca dată cu toate formatele încercate. Prima valoare: '{df[actual_date_col].iloc[0] if not df.empty else 'N/A'}'")
                return None

        df = df.set_index(actual_date_col)
        df = df[~df.index.isnull()] # Elimină rândurile unde indexul (data) nu a putut fi parsat (este NaT)
        if df.empty:
            print(f"EROARE ({asset_name}): Nu există date după conversia indexului la DatetimeIndex și eliminarea valorilor NaT.")
            return None

        df = df.sort_index()

        # Filtrare pe intervalul de date specificat
        if start_date_str:
            try:
                start_date_dt = pd.to_datetime(start_date_str)
                df = df[df.index >= start_date_dt]
            except ValueError as e:
                print(f"EROARE ({asset_name}): Format dată de început invalid '{start_date_str}': {e}")
                return None
        if end_date_str:
            try:
                end_date_dt = pd.to_datetime(end_date_str)
                df = df[df.index <= end_date_dt]
            except ValueError as e:
                print(f"EROARE ({asset_name}): Format dată de sfârșit invalid '{end_date_str}': {e}")
                return None

        if df.empty:
            print(f"EROARE ({asset_name}): Nu există date în intervalul specificat ({start_date_str if start_date_str else 'început'} - {end_date_str if end_date_str else 'sfârșit'}).")
            return None
            
        # Asigurarea că prețul este numeric și eliminarea valorilor non-numerice/lipsă
        price_series = df[actual_price_col]
        if price_series.dtype == 'object':
            price_series_cleaned = price_series.astype(str).str.replace(r'[^\d.]', '', regex=True)
            price_series_cleaned = price_series_cleaned.replace('', np.nan)
        else:
            price_series_cleaned = price_series

        df = df.assign(**{actual_price_col: pd.to_numeric(price_series_cleaned, errors='coerce')})
        df = df.dropna(subset=[actual_price_col])

        if df.empty:
            print(f"EROARE ({asset_name}): Nu există date valide de preț după curățare.")
            return None

        log_returns_series = np.log(df[actual_price_col] / df[actual_price_col].shift(1))
        df = df.assign(LogReturns=log_returns_series)
        
        df = df.dropna(subset=['LogReturns'])

        if df.empty or len(df['LogReturns']) < 2:
            print(f"EROARE ({asset_name}): Nu sunt suficiente date de randament după procesare.")
            return None

        trading_days_per_year = 252
        mean_daily_log_return = df['LogReturns'].mean()
        expected_annual_log_return = mean_daily_log_return * trading_days_per_year
        expected_annual_effective_return = np.exp(expected_annual_log_return) - 1
        std_dev_daily_log_return = df['LogReturns'].std(ddof=0)
        annual_volatility = std_dev_daily_log_return * np.sqrt(trading_days_per_year)
        num_data_points = len(df)

        return {
            'asset_name': asset_name,
            'expected_annual_return': expected_annual_effective_return,
            'annual_volatility': annual_volatility,
            'num_daily_returns_used': num_data_points,
            'mean_daily_log_return': mean_daily_log_return,
            'std_dev_daily_log_return': std_dev_daily_log_return,
            'first_date_used': df.index.min().strftime('%Y-%m-%d'),
            'last_date_used': df.index.max().strftime('%Y-%m-%d')
        }

    except FileNotFoundError:
        print(f"EROARE ({asset_name}): Fișierul {file_path} nu a fost găsit.")
        return None
    except Exception as e:
        print(f"EROARE ({asset_name}): A apărut o eroare neașteptată la procesarea {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Calea către fișierul CSV pentru Vestas
vestas_file = "vestas_history.csv" # Actualizat la fișierul corect

print("--- Calcularea Metricilor Istorice pentru Vestas ---")

# Pentru Vestas
# Formatul datei în vestas_history.csv este 'YYYY-MM-DD' (ex: '2000-05-10')
# Coloana de dată este sub header-ul 'Price' după skip rows.
# Coloana de preț este sub header-ul 'Close' după skip rows.
# Se sar rândurile 1 și 2 (0-indexat) după header-ul de pe rândul 0.
vestas_metrics = calculate_historical_metrics(
    file_path=vestas_file,
    asset_name="Vestas Wind Systems",
    date_col_name='Price',
    price_col_name='Close',
    date_format='%Y-%m-%d', # Corectat formatul datei
    skiprows_config=[1, 2], # Skip row 2 ("Ticker...") and row 3 ("Date...") (0-indexed AFTER header taken from row 0)
    start_date_str="2015-05-18",
    end_date_str="2025-05-18"
)

# --- Calcularea Metricilor Istorice pentru Ethereum (ETH-USD) ---
eth_file = "eth-usd_3y_history.csv"
print("\n--- Calcularea Metricilor Istorice pentru Ethereum (ETH-USD) ---")

eth_metrics = calculate_historical_metrics(
    file_path=eth_file,
    asset_name="Ethereum (ETH-USD)",
    date_col_name='Price',      # In eth-usd_10Y_history.csv, the first column 'Price' holds dates
    price_col_name='Close',     # The 'Close' column holds the price
    date_format='%Y-%m-%d',     # Date format e.g., '2017-11-09'
    skiprows_config=[1, 2],     # Skip the "Ticker" info row and the "Date,,,,,," row after the header
    start_date_str=None,        # Use all available data
    end_date_str=None           # Use all available data
)

# --- Calcularea Metricilor Istorice pentru Wise (WISE.L) ---
wise_file = "wise_3y_history.csv"
print("\n--- Calcularea Metricilor Istorice pentru Wise (WISE.L) ---")

wise_metrics = calculate_historical_metrics(
    file_path=wise_file,
    asset_name="Wise (WISE.L)",
    date_col_name='Price',      # In wise_3y_history.csv, the first column 'Price' holds dates
    price_col_name='Close',     # The 'Close' column holds the price
    date_format='%Y-%m-%d',     # Date format e.g., '2022-05-10'
    skiprows_config=[1, 2],     # Skip the "Ticker" info row and the "Date,,,,,," row after the header
    start_date_str=None,        # Use all available data
    end_date_str=None           # Use all available data
)

print("\n--- Rezultate Finale ---")
if vestas_metrics:
    print(f"\nPentru {vestas_metrics['asset_name']}:")
    print(f"  Perioada analizată: {vestas_metrics['first_date_used']} până la {vestas_metrics['last_date_used']} ({vestas_metrics['num_daily_returns_used']} zile de randament)")
    print(f"  Randament Anual Așteptat (istoric, efectiv): {vestas_metrics['expected_annual_return']:.2%}")
    print(f"  Volatilitate Anuală (istorică): {vestas_metrics['annual_volatility']:.2%}")
else:
    print("\nNu s-au putut calcula metricile pentru Vestas Wind Systems.")

if eth_metrics:
    print(f"\nPentru {eth_metrics['asset_name']}:")
    print(f"  Perioada analizată: {eth_metrics['first_date_used']} până la {eth_metrics['last_date_used']} ({eth_metrics['num_daily_returns_used']} zile de randament)")
    print(f"  Randament Anual Așteptat (istoric, efectiv): {eth_metrics['expected_annual_return']:.2%}")
    print(f"  Volatilitate Anuală (istorică): {eth_metrics['annual_volatility']:.2%}")
else:
    print("\nNu s-au putut calcula metricile pentru Ethereum (ETH-USD).")

if wise_metrics:
    print(f"\nPentru {wise_metrics['asset_name']}:")
    print(f"  Perioada analizată: {wise_metrics['first_date_used']} până la {wise_metrics['last_date_used']} ({wise_metrics['num_daily_returns_used']} zile de randament)")
    print(f"  Randament Anual Așteptat (istoric, efectiv): {wise_metrics['expected_annual_return']:.2%}")
    print(f"  Volatilitate Anuală (istorică): {wise_metrics['annual_volatility']:.2%}")
else:
    print("\nNu s-au putut calcula metricile pentru Wise (WISE.L).")

