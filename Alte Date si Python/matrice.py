import pandas as pd
import numpy as np

def clean_price_data(series):
    """ Curăță o serie de prețuri (string) și o convertește în float. """
    if series.dtype == 'object':
        series = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
        series.replace('', np.nan, inplace=True) # Înlocuiește string-urile goale cu NaN
    return pd.to_numeric(series, errors='coerce')

def load_and_prepare_asset_data(file_path, asset_ticker, 
                                date_col_name='Date', 
                                price_col_name='Price', 
                                date_format=None,
                                skiprows_config=None):
    """ Încarcă, curăță și calculează randamentele zilnice logaritmice pentru un activ. """
    try:
        # Load CSV, use first row as header, skip specified additional rows
        df = pd.read_csv(file_path, header=0, skiprows=skiprows_config)

        if df.empty:
            raise ValueError(f"DataFrame-ul este gol după încărcarea {file_path} cu skiprows.")

        # Identifică coloana de dată efectivă
        actual_date_col = date_col_name
        if actual_date_col not in df.columns:
            print(f"INFO ({asset_ticker}): Coloana de dată specificată '{actual_date_col}' nu a fost găsită în {file_path}. Se încearcă prima coloană: '{df.columns[0]}'.")
            actual_date_col = df.columns[0]
            if actual_date_col not in df.columns:
                 raise ValueError(f"EROARE ({asset_ticker}): Nu s-a putut determina o coloană de dată validă în {file_path}. Coloane disponibile: {df.columns.tolist()}")
        
        # Identifică coloana de preț efectivă
        actual_price_col = price_col_name
        if actual_price_col not in df.columns:
            print(f"INFO ({asset_ticker}): Coloana de preț specificată '{actual_price_col}' nu a fost găsită în {file_path}. Se caută alternative.")
            # Prioritize known good columns for specific assets after skiprows
            if asset_ticker == "ETH" and 'Close' in df.columns: actual_price_col = 'Close'
            elif asset_ticker == "WISE" and 'Close' in df.columns: actual_price_col = 'Close'
            elif 'Price' in df.columns: actual_price_col = 'Price' # General fallback
            elif 'Close' in df.columns: actual_price_col = 'Close' # General fallback
            else:
                potential_price_cols = [col for col in df.columns if col.lower() in ['close', 'price', 'adj close', 'last']]
                if potential_price_cols: actual_price_col = potential_price_cols[0]
                else: raise ValueError(f"Coloana de preț '{price_col_name}' sau alternative nu au fost găsite în {file_path}")
            print(f"INFO ({asset_ticker}): Se utilizează coloana de preț '{actual_price_col}' pentru {file_path}.")

        # Conversie dată
        try:
            if date_format:
                df[actual_date_col] = pd.to_datetime(df[actual_date_col], format=date_format, errors='raise')
            else:
                df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors='raise')
            print(f"INFO ({asset_ticker}): Coloana '{actual_date_col}' pentru {file_path} a fost parsat ca dată cu formatul specificat/inferat.")
        except Exception as e: 
            parsed_successfully = False
            # Folosim %m/%d/%y pentru datele de tip '5/18/20'
            common_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%d-%b-%y", "%m/%d/%y"]
            date_series_as_strings = df[actual_date_col].astype(str)
            for fmt in common_formats:
                if date_format and fmt == date_format: continue # Skip if primary format already failed
                try:
                    df[actual_date_col] = pd.to_datetime(date_series_as_strings, format=fmt, errors='raise')
                    parsed_successfully = True
                    print(f"INFO ({asset_ticker}): Data pentru {file_path} (col: '{actual_date_col}') a fost parsat cu formatul '{fmt}'.")
                    break
                except (ValueError, TypeError):
                    continue
            if not parsed_successfully:
                 raise ValueError(f"EROARE ({asset_ticker}): Eșec la parsarea datei pentru {file_path} (col: '{actual_date_col}', val: '{df[actual_date_col].iloc[0] if not df.empty else 'N/A'}'). Detalii: {e}")
        
        df = df.sort_values(by=actual_date_col)
        df = df.set_index(actual_date_col)
        
        # Curățare preț și calcul randamente logaritmice
        df[asset_ticker] = clean_price_data(df[actual_price_col])
        df = df.dropna(subset=[asset_ticker])
        
        if df.empty:
            raise ValueError(f"Nu există date de preț valide pentru {asset_ticker} după curățare.")
            
        df[f'{asset_ticker}_LogReturns'] = np.log(df[asset_ticker] / df[asset_ticker].shift(1))
        
        return df[[f'{asset_ticker}_LogReturns']].copy() # Returnează doar coloana de randamente

    except FileNotFoundError:
        print(f"EROARE: Fișierul {file_path} nu a fost găsit.")
        return None
    except ValueError as ve:
        print(str(ve))
        return None
    except Exception as e:
        print(f"EROARE: A apărut o eroare neașteptată la procesarea {file_path} pentru {asset_ticker}: {e}")
        return None

# Căile către fișierele CSV
eth_file = "ETH 5Y Data - Sheet1.csv"
wise_file = "wise_3y_history.csv"

print("--- Calcularea Matricei de Corelare pentru ETH și Wise PLC ---")

# Încărcarea și pregătirea datelor pentru ETH
# Format dată pentru ETH: 'Nov 17, 2023' -> '%b %d, %Y' -- Actualizat: '5/18/20' -> '%m/%d/%y'
# Coloană preț pentru ETH: 'Price' -- Actualizat: 'Close'
eth_returns_df = load_and_prepare_asset_data(
    eth_file, 
    "ETH", 
    date_col_name='Price',      # Header-ul coloanei cu datele (prima coloană din CSV)
    price_col_name='Close',     # Header-ul coloanei cu prețurile (a doua coloană din CSV)
    date_format='%m/%d/%y',     # Format: e.g., 5/18/20
    skiprows_config=[1, 2]      # Skip rândurile 2 și 3 (0-indexat după header)
)

# Încărcarea și pregătirea datelor pentru Wise PLC
# Format dată pentru WISE: '2022-05-10' -> '%Y-%m-%d'
# Coloană preț pentru WISE: 'Close'
wise_returns_df = load_and_prepare_asset_data(
    wise_file, 
    "WISE", 
    date_col_name='Price', 
    price_col_name='Close', 
    date_format='%Y-%m-%d', # Updated date format
    skiprows_config=[1, 2]
)

def calculate_and_print_correlation(df1, df2, name1, name2):
    """Calculează și afișează corelația între două seturi de randamente."""
    if df1 is not None and df2 is not None:
        print(f"\n--- Corelație între {name1} și {name2} ---")
        combined_returns = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
        combined_returns_cleaned = combined_returns.dropna()

        if len(combined_returns_cleaned) > 1:
            correlation_matrix = combined_returns_cleaned.corr()
            print("\nMatricea de Corelare a Randamentelor Zilnice Logaritmice:")
            print(correlation_matrix)

            first_date_overlap = combined_returns_cleaned.index.min().strftime('%Y-%m-%d')
            last_date_overlap = combined_returns_cleaned.index.max().strftime('%Y-%m-%d')
            num_overlapping_days = len(combined_returns_cleaned)

            print(f"\nAnaliza de corelare s-a bazat pe {num_overlapping_days} zile de randament suprapuse,")
            print(f"de la data de {first_date_overlap} până la {last_date_overlap}.")

            corr_value = correlation_matrix.iloc[0, 1]
            print(f"\nCoeficientul de corelație între randamentele zilnice logaritmice ale {name1} și {name2} este: {corr_value:.4f}")
            if corr_value > 0.7:
                interpretation = "Corelație pozitivă puternică: activele tind să se miște semnificativ în aceeași direcție."
            elif corr_value > 0.3:
                interpretation = "Corelație pozitivă moderată: activele tind să se miște oarecum în aceeași direcție."
            elif corr_value > -0.3:
                interpretation = "Corelație slabă sau neutră: mișcările prețurilor sunt în mare parte independente."
            elif corr_value > -0.7:
                interpretation = "Corelație negativă moderată: activele tind să se miște oarecum în direcții opuse."
            else:
                interpretation = "Corelație negativă puternică: activele tind să se miște semnificativ în direcții opuse."
            print(f"Interpretare: {interpretation}")
        else:
            print(f"\nNu sunt suficiente date suprapuse pentru a calcula o corelație semnificativă între {name1} și {name2}.")
            if not combined_returns.empty:
                 print(f"INFO: Perioada de suprapunere a datelor înainte de dropna: {combined_returns.index.min()} - {combined_returns.index.max()}")
                 print(f"INFO: Număr de zile în suprapunere inițială: {len(combined_returns)}")
            else:
                 print(f"INFO: Nu s-a găsit nicio suprapunere de date între {name1} și {name2}.")
    else:
        print(f"\nNu s-au putut calcula randamentele pentru {name1} și/sau {name2}. Corelația nu poate fi calculată.")

# Calcul și afișare corelație Wise & ETH
if wise_returns_df is not None and eth_returns_df is not None:
    calculate_and_print_correlation(wise_returns_df, eth_returns_df, "Wise PLC", "ETH")
else:
    print("\nNu s-au putut încărca datele pentru Wise și/sau ETH.")

# Calcul și afișare corelație Vestas & Wise
# if vestas_returns_df is not None and wise_returns_df is not None:
#     calculate_and_print_correlation(vestas_returns_df, wise_returns_df, "Vestas", "Wise PLC")
# else:
#     print("\nNu s-au putut încărca datele pentru Vestas și/sau Wise.")

# Calcul și afișare corelație Vestas & ETH
# if vestas_returns_df is not None and eth_returns_df is not None:
#     calculate_and_print_correlation(vestas_returns_df, eth_returns_df, "Vestas", "ETH")
# else:
#     print("\nNu s-au putut încărca datele pentru Vestas și/sau ETH.")

# if eth_returns_df is not None and wise_returns_df is not None:
#     # Combinarea DataFrame-urilor de randamente pe baza indexului de dată (aliniere)
#     # Se păstrează doar datele pentru care există înregistrări în ambele seturi (intersecție)
#     combined_returns = pd.merge(eth_returns_df, wise_returns_df, left_index=True, right_index=True, how='inner')
    
#     # Eliminarea rândurilor unde oricare dintre randamente este NaN (primul rând din fiecare serie originală
#     # și orice alte nepotriviri de date care ar putea duce la NaN după merge)
#     combined_returns_cleaned = combined_returns.dropna()

#     if len(combined_returns_cleaned) > 1: # Avem nevoie de cel puțin 2 puncte de date pentru corelație
#         # Calculul matricei de corelare
#         correlation_matrix = combined_returns_cleaned.corr()
        
#         print("\nMatricea de Corelare a Randamentelor Zilnice Logaritmice (perioada de suprapunere):")
#         print(correlation_matrix)
        
#         first_date_overlap = combined_returns_cleaned.index.min().strftime('%Y-%m-%d')
#         last_date_overlap = combined_returns_cleaned.index.max().strftime('%Y-%m-%d')
#         num_overlapping_days = len(combined_returns_cleaned)
        
#         print(f"\nAnaliza de corelare s-a bazat pe {num_overlapping_days} zile de randament suprapuse,")
#         print(f"de la data de {first_date_overlap} până la {last_date_overlap}.")
        
#         # Interpretare scurtă
#         corr_value = correlation_matrix.iloc[0, 1]
#         print(f"\nCoeficientul de corelație între randamentele zilnice logaritmice ale ETH și Wise PLC este: {corr_value:.4f}")
#         if corr_value > 0.7:
#             interpretation = "Corelație pozitivă puternică: activele tind să se miște semnificativ în aceeași direcție."
#         elif corr_value > 0.3:
#             interpretation = "Corelație pozitivă moderată: activele tind să se miște oarecum în aceeași direcție."
#         elif corr_value > -0.3:
#             interpretation = "Corelație slabă sau neutră: mișcările prețurilor sunt în mare parte independente."
#         elif corr_value > -0.7:
#             interpretation = "Corelație negativă moderată: activele tind să se miște oarecum în direcții opuse."
#         else:
#             interpretation = "Corelație negativă puternică: activele tind să se miște semnificativ în direcții opuse."
#         print(f"Interpretare: {interpretation}")

#     else:
#         print("\nNu sunt suficiente date suprapuse pentru a calcula o corelație semnificativă între ETH și Wise PLC.")
#         if not combined_returns.empty:
#              print(f"INFO: Perioada de suprapunere a datelor înainte de dropna: {combined_returns.index.min()} - {combined_returns.index.max()}")
#              print(f"INFO: Număr de zile în suprapunere inițială: {len(combined_returns)}")
#         else:
#              print("INFO: Nu s-a găsit nicio suprapunere de date între cele două active.")

# else:
#     print("\nNu s-au putut calcula randamentele pentru unul sau ambele active. Matricea de corelare nu poate fi calculată.")

