import json # Import json for reading and writing
import numpy as np
import pandas as pd
# from pypdf import PdfReader # No longer needed

# Datele de intrare pentru active
er_ts = 0.0720  # Randament Așteptat Titluri de Stat (proxy T1 an)
vol_ts = 0.00   # Volatilitate Titluri de Stat

er_wise = 0.4800 # Randament Așteptat Wise PLC
vol_wise = 0.4143 # Volatilitate Wise PLC

er_eth = 0.4018  # Randament Așteptat ETH
vol_eth = 0.6624  # Volatilitate ETH

# Corelații
corr_wise_eth = 0.14
# Corelațiile cu Titluri de Stat (TS) nu sunt necesare deoarece vol_ts = 0

# Rata fără risc pentru Sharpe Ratio.
# Având în vedere că Titlurile de Stat (TS) au vol_ts = 0.00,
# randamentul lor așteptat er_ts este considerat rata fără risc.
rf_rate = er_ts

# Calea către fișierul JSON de intrare
input_json_file_path = "triplets.json"
all_portfolio_sharpe_data = []

print(f"Citire date din '{input_json_file_path}'...")
loaded_sim_data_list = []
try:
    with open(input_json_file_path, 'r') as f:
        data_from_json = json.load(f)
    if not isinstance(data_from_json, dict) or "triplets" not in data_from_json or not isinstance(data_from_json.get("triplets"), list):
        print(f"EROARE: '{input_json_file_path}' nu conține cheia 'triplets' cu o listă de rezultate sau formatul este incorect.")
        loaded_sim_data_list = []
    else:
        loaded_sim_data_list = data_from_json["triplets"]
except FileNotFoundError:
    print(f"EROARE: Fișierul '{input_json_file_path}' nu a fost găsit.")
except json.JSONDecodeError:
    print(f"EROARE: Nu s-a putut decoda JSON din '{input_json_file_path}'. Verificați formatul.")

if not loaded_sim_data_list:
    print("Nu există date de simulare de procesat. Se oprește scriptul.")
    exit()

print(f"S-au încărcat {len(loaded_sim_data_list)} seturi de date de simulare.")
print("Calcularea Sharpe Ratios...")

total_simulations = len(loaded_sim_data_list)

for idx, weights_pct in enumerate(loaded_sim_data_list):
    # Verifică dacă weights_pct este o listă și are 3 elemente
    if not isinstance(weights_pct, list) or len(weights_pct) != 3:
        # print(f"Atenție: Intrarea {idx+1} nu este o listă validă de 3 ponderi. Se omite. Detalii: {weights_pct}")
        continue

    w_ts_pct = weights_pct[0]
    w_wise_pct = weights_pct[1]
    w_eth_pct = weights_pct[2]

    # Conversia procentelor în zecimale
    w_ts = w_ts_pct / 100.0
    w_wise = w_wise_pct / 100.0
    w_eth = w_eth_pct / 100.0

    # Verifică dacă suma ponderilor este 100% (sau 1.0 în zecimale)
    if not np.isclose(w_ts + w_wise + w_eth, 1.0):
        # print(f"Atenție: Suma ponderilor nu este 1 pentru ({w_ts_pct}, {w_wise_pct}, {w_eth_pct}). Se omite.")
        continue

    # Calculul randamentului așteptat al portofoliului (E_Rp)
    e_rp = (w_ts * er_ts) + (w_wise * er_wise) + (w_eth * er_eth)
    
    # Calculul varianței portofoliului (Var_p)
    # Deoarece vol_ts = 0, termenii care implică w_ts și vol_ts sunt zero și sunt omiși.
    var_p = (w_wise**2 * vol_wise**2) + \
            (w_eth**2 * vol_eth**2) + \
            (2 * w_wise * w_eth * vol_wise * vol_eth * corr_wise_eth)
            
    sigma_p = np.sqrt(var_p)
    sharpe_ratio = 0
    if sigma_p > 1e-6: # Evită împărțirea la zero sau la un număr foarte mic
        sharpe_ratio = (e_rp - rf_rate) / sigma_p
    
    all_portfolio_sharpe_data.append({
        'W_TS_pct': w_ts_pct,
        'W_Wise_pct': w_wise_pct,
        'W_ETH_pct': w_eth_pct,
        'E_Rp': e_rp,
        'Sigma_p': sigma_p,
        'Sharpe_Ratio': sharpe_ratio
    })

    # Actualizare progres
    if total_simulations > 0:
        percentage_done = ((idx + 1) / total_simulations) * 100
        print(f"\rProcesare Sharpe Ratios: {percentage_done:.0f}% finalizat", end="")

if total_simulations > 0:
    print() # Newline după finalizarea buclei

# Sortare după Sharpe Ratio descrescător înainte de a salva
all_portfolio_sharpe_data.sort(key=lambda x: x['Sharpe_Ratio'], reverse=True)

# Scrierea tuturor datelor Sharpe Ratio în allsharpe_3assets.json
output_sharpe_json_file = "allsharpe_3assets.json"
if all_portfolio_sharpe_data:
    try:
        with open(output_sharpe_json_file, 'w') as outfile:
            json.dump(all_portfolio_sharpe_data, outfile, indent=4)
        print(f"\nToate datele Sharpe Ratio au fost salvate în '{output_sharpe_json_file}'.")
    except IOError:
        print(f"EROARE: Nu s-a putut scrie în fișierul '{output_sharpe_json_file}'.")
else:
    print("\nNu s-au calculat date Sharpe Ratio pentru a fi salvate.")

# Eliminarea vechii logici de afișare a DataFrame-ului și a print-urilor specifice PDF

