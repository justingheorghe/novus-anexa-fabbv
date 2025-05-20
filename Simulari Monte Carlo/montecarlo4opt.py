import numpy as np
import json
import sys # NEW: Added for progress bar

# Parametrii portofoliului și simulării
initial_investment = 100000  # Investiție inițială în EUR
# Ponderi: [Titluri Stat, Vestas, Wise, ETH]
# weights = np.array([0.50, 0.10, 0.10, 0.30]) # REMOVED - Weights will come from quadruplets
asset_names = ['Titluri Stat', 'Vestas', 'Wise PLC', 'ETH']

# Caracteristicile anuale ale activelor
# Randamente medii anuale estimate (zecimal)
mean_returns = np.array([0.0660, 0.0621, 0.0535, 0.4018])
# Volatilități anuale (deviații standard, zecimal)
volatilities = np.array([0.00, 0.4075, 0.4327, 0.6624])

# Matricea de corelație pentru activele riscante (Vestas, Wise, ETH)
# Ordinea în matrice: Vestas, Wise, ETH
corr_matrix_risky = np.array([
    [1.00, 0.15, 0.11],  # Vestas vs (Vestas, Wise, ETH)
    [0.15, 1.00, 0.17],  # Wise vs (Vestas, Wise, ETH)
    [0.11, 0.17, 1.00]   # ETH vs (Vestas, Wise, ETH)
])

num_years = 5
num_simulations = 10000

# NEW: Load quadruplets from JSON
QUADRUPLET_FILE = "quadruplets_divisible_by_5.json"
try:
    with open(QUADRUPLET_FILE, "r") as f:
        quadruplets_data = json.load(f)
    if 'quadruplets_divisible_by_5' not in quadruplets_data:
        raise ValueError("JSON file must contain 'quadruplets_divisible_by_5' key")
    loaded_quadruplets = quadruplets_data['quadruplets_divisible_by_5']
    if not loaded_quadruplets: # Check if the list is empty
        raise ValueError("No quadruplets found in the JSON file (list is empty).")
    if not isinstance(loaded_quadruplets, list) or not all(isinstance(q, list) for q in loaded_quadruplets):
        raise ValueError("Quadruplets must be a list of lists.")
except FileNotFoundError:
    print(f"EROARE: Fișierul {QUADRUPLET_FILE} nu a fost găsit.")
    exit(1)
except json.JSONDecodeError:
    print(f"EROARE: Fișierul {QUADRUPLET_FILE} nu este un JSON valid.")
    exit(1)
except ValueError as ve:
    print(f"EROARE în structura datelor din {QUADRUPLET_FILE}: {ve}")
    exit(1)

all_simulation_results = []

# Pregătirea pentru generarea randamentelor corelate (doar pentru activele riscante)
# Activele riscante sunt Vestas, Wise, ETH (indecșii 1, 2, 3 în array-urile principale)
mean_returns_risky = mean_returns[1:]
volatilities_risky = volatilities[1:] # volatilities for risky assets, used in simulation
# Cholesky decomposition
try:
    cholesky_decomp_risky = np.linalg.cholesky(corr_matrix_risky)
except np.linalg.LinAlgError:
    print("EROARE: Matricea de corelație pentru activele riscante nu este pozitiv definită.")
    print("Se continuă simularea presupunând corelație zero între activele riscante.")
    cholesky_decomp_risky = np.eye(len(mean_returns_risky))


# NEW: Loop over each quadruplet to use as weights
print("Inițiere procesare Monte Carlo pentru seturile de ponderi...") # NEW: Initial message
total_quadruplets = len(loaded_quadruplets)
processed_quadruplets_count = 0

for idx, quadruplet_values in enumerate(loaded_quadruplets):
    if not isinstance(quadruplet_values, list) or len(quadruplet_values) != len(asset_names):
        print(f"\nAVERTISMENT: Quadrupletul invalid {quadruplet_values} la indexul {idx} va fi omis (format incorect sau număr greșit de elemente).")
        total_quadruplets -= 1 # Adjust total count if one is skipped early
        continue
    
    current_weights = np.array(quadruplet_values) / 100.0
    
    if not np.isclose(np.sum(current_weights), 1.0):
        print(f"\nAVERTISMENT: Ponderile {current_weights.tolist()} din quadrupletul {quadruplet_values} (index {idx}) nu însumează 1.0 după normalizare. Se omite.")
        total_quadruplets -= 1 # Adjust total count
        continue

    # Calculul teoretic al randamentului și volatilității portofoliului (pentru ponderile curente)
    theoretical_mean_return_p = np.sum(current_weights * mean_returns)
    
    # Varianța portofoliului (doar componenta riscantă contează, volatilitatea[0] este 0)
    var_p_manual = (current_weights[1]**2 * volatilities[1]**2) + \
                   (current_weights[2]**2 * volatilities[2]**2) + \
                   (current_weights[3]**2 * volatilities[3]**2) + \
                   2 * current_weights[1] * current_weights[2] * volatilities[1] * volatilities[2] * corr_matrix_risky[0,1] + \
                   2 * current_weights[1] * current_weights[3] * volatilities[1] * volatilities[3] * corr_matrix_risky[0,2] + \
                   2 * current_weights[2] * current_weights[3] * volatilities[2] * volatilities[3] * corr_matrix_risky[1,2]
    theoretical_std_dev_p = np.sqrt(var_p_manual)

    current_run_final_portfolio_values = [] # NEW: Stores final values for this specific quadruplet's simulations

    # Rularea simulărilor Monte Carlo
    for _ in range(num_simulations): # MODIFIED - loop variable i to _
        portfolio_value = initial_investment
        for _ in range(num_years): # MODIFIED - loop variable year to _
            # Generarea randamentelor anuale aleatoare
            ts_return_annual = mean_returns[0] # Este determinist
            
            uncorrelated_randoms_risky = np.random.normal(0, 1, len(mean_returns_risky))
            correlated_randoms_risky = np.dot(cholesky_decomp_risky, uncorrelated_randoms_risky)
            risky_asset_returns_annual = mean_returns_risky + volatilities_risky * correlated_randoms_risky
            
            all_asset_returns_annual = np.concatenate(([ts_return_annual], risky_asset_returns_annual))
            portfolio_return_annual = np.sum(current_weights * all_asset_returns_annual) # MODIFIED - uses current_weights
            
            portfolio_value *= (1 + portfolio_return_annual)
        
        current_run_final_portfolio_values.append(portfolio_value) # MODIFIED - appends to local list

    current_final_portfolio_values_np = np.array(current_run_final_portfolio_values)

    # Calculul statisticilor cheie din simulare pentru acest set de ponderi
    mean_final_val = np.mean(current_final_portfolio_values_np)
    median_final_val = np.median(current_final_portfolio_values_np)
    p5_val = np.percentile(current_final_portfolio_values_np, 5)
    p95_val = np.percentile(current_final_portfolio_values_np, 95)
    std_dev_final_val = np.std(current_final_portfolio_values_np)
    prob_loss = np.sum(current_final_portfolio_values_np < initial_investment) / num_simulations * 100

    result_for_quadruplet = {
        "Weights": quadruplet_values, # MODIFIED: Use original quadruplet values and capitalized key
        "Mean": mean_final_val, # MODIFIED: Key name change
        "Median": median_final_val, # MODIFIED: Key name change
        "5th_Percentile": p5_val, # MODIFIED: Key name change
        "95th_Percentile": p95_val # MODIFIED: Key name change
        # REMOVED: theoretical_annual_mean_return
        # REMOVED: theoretical_annual_std_dev
        # REMOVED: simulated_final_values (full list)
        # REMOVED: simulated_std_dev_final_value
        # REMOVED: simulated_probability_of_loss_percent
    }
    all_simulation_results.append(result_for_quadruplet)
    processed_quadruplets_count += 1

    # NEW: Progress bar logic
    percent_done = (processed_quadruplets_count / total_quadruplets) * 100 if total_quadruplets > 0 else 100
    bar_length = 40
    filled_length = int(bar_length * processed_quadruplets_count // total_quadruplets) if total_quadruplets > 0 else bar_length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rProcesare: [{bar}] {percent_done:.2f}% ({processed_quadruplets_count}/{total_quadruplets})')
    sys.stdout.flush()

if total_quadruplets > 0: # Ensure we print a newline only if progress was shown
    sys.stdout.write('\n') # NEW: Newline after progress bar is complete
sys.stdout.flush() # Ensure the newline is printed

# NEW: Salvarea tuturor rezultatelor într-un fișier JSON
OUTPUT_JSON_FILE = "monte_carlo_simulations_output.json"
with open(OUTPUT_JSON_FILE, "w") as f_out:
    json.dump(all_simulation_results, f_out, indent=4)

# MODIFIED: Ensure this print is on a new line and clear
if not all_simulation_results and loaded_quadruplets: # If all valid quadruplets were skipped
    print(f"\nNiciun set de ponderi valid nu a fost procesat. Verificati fisierul {QUADRUPLET_FILE}.")
elif not loaded_quadruplets: # If the input file was empty to begin with
    print(f"\nFișierul de intrare {QUADRUPLET_FILE} nu conține niciun quadruplet.")
elif all_simulation_results:
    print(f"Procesare finalizată.\nToate rezultatele simulărilor Monte Carlo ({len(all_simulation_results)} seturi de ponderi procesate) au fost salvate în: {OUTPUT_JSON_FILE}")
else: # Catch-all for other scenarios, e.g. if loaded_quadruplets was initially empty and total_quadruplets became 0
    print(f"\nNicio simulare nu a fost efectuată. Verificati fisierul {QUADRUPLET_FILE} și setările.")
