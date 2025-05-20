import numpy as np
import json # Import the json module
# import matplotlib.pyplot as plt # Removed for no visual output

# Portfolio parameters
expected_returns = np.array([0.072, 0.4018, 0.48])  # Annual expected returns #1st is TS #2nd is ETH #3rd is Wise
volatilities = np.array([0.000001, 0.6624, 0.4143])    # Annual volatilities
correlation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.14],
    [0.0, 0.14, 1.0]
])

cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

num_simulations = 100000
initial_investment = 100000
T = 5  # Investment horizon in years

expected_returns_T = expected_returns * T
cov_matrix_T = cov_matrix * T
L = np.linalg.cholesky(cov_matrix_T)

# Load triplets from triplets.json
triplets_file_path = "triplets.json"
triplets = []
try:
    with open(triplets_file_path, 'r') as f:
        data = json.load(f) # Load the entire JSON object
        if isinstance(data, dict) and "triplets" in data and isinstance(data["triplets"], list):
            triplets = data["triplets"] # Extract the list associated with the "triplets" key
        else:
            print(f"EROARE: '{triplets_file_path}' nu conține un obiect JSON cu cheia 'triplets' care să mapeze la o listă.")
            triplets = [] # Ensure triplets is empty if format is wrong

    # Validate that each item in the triplets list is a list/tuple of 3 numbers
    if triplets and not all(isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(n, (int, float)) for n in t) for t in triplets):
        print(f"EROARE: Lista 'triplets' din '{triplets_file_path}' conține elemente care nu sunt liste/tupluri valide de 3 numere.")
        triplets = [] # Reset to empty if content is invalid

except FileNotFoundError:
    print(f"EROARE: Fișierul '{triplets_file_path}' nu a fost găsit.")
    triplets = [] # Reset to empty
except json.JSONDecodeError:
    print(f"EROARE: Nu s-a putut decoda JSON din '{triplets_file_path}'. Verificați formatul.")
    triplets = [] # Reset to empty
except Exception as e:
    print(f"EROARE: O eroare neașteptată la citirea '{triplets_file_path}': {e}")
    triplets = []

# Monte Carlo Simulation for each triplet
all_simulation_outputs = []

if not triplets:
    print("Nicio combinație validă (triplet) nu a fost încărcată sau fișierul nu a putut fi procesat. Se oprește simularea.")
else:
    num_assets = len(expected_returns) # Determine number of assets
    total_triplets = len(triplets)
    print(f"Se rulează simulările Monte Carlo pentru {total_triplets} combinații de ponderi...")

    # Progress bar settings
    bar_length = 50

    for i, triplet_raw in enumerate(triplets):
        weights = np.array(triplet_raw) / 100.0 # Assuming weights in triplets are percentages

        if len(weights) != num_assets:
            print(f"Atenționare: Tripletul {triplet_raw} (index {i}) nu are numărul corect de ponderi ({len(weights)} vs {num_assets}). Acest triplet va fi omis.")
            continue

        Z = np.random.multivariate_normal(np.zeros(num_assets), np.eye(num_assets), num_simulations)
        simulated_asset_returns_T = expected_returns_T + Z @ L.T
        portfolio_simulated_returns_T = simulated_asset_returns_T @ weights
        final_portfolio_values = initial_investment * (1 + portfolio_simulated_returns_T)

        # Calculate statistics
        mean_val = np.mean(final_portfolio_values)
        median_val = np.median(final_portfolio_values)
        percentile_5 = np.percentile(final_portfolio_values, 5)
        percentile_95 = np.percentile(final_portfolio_values, 95)

        all_simulation_outputs.append({
            "Weights": triplet_raw, # Store original triplet values
            "Mean": mean_val,
            "Median": median_val,
            "5th_Percentile": percentile_5,
            "95th_Percentile": percentile_95
        })

        # Update progress bar
        progress = (i + 1) / total_triplets
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'Progres: |{bar}| {progress*100:.1f}% Complet ({i+1}/{total_triplets})', end='\r')

    print() # Ensure the next print is on a new line
    print("Simulările Monte Carlo au fost finalizate.") # Newline after progress bar

    output_file_path = "finalmontesims.json"
    try:
        with open(output_file_path, 'w') as f:
            json.dump(all_simulation_outputs, f, indent=4) # Save list directly
        print(f"Rezultatele simulării Monte Carlo au fost salvate în '{output_file_path}'")
    except IOError:
        print(f"EROARE: Nu s-a putut scrie în fișierul '{output_file_path}'.")
    except TypeError as e:
        print(f"EROARE: Eroare de serializare JSON la salvarea rezultatelor: {e}")

print("Scriptul de simulare a ajuns la final.")

# Removed plotting and individual print statements for statistics