import numpy as np
import json
import matplotlib.pyplot as plt # Added for plotting

# Portfolio Parameters
# Assets: 0: Titluri de stat (Government Bonds), 1: Vestas, 2: Wise, 3: ETH
asset_names = ["Titluri de stat", "Wise", "ETH"]

# Weights for the portfolio
weights = np.array([0.52, 0.38, 0.10]) # Titluri de stat: 60%, Wise: 25%, ETH: 15%

# Annual expected returns
# For Titluri de stat, using 0.07 as a placeholder (similar to the first asset in sims.py).
# For Vestas, Wise, ETH, using user-provided values.
expected_returns = np.array([
    0.072,   # Titluri de stat (7.2%)
    0.48,    # Wise (48%)
    0.4018   # ETH (40.18%)
])

# Annual volatilities
# For Titluri de stat, using 0.000001 as a placeholder (very low, similar to the first asset in sims.py).
# For Vestas, Wise, ETH, using user-provided values.
volatilities = np.array([
    0.0,      # Titluri de stat (0%)
    0.4143,   # Wise (41.43%)
    0.6624    # ETH (66.24%)
])

# Correlation Matrix
# Order: Titluri de stat, Wise, ETH
# ETH - WISE: 0.14
# Titluri de stat assumed to have 0 correlation with other assets.
correlation_matrix = np.array([
    [1.0, 0.0,  0.0 ],  # Titluri de stat
    [0.0, 1.0,  0.14],  # Wise
    [0.0, 0.14, 1.0 ]   # ETH
])

# Calculate covariance matrix
# Ensure volatilities and correlation_matrix have compatible dimensions
if volatilities.shape[0] != correlation_matrix.shape[0]:
    raise ValueError(f"Shape mismatch: volatilities ({volatilities.shape[0]}) and correlation_matrix ({correlation_matrix.shape[0]}) must have the same number of assets.")
cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

# Simulation parameters
num_simulations = 100000
initial_investment = 100000
T = 5  # Investment horizon in years (e.g., 5 years)

# Time-adjusted parameters
expected_returns_T = expected_returns * T
cov_matrix_T = cov_matrix * T

# Cholesky decomposition
# Adding a small epsilon to the diagonal for numerical stability if needed,
# but usually np.linalg.cholesky handles positive semi-definite matrices well.
# Ensure cov_matrix_T is positive definite, or handle potential np.linalg.LinAlgError
try:
    L = np.linalg.cholesky(cov_matrix_T)
except np.linalg.LinAlgError as e:
    print(f"Cholesky decomposition failed: {e}")
    print("The covariance matrix might not be positive definite.")
    # Attempt to regularize the matrix by adding a small value to the diagonal
    epsilon = 1e-6 
    cov_matrix_T_reg = cov_matrix_T + np.eye(cov_matrix_T.shape[0]) * epsilon
    try:
        print("Attempting Cholesky decomposition with regularized covariance matrix...")
        L = np.linalg.cholesky(cov_matrix_T_reg)
        print("Cholesky decomposition successful with regularization.")
    except np.linalg.LinAlgError as e_reg:
        print(f"Cholesky decomposition failed even with regularization: {e_reg}")
        print("Exiting script. Please check the covariance matrix inputs.")
        exit()


# Run Monte Carlo simulation
print(f"Running {num_simulations} Monte Carlo simulations for {T} years...")
portfolio_values = np.zeros(num_simulations)
num_assets = len(asset_names)

for i in range(num_simulations):
    # Generate random normal variables
    random_normals = np.random.normal(size=num_assets)
    
    # Calculate correlated returns for the period T
    # Formula: R_T = E[R_T] + L * Z
    # where Z is a vector of standard normal random variables
    correlated_period_returns = expected_returns_T + L @ random_normals
    
    # Calculate portfolio return for the period T
    portfolio_period_return = np.dot(weights, correlated_period_returns)
    
    # Calculate final portfolio value
    final_value = initial_investment * (1 + portfolio_period_return)
    portfolio_values[i] = final_value

# Calculate statistics
mean_val = np.mean(portfolio_values)
median_val = np.median(portfolio_values)
percentile_5 = np.percentile(portfolio_values, 5)
percentile_95 = np.percentile(portfolio_values, 95)

# Output results
print("\nSimulation Results:")
print(f"Initial Investment: ${initial_investment:,.2f}")
print(f"Investment Horizon: {T} years")
print("\nPortfolio Allocation:")
for i in range(len(asset_names)):
    print(f"- {asset_names[i]}: {weights[i]*100:.1f}%")

print("\nProjected Portfolio Value after {} years:".format(T))
print(f"Mean: ${mean_val:,.2f}")
print(f"Median: ${median_val:,.2f}")
print(f"5th Percentile (Value at Risk estimate): ${percentile_5:,.2f}")
print(f"95th Percentile: ${percentile_95:,.2f}")

# Generate and save histogram
plt.figure(figsize=(10, 6))
plt.hist(portfolio_values, bins=100, alpha=0.75, color='skyblue', edgecolor='black')
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_val:,.2f}')
plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_val:,.2f}')
plt.axvline(percentile_5, color='purple', linestyle='dashed', linewidth=2, label=f'5th Pctl: ${percentile_5:,.2f}')
plt.axvline(percentile_95, color='orange', linestyle='dashed', linewidth=2, label=f'95th Pctl: ${percentile_95:,.2f}')

plt.title(f'Monte Carlo Simulation of Portfolio Value ({num_simulations} Simulations, {T} Years)')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plot_file_path = "montecarlo_histogram.png"
try:
    plt.savefig(plot_file_path)
    print(f"\nHistogram saved to '{plot_file_path}'")
except Exception as e:
    print(f"ERROR: Could not save histogram: {e}")

# plt.show() # Uncomment to display the plot interactively if running in a GUI environment

# Store results in a dictionary
results_data = {
    "simulation_parameters": {
        "initial_investment": initial_investment,
        "investment_horizon_years": T,
        "num_simulations": num_simulations
    },
    "portfolio_composition": {asset_names[i]: weights[i] for i in range(len(asset_names))},
    "asset_assumptions": {
        "expected_annual_returns": {asset_names[i]: expected_returns[i] for i in range(len(asset_names))},
        "annual_volatilities": {asset_names[i]: volatilities[i] for i in range(len(asset_names))},
        "correlation_matrix": correlation_matrix.tolist() # Convert numpy array to list for JSON
    },
    "projected_values": {
        "mean": mean_val,
        "median": median_val,
        "5th_percentile": percentile_5,
        "95th_percentile": percentile_95
    }
}

# Write results to a JSON file
output_json_file_path = "montecarlo_results.json"
try:
    with open(output_json_file_path, 'w') as outfile:
        json.dump(results_data, outfile, indent=4)
    print(f"\nResults have been saved to '{output_json_file_path}'")
except IOError:
    print(f"ERROR: Could not write results to '{output_json_file_path}'.")

print("\nMonte Carlo simulation finished.")
