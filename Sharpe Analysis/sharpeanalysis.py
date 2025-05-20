import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
# from scipy.interpolate import griddata # No longer needed for this approach

# Fix file path issue
def find_file(filename, script_dir):
    path_in_script_dir = os.path.join(script_dir, filename)
    if os.path.exists(path_in_script_dir):
        return path_in_script_dir
    
    # Check for 'Sharpe Analysis' subdirectory if script is run from parent of script_dir
    # Assumes script_dir is .../FABBV calulations/Sharpe Analysis
    # and CWD could be .../FABBV calulations/
    # Target: .../FABBV calulations/Sharpe Analysis/allsharpe.json
    # Construct path assuming CWD is parent of script_dir, and filename is in script_dir
    # Correct path would be os.path.join(os.getcwd(), os.path.basename(script_dir), filename)
    # Let's simplify: if CWD is the parent of script_dir, try script_dir/filename
    # This is already covered by the first check if script_dir is correct.

    # More robust check: if CWD is the parent of where the script is, 
    # and allsharpe.json is in the script's directory.
    # script_name_in_cwd = os.path.join(os.getcwd(), os.path.basename(script_dir), filename)
    # This logic was becoming overly complex, simpler approach:
    # 1. Try script_dir / filename (most direct)
    # 2. Try cwd / filename (if running from within the dir)
    # 3. Try cwd / 'Sharpe Analysis' / filename (if running from parent of Sharpe Analysis)

    path_in_cwd = os.path.join(os.getcwd(), filename)
    if os.path.exists(path_in_cwd):
        return path_in_cwd

    path_in_subdir_of_cwd = os.path.join(os.getcwd(), "Sharpe Analysis", filename)
    if os.path.exists(path_in_subdir_of_cwd):
        return path_in_subdir_of_cwd
    
    # Fallback for the case where script is in Sharpe Analysis, and run from parent of Sharpe Analysis
    # os.path.abspath(__file__) gives /path/to/FABBV calulations/Sharpe Analysis/sharpeanalysis.py
    # script_directory is /path/to/FABBV calulations/Sharpe Analysis
    # so path_in_script_dir is /path/to/FABBV calulations/Sharpe Analysis/allsharpe.json
    # This should be the primary correct path.

    return filename # Fallback, will likely cause error if not found by above

script_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = find_file('allsharpe_3assets.json', script_directory)

print(f"Attempting to load portfolio data from: {json_file_path}")

try:
    with open(json_file_path, 'r') as f:
        portfolios = json.load(f)
    print(f"Successfully loaded {len(portfolios)} portfolios.")
except FileNotFoundError:
    # Attempt to load from 'Sharpe Analysis/allsharpe_3assets.json' if current path fails and CWD is parent
    # This case should be handled if running the script directly using python sharpeanalysis.py from within Sharpe Analysis folder
    # or python "Sharpe Analysis/sharpeanalysis.py" from parent.
    # The issue might be if the find_file logic is not exhaustive for all execution methods.
    # The most direct path from the script location should be preferred.
    direct_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'allsharpe_3assets.json')
    print(f"Primary attempt failed. Trying direct path: {direct_path}")
    try:
        with open(direct_path, 'r') as f:
            portfolios = json.load(f)
            json_file_path = direct_path # Update path if successful
        print(f"Successfully loaded {len(portfolios)} portfolios from direct path.")
    except FileNotFoundError:
        print(f"Error: {json_file_path} (and {direct_path}) file not found.")
        print("Script directory:", script_directory)
        print("Current working directory:", os.getcwd())
        exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format in allsharpe.json.")
    exit(1)

returns = np.array([p['E_Rp'] for p in portfolios])
volatilities = np.array([p['Sigma_p'] for p in portfolios])
sharpe_ratios = np.array([p['Sharpe_Ratio'] for p in portfolios])

max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe_portfolio = portfolios[max_sharpe_idx]

print("\n--- Optimal Portfolio (Maximum Sharpe Ratio) ---")
print(f"Titluri de Stat: {max_sharpe_portfolio['W_TS_pct']}%")
print(f"Wise: {max_sharpe_portfolio['W_Wise_pct']}%")
print(f"ETH: {max_sharpe_portfolio['W_ETH_pct']}%")
print(f"Expected Return: {max_sharpe_portfolio['E_Rp']:.2%}")
print(f"Volatility: {max_sharpe_portfolio['Sigma_p']:.2%}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe_Ratio']:.4f}")

equal_weights = [100/3, 100/3, 100/3]
equal_weighted_portfolio = None
min_equal_diff = float('inf')

for p in portfolios:
    current_diff = (abs(p['W_TS_pct'] - equal_weights[0]) +
                    abs(p['W_Wise_pct'] - equal_weights[1]) +
                    abs(p['W_ETH_pct'] - equal_weights[2]))
    if current_diff < min_equal_diff:
        min_equal_diff = current_diff
        equal_weighted_portfolio = p
    if current_diff == 0:
        break

if equal_weighted_portfolio is None and portfolios:
    equal_weighted_portfolio = {
        'W_TS_pct': equal_weights[0], 'W_Wise_pct': equal_weights[1], 'W_ETH_pct': equal_weights[2],
        'E_Rp': np.mean(returns), 'Sigma_p': np.mean(volatilities), 'Sharpe_Ratio': np.mean(sharpe_ratios),
    }
elif not portfolios:
    print("Error: No portfolios loaded, cannot define equal weighted portfolio.")
    equal_weighted_portfolio = {
        'W_TS_pct': 100/3, 'W_Wise_pct': 100/3, 'W_ETH_pct': 100/3,
        'E_Rp': 0, 'Sigma_p': 0, 'Sharpe_Ratio': 0
    }

# Define the target allocation for the "Balanced" portfolio
target_balanced_alloc = {'W_TS_pct': 52, 'W_Wise_pct': 38, 'W_ETH_pct': 10}
middle_ground_portfolio = None
min_diff = float('inf')

for p in portfolios:
    diff = (abs(p['W_TS_pct'] - target_balanced_alloc['W_TS_pct']) +
            abs(p['W_Wise_pct'] - target_balanced_alloc['W_Wise_pct']) +
            abs(p['W_ETH_pct'] - target_balanced_alloc['W_ETH_pct']))
    if diff < min_diff:
        min_diff = diff
        middle_ground_portfolio = p
    if diff == 0:
        break

if middle_ground_portfolio:
    print("\n--- User-Defined Balanced Portfolio (Closest Match to 52/38/10) ---")
    print(f"Found with difference score: {min_diff} (lower is better)")
    print(f"Titluri de Stat: {middle_ground_portfolio['W_TS_pct']}%")
    print(f"Wise: {middle_ground_portfolio['W_Wise_pct']}%")
    print(f"ETH: {middle_ground_portfolio['W_ETH_pct']}%")
    print(f"Expected Return: {middle_ground_portfolio['E_Rp']:.2%}")
    print(f"Volatility: {middle_ground_portfolio['Sigma_p']:.2%}")
    print(f"Sharpe Ratio: {middle_ground_portfolio.get('Sharpe_Ratio', np.nan):.4f}")
else:
    if portfolios:
        middle_ground_portfolio = portfolios[len(portfolios)//2] 
    else:
        middle_ground_portfolio = {
            'W_TS_pct': target_balanced_alloc['W_TS_pct'],
            'W_Wise_pct': target_balanced_alloc['W_Wise_pct'],
            'W_ETH_pct': target_balanced_alloc['W_ETH_pct'],
            'E_Rp': 0, 'Sigma_p': 0, 'Sharpe_Ratio': 0
        }
    print("\n--- WARNING: Could not find a close match for user-defined balanced portfolio. Using arbitrary/dummy portfolio. ---")

# Calculate Efficient Frontier points
efficient_portfolios_indices = []
sorted_returns_indices = np.argsort(returns)
max_vol_for_return = -1
# Simpler way to get points for frontier: iterate through unique returns and find min volatility
# However, a common way is to iterate sorted by volatility and keep points that improve return for similar/less volatility
# Or iterate sorted by return and keep points that improve volatility for similar/more return.
# For visualization, we want points that are not dominated.

# Create a list of (volatility, return, original_index) for sorting
pts = sorted([(volatilities[i], returns[i], sharpe_ratios[i], i) for i in range(len(portfolios))])

efficient_frontier_indices = []
max_return_so_far = -float('inf')
# Iterate through portfolios sorted by volatility
# A point is on the frontier if no other point offers higher return for same or lower volatility
# More robust: iterate by volatility, keep if it has higher return than last frontier point with similar/lower volatility
# Or, iterate by return, keep if it has lower vol than last frontier point with similar/higher return

dominated = [False] * len(portfolios)
for i in range(len(portfolios)):
    for j in range(len(portfolios)):
        if i == j: continue
        if returns[j] >= returns[i] and volatilities[j] <= volatilities[i] and (returns[j] > returns[i] or volatilities[j] < volatilities[i]):
            dominated[i] = True
            break
efficient_indices = [i for i, d in enumerate(dominated) if not d]

if not efficient_indices: # Fallback if simple non-dominated logic fails badly
    # Fallback to a simpler approach if no points found (e.g. data issues)
    # Take points with sharpe ratio > median sharpe ratio as a proxy for frontier-like points
    median_sharpe = np.median(sharpe_ratios)
    efficient_indices = [i for i, sr in enumerate(sharpe_ratios) if sr > median_sharpe]
    if not efficient_indices: # if still no points, take all points (no frontier line)
        efficient_indices = list(range(len(portfolios)))

efficient_volatilities = volatilities[efficient_indices]
efficient_returns = returns[efficient_indices]
sorted_frontier_indices = np.argsort(efficient_volatilities)
efficient_volatilities = efficient_volatilities[sorted_frontier_indices]
efficient_returns = efficient_returns[sorted_frontier_indices]

print("\nCreating 2D Efficient Frontier visualization...")
plt.figure(figsize=(12, 9))
plt.grid(True, alpha=0.3, color='#d0d0d0')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
plt.gca().set_facecolor('#f0f0f8')

cmap = plt.cm.inferno # black/purple (low) to yellow (high)
norm = Normalize(vmin=np.percentile(sharpe_ratios, 5), vmax=np.percentile(sharpe_ratios, 95)) # Normalize based on percentiles for better color spread

sc = plt.scatter(
    volatilities, returns, c=sharpe_ratios, cmap=cmap, norm=norm,
    s=70, alpha=0.8, edgecolors='none', zorder=2
)

if len(efficient_volatilities) > 1: # Only plot if we have enough points for a line
    plt.plot(efficient_volatilities, efficient_returns, 'w-', linewidth=2.5, zorder=3, alpha=0.7)

# User-Defined Balanced Portfolio (Blue Dot)
plt.scatter(
    middle_ground_portfolio['Sigma_p'], middle_ground_portfolio['E_Rp'],
    s=250, color='blue', marker='o', edgecolors='white', linewidth=2, zorder=5,
    label=f'Balanced Portfolio ({target_balanced_alloc["W_TS_pct"]}/{target_balanced_alloc["W_Wise_pct"]}/{target_balanced_alloc["W_ETH_pct"]})'
)

# Maximum Sharpe Ratio Portfolio (Green Star)
plt.scatter(
    max_sharpe_portfolio['Sigma_p'], max_sharpe_portfolio['E_Rp'],
    s=350, color='green', marker='*', edgecolors='white', linewidth=1.5, zorder=5,
    label=f'Max Sharpe Ratio ({max_sharpe_portfolio["W_TS_pct"]}/{max_sharpe_portfolio["W_Wise_pct"]}/{max_sharpe_portfolio["W_ETH_pct"]})'
)

cbar = plt.colorbar(sc)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=12)

plt.xlabel('Volatility', fontsize=12)
plt.ylabel('Expected Return', fontsize=12)
plt.title('Portfolio Optimization: Risk-Return Landscape', fontsize=16)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

balanced_weights_text = f"TS: {target_balanced_alloc['W_TS_pct']}%\nWise: {target_balanced_alloc['W_Wise_pct']}%\nETH: {target_balanced_alloc['W_ETH_pct']}%"
plt.annotate(
    f'Balanced (User Defined)\nReturn: {middle_ground_portfolio["E_Rp"]:.2%}\nVol: {middle_ground_portfolio["Sigma_p"]:.2%}\nSharpe: {middle_ground_portfolio.get("Sharpe_Ratio", np.nan):.2f}\n{balanced_weights_text}',
    xy=(middle_ground_portfolio['Sigma_p'], middle_ground_portfolio['E_Rp']),
    xytext=(middle_ground_portfolio['Sigma_p'] + 0.02, middle_ground_portfolio['E_Rp'] + 0.02),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8, connectionstyle="arc3,rad=.2"),
    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="gray", alpha=0.9)
)

plt.figtext(0.5, 0.01, 
           "Efficient Frontier: Optimal trade-off portfolios (higher return for same/lower risk, or lower risk for same/higher return).",
           fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8, pad=0.2))

plt.legend(loc='lower right', fontsize=10)
plt.savefig('efficient_frontier_plot.png', dpi=300, bbox_inches='tight')
print("2D Visualization saved as 'efficient_frontier_plot.png'")
plt.close()

print("\nCreating 3D visualization...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    volatilities, returns, sharpe_ratios, c=sharpe_ratios, cmap=cmap, norm=norm,
    s=50, alpha=0.8
)

# Highlight Middle Ground Portfolio
ax.scatter(
    [middle_ground_portfolio['Sigma_p']], [middle_ground_portfolio['E_Rp']], [middle_ground_portfolio.get('Sharpe_Ratio', np.nan)],
    color='blue', s=200, marker='o', edgecolors='white', linewidth=2, 
    label=f'Balanced ({target_balanced_alloc["W_TS_pct"]}/{target_balanced_alloc["W_Wise_pct"]}/{target_balanced_alloc["W_ETH_pct"]})'
)
# Highlight Max Sharpe Portfolio
ax.scatter(
    [max_sharpe_portfolio['Sigma_p']], [max_sharpe_portfolio['E_Rp']], [max_sharpe_portfolio['Sharpe_Ratio']],
    color='green', s=300, marker='*', edgecolors='white', linewidth=1.5, 
    label=f'Max Sharpe ({max_sharpe_portfolio["W_TS_pct"]}/{max_sharpe_portfolio["W_Wise_pct"]}/{max_sharpe_portfolio["W_ETH_pct"]})'
)
# Highlight Equally Weighted Portfolio
ax.scatter(
    [equal_weighted_portfolio['Sigma_p']], [equal_weighted_portfolio['E_Rp']], [equal_weighted_portfolio['Sharpe_Ratio']],
    color='yellow', s=200, marker='o', edgecolors='black', linewidth=1.5, 
    label=f'Equally Weighted ({equal_weighted_portfolio["W_TS_pct"]}/{equal_weighted_portfolio["W_Wise_pct"]}/{equal_weighted_portfolio["W_ETH_pct"]})'
)

ax.set_xlabel('Volatility', fontsize=12)
ax.set_ylabel('Expected Return', fontsize=12)
ax.set_zlabel('Sharpe Ratio', fontsize=12)
ax.set_title('3D Portfolio Metrics Landscape', fontsize=14)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=12)
ax.legend(loc='upper left')
plt.savefig('portfolio_3d_plot.png', dpi=300, bbox_inches='tight')
print("3D visualization saved as 'portfolio_3d_plot.png'")
plt.close('all')

print("\nAll visualizations have been saved successfully!")
print("1. efficient_frontier_plot.png - 2D risk-return landscape")
print("2. portfolio_3d_plot.png - 3D visualization of metrics")
