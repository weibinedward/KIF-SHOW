import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize

# Filter out warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Define the list of stocks and convert to uppercase
stock_list = [
    'anto',
    'byit',
    'hyud',
    'mks',
    'stan',
    '0qb8.ln',
    '0qzd.ln',
    '0r2y.ln',
    'air.fp',
    'bbva.e.dx',
    'bmw.d.dx',
]

# Create an empty DataFrame to hold all stock data
df_all = pd.DataFrame()

# Loop through each stock to read the data and generate analysis
for stock in stock_list:
    try:
        # Read CSV data
        df = pd.read_csv(f"{stock}.csv")
        df = df[:-1]

        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'])

        # Filter the data within the specified date range
        start_date = pd.to_datetime('2023-01-05')
        end_date = pd.to_datetime('2023-06-05')
        df = df.loc[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

        # Set 'Time' column as the index
        df.set_index('Time', inplace=True)

        # Extract the 'Last' column (closing prices) and rename it to the stock name
        df_stock = df[['Last']].rename(columns={'Last': stock})

        # Append the stock data to the consolidated DataFrame
        df_all = pd.concat([df_all, df_stock], axis=1)
    except Exception as e:
        print(f"Couldn't fetch data for {stock}. Error: {e}")

# Fill missing values in the data using polynomial interpolation
StockPrices = df_all.interpolate(method='polynomial', order=3)
print(StockPrices)

# Calculate daily returns and drop missing values
StockReturns = StockPrices.pct_change().dropna()

# Copy returns data to a new variable for convenience
stock_return = StockReturns.copy()

# Calculate cumulative returns for each stock
cumulative_returns = (1 + stock_return).cumprod()

# Plot returns
plt.figure(figsize=(12, 8))
for stock in stock_return.columns:
    plt.plot(stock_return.index, stock_return[stock], label=stock)

# Set plot labels and title
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Returns of Stocks')
plt.legend()

# Display the plot
plt.show()

# Plot cumulative returns
plt.figure(figsize=(12, 8))
for stock in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)

# Set plot labels and title
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns of Stocks')
plt.legend()

# Display the plot
plt.show()

# Set the number of stocks in the portfolio
numstocks = 11

# Allocate equal weights to each item
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)

# Calculate the returns of the equally weighted portfolio
StockReturns['Portfolio_EW'] = stock_return.mul(portfolio_weights_ew, axis=1).sum(axis=1)

# Define a function to plot cumulative returns
def cumulative_returns_plot(name_list):
    for name in name_list:
        CumulativeReturns = ((1+StockReturns[name]).cumprod()-1)
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()

# Plot cumulative returns
cumulative_returns_plot(['Portfolio_EW'])

# Calculate the correlation matrix
correlation_matrix = stock_return.corr()

# Calculate the covariance matrix
cov_mat = stock_return.cov()

# Calculate the annualized covariance matrix
cov_mat_annual = cov_mat * 252

# Print the covariance matrix
print(cov_mat_annual)

# Import seaborn
import seaborn as sns

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu",
            linewidths=0.3,
            annot_kws={"size": 10})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Define the objective function to minimize - negative portfolio return
# Assume a risk-free rate of 2%
risk_free_rate = 0.02

def objective(weights):
    portfolio_return = np.sum(np.dot(stock_return.mean(), weights)) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_mat_annual, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio

# Define the constraint - weights must sum up to 1
constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Define the bounds for the weights - between 0 and 1
bounds = [(0, 1)] * numstocks

# Set an initial guess for the weights - equal allocation
initial_guess = np.repeat(1/numstocks, numstocks)

# Minimize the objective function
result = minimize(objective, initial_guess, method='SLSQP', constraints=constraint, bounds=bounds)

# Get the optimized weights
portfolio_weights_opt = result.x

# Calculate the optimized portfolio returns
StockReturns['Portfolio_Opt'] = stock_return.mul(portfolio_weights_opt, axis=1).sum(axis=1)

# Plot the cumulative returns of the optimized portfolio and equal-weighted portfolio
plt.figure(figsize=(12, 8))
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_Opt'])

# Calculate expected returns and volatilities
returns = stock_return.mean()
volatilities = stock_return.std()

# Define the number of points on the efficient frontier
num_points = 100

# Define the range of target returns for the efficient frontier
target_returns = np.linspace(returns.min(), returns.max(), num_points)

# Initialize arrays to store portfolio weights, returns, and volatilities
portfolio_weights_efficient = np.zeros((num_points, numstocks))
portfolio_returns_efficient = np.zeros(num_points)
portfolio_volatilities_efficient = np.zeros(num_points)

# Define the objective function to minimize portfolio volatility
def objective(weights):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(stock_return.cov(), weights)))
    return portfolio_volatility

# Define the constraint - weights must sum up to 1
constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Define the bounds for the weights - between 0 and 1
bounds = [(0, 1)] * numstocks

# Perform portfolio optimization for different target returns
for i, target_return in enumerate(target_returns):
    # Set the target return constraint
    target_constraint = {'type': 'eq', 'fun': lambda weights: np.dot(weights, returns) - target_return}

    # Set an initial guess for the weights - equal allocation
    initial_guess = np.repeat(1 / numstocks, numstocks)

    # Minimize the objective function with the return constraint
    result = minimize(objective, initial_guess, method='SLSQP', constraints=[constraint, target_constraint],
                      bounds=bounds)

    # Get the optimized weights
    portfolio_weights_efficient[i, :] = result.x

    # Calculate the optimized portfolio return and volatility
    portfolio_returns_efficient[i] = target_return
    portfolio_volatilities_efficient[i] = np.sqrt(np.dot(result.x.T, np.dot(stock_return.cov(), result.x)))

# Plot the efficient frontier and individual stock points
plt.figure(figsize=(12, 8))
plt.scatter(portfolio_volatilities_efficient, portfolio_returns_efficient, c='gray', alpha=0.5, label='Efficient Frontier')
plt.scatter(volatilities, returns, label='Individual Stocks', c='green')

plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier and Individual Stocks')
plt.legend()
plt.show()

# Define the objective function to minimize portfolio volatility
def minimize_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(stock_return.cov() * 252, weights)))

# Minimize volatility to obtain minimum volatility portfolio weights
result_min_vol = minimize(minimize_volatility, initial_guess, method='SLSQP', constraints=constraint, bounds=bounds)

# Get the minimum volatility portfolio weights
portfolio_weights_min_vol = result_min_vol.x

# Calculate the returns of the minimum volatility portfolio
StockReturns['Portfolio_Min_Vol'] = stock_return.mul(portfolio_weights_min_vol, axis=1).sum(axis=1)

# Round the weights to four decimal places
weights_opt = {stock: round(weight, 4) for stock, weight in zip(stock_list, portfolio_weights_opt)}
weights_min_vol = {stock: round(weight, 4) for stock, weight in zip(stock_list, portfolio_weights_min_vol)}

print('Optimized Portfolio Weights:', weights_opt)
print('Minimum Volatility Portfolio Weights:', weights_min_vol)

# Calculate the risk and return of the minimum volatility portfolio
min_vol_return = np.dot(portfolio_weights_min_vol, returns)
min_vol_risk = np.sqrt(np.dot(portfolio_weights_min_vol.T, np.dot(stock_return.cov(), portfolio_weights_min_vol)))

# Calculate the risk and return of the optimized portfolio
opt_return = np.dot(portfolio_weights_opt, returns)
opt_risk = np.sqrt(np.dot(portfolio_weights_opt.T, np.dot(stock_return.cov(), portfolio_weights_opt)))

# Mark the minimum volatility portfolio and the optimized portfolio on the efficient frontier plot
plt.figure(figsize=(12, 8))
plt.scatter(portfolio_volatilities_efficient, portfolio_returns_efficient, c='gray', alpha=0.5, label='Efficient Frontier')
plt.scatter(volatilities, returns, label='Individual Stocks', c='green')
plt.scatter(min_vol_risk, min_vol_return, color='green', marker='.', s=200)
plt.scatter(opt_risk, opt_return, color='b', marker='.', s=200)
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier and Individual Stocks')
plt.legend(['Efficient Frontier', 'Individual Stocks', 'Minimum Volatility Portfolio', 'Optimized Portfolio'])
plt.show()

# Calculate cumulative returns for each portfolio
cumulative_returns_opt = (1 + StockReturns['Portfolio_Opt']).cumprod() - 1
cumulative_returns_min_vol = (1 + StockReturns['Portfolio_Min_Vol']).cumprod() - 1
cumulative_returns_ew = (1 + StockReturns['Portfolio_EW']).cumprod() - 1

# Plot the cumulative returns of the portfolios
plt.figure(figsize=(12, 8))
plt.plot(cumulative_returns_opt.index, cumulative_returns_opt, label='Optimized Portfolio')
plt.plot(cumulative_returns_min_vol.index, cumulative_returns_min_vol, label='Minimum Volatility Portfolio')
plt.plot(cumulative_returns_ew.index, cumulative_returns_ew, label='Equal-Weighted Portfolio')

# Set plot labels and title
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns of Portfolios')
plt.legend()

# Display the plot
plt.show()

# Calculate the annualized return
annual_return_opt = np.sum(StockReturns['Portfolio_Opt']) * 252
annual_return_min_vol = np.sum(StockReturns['Portfolio_Min_Vol']) * 252

# Calculate the annualized volatility
annual_volatility_opt = np.sqrt(np.dot(portfolio_weights_opt.T, np.dot(cov_mat_annual, portfolio_weights_opt)))
annual_volatility_min_vol = np.sqrt(np.dot(portfolio_weights_min_vol.T, np.dot(cov_mat_annual, portfolio_weights_min_vol)))

# Calculate the Sharpe ratio (assuming a risk-free rate of 2%)
risk_free_rate = 0.02
sharpe_ratio_opt = (annual_return_opt - risk_free_rate) / annual_volatility_opt
sharpe_ratio_min_vol = (annual_return_min_vol - risk_free_rate) / annual_volatility_min_vol

# Print the results
print("Optimized Portfolio:")
print("Annualized Return: {:.2%}".format(annual_return_opt))
print("Annualized Volatility: {:.2%}".format(annual_volatility_opt))
print("Sharpe Ratio: {:.2f}".format(sharpe_ratio_opt))

print("\nMinimum Volatility Portfolio:")
print("Annualized Return: {:.2%}".format(annual_return_min_vol))
print("Annualized Volatility: {:.2%}".format(annual_volatility_min_vol))
print("Sharpe Ratio: {:.2f}".format(sharpe_ratio_min_vol))
