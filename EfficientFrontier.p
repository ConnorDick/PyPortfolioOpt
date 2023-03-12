import pandas as pd
import pandas_datareader.data as web
import datetime
from functools import reduce
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
start = datetime.datetime(2022, 1, 1)
end = datetime.datetime(2022, 12, 1)

#get data for single stock
def singleStockData(ticker):
    stockInfo = web.DataReader(f"{ticker}","yahoo",start,end)
    stockInfo[f'{ticker}'] = stockInfo["Close"]
    stockInfo = stockInfo[[f'{ticker}']] 
    return stockInfo 

#get data for portfolio of stocks
def multipleStockData(tickers):
    df = []
    for i in tickers:
        df.append(singleStockData(i))
    df = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='outer'), df)
    print(df)
    return df

#Choose options for portofolio investments
investments = ["MSFT", "TSLA", "DIA", "LMT", "SPY"]
#Compile stock price history for each stock in portfolio
portfolio = multipleStockData(investments)

#determine expected return from historical returns
meanHistoricalReturn = mean_historical_return(portfolio)
#estimates covariance matrix
ledoitWolfShrinkage = CovarianceShrinkage(portfolio).ledoit_wolf()

efficiency = EfficientFrontier(meanHistoricalReturn, ledoitWolfShrinkage)
weights = efficiency.max_sharpe()
cleanedWeights = efficiency.clean_weights()
print(cleanedWeights)

#Based on weights, determine optimal allocation given total portfolio value
efficiency.portfolio_performance(verbose=True)
latestPrices = get_latest_prices(portfolio)
alloc = DiscreteAllocation(weights, latestPrices, total_portfolio_value=10000)
alloc, leftover = alloc.lp_portfolio()
print("Discrete allocation:", alloc)
