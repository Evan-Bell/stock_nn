from setup import *

# torch.set_printoptions(threshold=5, precision=3, sci_mode=False, linewidth=200)

# x = torch.rand(1000, 1,1)
# print(x)


# msft = yf.Ticker("MSFT")

# # get historical market data
# hist = msft.history(period="1d")
# print(hist)

# # show meta information about the history (requires history() to be called first)
# msft.history_metadata
# # show financials:
# # - income statement
# income = msft.income_stmt
# msft.quarterly_income_stmt
# # - balance sheet
# msft.balance_sheet
# msft.quarterly_balance_sheet
# # - cash flow statement
# msft.cashflow
# msft.quarterly_cashflow

# # show holders
# msft.major_holders
# msft.institutional_holders
# msft.mutualfund_holders

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
#msft.earnings_dates
# see `Ticker.get_income_stmt()` for more options

def get_inp_data_tickers(train_dist, label_delay, tickers = ['AAPL']):
  X = np.zeros((1, train_dist, 3))
  Y = np.zeros((1, 1, 3))

  for ticker in tickers:
    df = yf.download(ticker, period='max', threads=True)
    vals = np.array(df[['Low', 'High', 'Open', 'Close', 'Adj Close']].values)
    #print(vals)
    adj = 100*(100*vals[:,4]-100*vals[:,3])/(100*vals[:,3])
    change = 100*(100*vals[:,3]-100*vals[:,2])/(100*vals[:,2])
    spread = 100*(100*vals[:,1]-100*vals[:,0])/(100*vals[:,0])

    #FEATURES ARE
    '''
    adj -> % change in Adj Close over Close
    change -> % change in Close over Open
    spread -> % change in High over Low
    '''



    #print(adj.shape, change.shape, spread)
    x = np.vstack((adj, change, spread)).T

    #print(x.shape)
    #print(x.shape, x)

    samples = (x.shape[0] // (train_dist + label_delay))**2
    
    for _ in range(samples):
      ind = random.randint(0, x.shape[0]-train_dist-label_delay-1)
      sample = x[ind:ind+train_dist,:]
      label = x[ind+train_dist+label_delay,:]

      sample = sample.reshape((1,sample.shape[0], sample.shape[1]))
      label = label.reshape((1,1, label.shape[0]))

      #print(sample.shape, sample)
      X = np.vstack([X, sample])
      Y = np.vstack([Y, label])

  print('Fresh X: ', X.shape)
  print('Fresh Y: ', Y.shape)
  return X, Y
#get_inp_data()


