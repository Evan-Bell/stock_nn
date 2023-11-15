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

def get_inp_data_tickers(train_dist, label_delay, n_inputs=2, n_outputs=1, tickers = ['AAPL']):
  X = np.empty((0, train_dist, n_inputs))
  Y = np.empty((0, 1, n_outputs))

  total_points_in_data = 0

  # torch.zeros(1, train_dist, n_inputs)
  # torch.zeros(1, 1, n_outputs)

  for ticker in tickers:
    df = yf.download(ticker, period='max', threads=True)
    x = 100*np.array(df[['Low', 'High', 'Open', 'Close', "Volume", "Adj Close"]].values)

    total_points_in_data += x.shape[0]
    print(ticker, ": ", x.shape[0])

    samples = int((x.shape[0] // (train_dist + label_delay))**1.5)
    #max(0, 2*(x.shape[0] - (train_dist + label_delay)) )
    
    #temporary matrices to store this ticker batch of data
    tX = np.zeros((samples, train_dist, n_inputs))
    tY = np.zeros((samples, 1, n_outputs))

    for _ in range(samples):
      ind = random.randint(0, x.shape[0]-train_dist-label_delay-1)
      
      sample = x[ind:ind+train_dist, :]
      #FEATURES ARE
      '''
      adj -> digits of Volume (log 10)
      change -> % change in Close over Open
      spread -> % change in High over Low
      '''
      low = sample[:,0]
      high = sample[:,1]
      openn = sample[:,2]
      close = sample[:,3]
      volume = sample[:,4]

      volume = np.where(volume > 0, 0.01*volume, 1)
      adj = np.log10(volume)
      change = 100*(close-openn)/openn
      spread = 100*(high-low)/low


      sample = np.vstack((adj, change, spread)).T
      sample = sample.reshape((1,sample.shape[0], sample.shape[1]))
      tX[_,:,:] = sample


      label = 100*(x[ind+train_dist+label_delay,3] - x[ind+train_dist, 3]) / x[ind+train_dist, 3]
      label = label.reshape((1,1,n_outputs))
      tY[_,:,:] = label

    X = np.vstack([X, tX])
    Y = np.vstack([Y, tY])

  print('Fresh X: ', X.shape)
  print('Fresh Y: ', Y.shape)
  print('Total points sampled: ', X.shape[0]*train_dist, ' / ', total_points_in_data)
  print('Sample datapoint', X[X.shape[0]//2, :, :])
  print( Y[X.shape[0]//2, :, :])
  return X, Y


# start = datetime.now() 
# get_inp_data_tickers(90,30, 3,1,['AAPL', 'MSFT', 'GOOGL', 'SPY', 'AMZN', 'ESGRP'])
# end = datetime.now()
# print(end - start)



