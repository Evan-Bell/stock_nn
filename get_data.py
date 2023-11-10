import setup

# torch.set_printoptions(threshold=5, precision=3, sci_mode=False, linewidth=200)

# x = torch.rand(1000, 1,1)
# print(x)


msft = yf.Ticker("MSFT")

# get historical market data
hist = msft.history(period="1d")
print(hist)

# show meta information about the history (requires history() to be called first)
msft.history_metadata
# show financials:
# - income statement
income = msft.income_stmt
msft.quarterly_income_stmt
# - balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet
# - cash flow statement
msft.cashflow
msft.quarterly_cashflow

# show holders
msft.major_holders
msft.institutional_holders
msft.mutualfund_holders

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
#msft.earnings_dates
# see `Ticker.get_income_stmt()` for more options

def get_inp_data():
    ticker_symbol = 'AAPL'

    # Define the start and end dates for your data retrieval
    start_date = "2010-04-28"
    end_date = "2019-05-29" #(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=day_diff)).strftime('%Y-%m-%d')

    # Retrieve historical data using yfinance
    data = yf.download(ticker_symbol, start=start_date, end=end_date)


    # Calculate the date for one month from now
    one_month_later_start = "2010-05-29"
    one_month_later_end = "2019-06-29"

    # Extract relevant columns and create a DataFrame
    target_data = data[['Low', 'High']]

    print(target_data)

    #['Close', 'Low', 'High', 'Open', 'Adj Close']]
    #print(data)

    # Retrieve the closing price one month ahead
    closing_price_one_month_later = yf.download(ticker_symbol, start=one_month_later_start, end=one_month_later_end)

    print(closing_price_one_month_later)
    closing_price_one_month_later = closing_price_one_month_later['Close'].values

    #print(closing_price_one_month_later)

    Y = np.where((np.array(closing_price_one_month_later) - np.array(data['Close'].values)) <= 0, 0.0, 1.0)
    #Y = np.array(closing_price_one_month_later)
    #NORMAL ABOVE
    Y = Y.reshape(Y.shape[0],1)
    X = np.array(target_data.values)



    print(X.shape,Y.shape)
    return X, Y


def getTimeShiftedInpData(shift_type, shift_amount):
  """
  Shift input data over by time amount
  """
  pass

def getDataFromNames(input_groups: list, output_groups: list):
  """
  Given the input and output groups, construct the data
  """
  pass
  return X,Y

def splitData(X, Y, training_ratio=0.8):
  """
  splits input data into training and testing
  """
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=training_ratio, shuffle=True)

  X_val, garbage1, Y_val, garbage2 = train_test_split(X, Y, train_size=0.4, shuffle=True)

  print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
  return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)

train_data, test_data, val_data = splitData(X, Y, 0.8)

