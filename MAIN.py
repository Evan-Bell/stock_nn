from setup import *
from get_data import *
from handle_data import *

from test_setup import *

from test_network_funcs import * 
from LinNetwork_funcs import * 
from ConvNetwork import *
from training_funcs import * 


# check if running on CPU or GPU

tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']

print('starting')
batch_size = 50
epochs = 500

if not True:
    epochs = 50
    tickers = ['GOOGL']

train_inp_len = 90
inp_to_label_delay = 30
features = 3

X,Y = get_inp_data_tickers(train_inp_len, inp_to_label_delay, tickers)
data_source = Data_obj(X,Y,(train_inp_len, features))

#network = UpDownNetwork(2, 5, 1).to(device)
#SimpleNetwork(2, 5, 1).to(device)
network = ConvNetwork(train_inp_len, features, 5).to(device)


#loss function
#loss_func = nn.MSELoss()
loss_func = nn.MSELoss().to(device)


run_network(data_source, network, loss_func, batch_size = batch_size, epochs = epochs, learning_rate=0.001, print_out = False)

visualize_test(network, data_source.loaders[2])

