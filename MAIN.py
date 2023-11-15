from setup import *
from get_data import *
from handle_data import *

from test_setup import *

from test_network_funcs import * 
from LinearNetwork import * 
from ConvNetwork import *
from training_funcs import * 

from save_load_network import *



tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'AMZN', 'ESGRP']

print('starting')
batch_size = 200
learning_rate = 0.004
epochs = 2000
input_dims = 3
output_dims = 1

if  not True:
    epochs = 100
    batch_size = 30
    tickers = ['GOOGL']

train_inp_len = 60
inp_to_label_delay = 15

X,Y = get_inp_data_tickers(train_inp_len, inp_to_label_delay, input_dims, output_dims, tickers)

data_source = Data_obj(X,Y,(train_inp_len, input_dims))
data_source.make_loaders(batch_size)

#network = UpDownNetwork(2, 5, 1).to(device)
#SimpleNetwork(2, 5, 1).to(device)
network = ConvNetwork(train_inp_len, input_dims, output_dims, representation_size = 6).to(device)


#loss function
loss_func = nn.MSELoss().to(device)



# hist = []
# for b in [5,10,25,50,75,100,250,500,1000,1500,2000]:
#     torch.manual_seed(3787436)
#     network = ConvNetwork(train_inp_len, input_dims, output_dims, representation_size = 6).to(device)
#     network = run_network(data_source, network, loss_func, batch_size = b, epochs = epochs, learning_rate=learning_rate, show_graph = False, print_out = True)
#     test_loss = test(network, data_source.loaders[2], loss_func)
#     print('Batch size: ', b,' \tTESTING LOSS: ', test_loss)
#     hist.append((b,test_loss))

# print(hist)
# plt.plot(*zip(*hist))
# plt.title('Test loss')  # Set a title for the first plot
# plt.show()

# waste_time()


network = run_network(data_source, network, loss_func, batch_size = batch_size, epochs = epochs, learning_rate=learning_rate, show_graph = False)
tag = save_model(network)
temp_model = load_model(tag)

test_loss = test(temp_model, data_source.loaders[2], loss_func)
print('TESTING LOSS: ', test_loss)

visualize_test(temp_model, data_source.loaders[2])

