import setup
import get_data
import handle_data

import train
import test_setup

import test_network_funcs
import LinNetwork_funcs
import training_funcs

X,Y = get_inp_data()
train_data, test_data, val_data = splitData(X, Y, 0.8)
train_loader, val_loader, test_loader = creater_trainers(train_data, test_data, val_data)

# check if running on CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a simple autoencoder
network = UpDownNetwork(2, 5, 1).to(device)
#SimpleNetwork(2, 5, 1).to(device)


#loss function
#loss_func = nn.MSELoss()
loss_func = nn.MSELoss()

run_network(batch_size = 50, epochs = 1000, print_out = True)

