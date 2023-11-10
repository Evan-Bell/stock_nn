def train(model, device, train_loader, optimizer, val_loader=None, loss_func = nn.MSELoss()):
    '''
    Function for training our networks. One call to train() performs a single
    epoch for training.

    model: an instance of our model, in this assignment, this will be your autoencoder

    device: either "cpu" or "cuda", depending on if you're running with GPU support

    train_loader: the dataloader for the training set

    optimizer: optimizer used for training (the optimizer implements SGD)

    val_loader: (optional) validation set to include
    '''

    # Set the model to training mode.
    model.train()

    #we'll keep adding the loss of each batch to total_loss, so we can calculate
    #the average loss at the end of the epoch.
    total_loss = 0

    # We'll iterate through each batch. One call of train() trains for 1 epoch.
    # batch_idx: an integer representing which batch number we're on
    # input: a pytorch tensor representing a batch of input images.
    for batch_idx, (input, Y) in enumerate(train_loader):

        # This line sends data to GPU if you're using a GPU
        input = input.to(device)
        Y = Y.to(device)


        # Zero out gradients from previous iteration
        optimizer.zero_grad()

        # feed our input through the network
        output = model.forward(input)

        loss_function = loss_func
        loss_value = loss_function(output,Y)

        # Perform backprop
        loss_value.backward()
        optimizer.step()

        #accumulate loss to later calculate the average
        total_loss += loss_value

    return total_loss.item()/len(train_loader)



def test(model, device, test_loader, loss_func = nn.MSELoss()):
    '''
    Function for testing our models. One call to test() runs through every
    datapoint in our dataset once.

    model: an instance of our model, in this assignment, this will be your autoencoder

    device: either "cpu" or "cuda:0", depending on if you're running with GPU support

    test_loader: the dataloader for the data to run the model on
    '''
    # set model to evaluation mode
    model.eval()

    # we'll keep track of total loss to calculate the average later
    test_loss = 0

    # don’t track gradients in testing, since no backprop
    with torch.no_grad():
        # iterate through each test image
        for input, Y in test_loader:

            # send input image to GPU if using GPU
            input = input.to(device)
            Y = Y.to(device)

            # run input through our model
            output = model(input)

            loss_function = loss_func
            test_loss += loss_function(output,Y)

    # calculate average loss per batch
    test_loss /= len(test_loader)
    return test_loss.item()


def run_network(representation_size = 5, epochs = 200, batch_size = 50, learning_rate = 0.0001, print_out=True, show_graph = True):

  #remake data into batches
  train_loader, val_loader, test_loader = make_loaders(batch_size)

  # We'll use the Adam optimization of gradient descent.
  optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)


  # Hold the best model
  best_mse = np.inf   # init to infinity
  best_weights = None
  train_history = []
  test_history = []


  for epoch in range(1, epochs+1):
      train_loss = train(network, device, train_loader, optimizer, loss_func = loss_func)
      val_loss = test(network, device, val_loader, loss_func = loss_func)
      train_history.append(train_loss)
      test_history.append(train_loss)
      if val_loss < best_mse:
          best_mse = val_loss
          best_weights = copy.deepcopy(network.state_dict())
      if epoch % 50 == 0 and print_out:
        print('Train Epoch: {:02d} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))
        test_single_output(network, device, val_loader)

  # restore model and return best accuracy
  network.load_state_dict(best_weights)

  if print_out:
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

  if show_graph:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))  # Adjust the figure size as needed

    # Plot the data in the first subplot
    ax1.plot(train_history)
    ax1.set_title('Train loss')  # Set a title for the first plot

    # Plot the data in the second subplot
    ax2.plot(test_history)
    ax2.set_title('Test loss')  # Set a title for the second plot

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots side by side
    plt.show()
  return test(network, device, test_loader)