import setup


def print_t(*args):
  out = []
  for t in args:
    numpy_array = t.cpu().numpy().T

    # Convert the NumPy array to a Python list
    python_list = numpy_array.tolist()

    # Print the Python list
    out.append([*zip(*python_list)])

  for i in zip(*out):
    print(i)


def visualize_test(model, device, test_loader):
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

            loss_function = torch.nn.MSELoss()
            test_loss += loss_function(output,Y)
            print_t(input,Y,output)

