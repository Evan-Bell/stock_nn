from setup import *

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


def visualize_test(model, test_loader):
    '''
    Function for testing our models. One call to test() runs through every
    datapoint in our dataset once.

    model: an instance of our model, in this assignment, this will be your autoencoder

    device: either "cpu" or "cuda:0", depending on if you're running with GPU support

    test_loader: the dataloader for the data to run the model on
    '''
    # set model to evaluation mode
    model.eval()

    # don’t track gradients in testing, since no backprop
    with torch.no_grad():
        # iterate through each test image
        for input, Y in test_loader:

            # send input image to GPU if using GPU
            input = input.to(device)
            Y = Y.to(device)

            # run input through our model
            output = model(input)

            print(input)
            com = torch.cat((Y,output), 1)
            com = com.type(torch.double)
            print(com)
            return

