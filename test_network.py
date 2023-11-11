from setup import *

def test_single_output(model, test_loader):
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
      for input, Y in test_loader:
          # send input image to GPU if using GPU
          input = input.to(device)
          Y = Y.to(device)

          # run input through our model
          output = model(input)
          print(input)
          print(output)
          return