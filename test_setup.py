def get_modules(module_object):
    '''
    gets a list of modules without nn.Sequential groupings, as a list of strings
    '''
    modules_list = []
    print(module_object)
    for module in module_object.children():
        if isinstance(module, nn.Sequential):
            modules_list += get_modules(module)
        else:
            modules_list.append(str(module))
    return modules_list

def test_single_output(model, device, test_loader):
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