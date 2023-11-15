from setup import *
from ConvNetwork import *
from LinearNetwork import *

directory = './Saved_networks'

def all_models():
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.pt'):
                network_path_pt = directory + '/' + filename
                network_path_weights = (directory + '/' + filename)[:-3]
                model = torch.jit.load(network_path_pt)
                model.load_state_dict(torch.load(network_path_weights))
                for param_tensor in model.state_dict():
                    print('\n',filename, param_tensor, "\t", model.state_dict()[param_tensor].size())

def load_model(name):
    print('Loading model from: ', name)
    network_path_pt = directory + '/' + name + ".pt"
    network_path_weights = directory + '/' + name
    model = torch.jit.load(network_path_pt)
    model.load_state_dict(torch.load(network_path_weights))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model = model.to(device)
    return model

def save_model(model, dat = True):
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%H_%M_%S__%d_%m_%Y")
    print("File saved to: ", dt_string)
    torch.save(model.state_dict(), "./Saved_networks/" + dt_string)

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save("./Saved_networks/" + dt_string + '.pt') # Save
    return dt_string
