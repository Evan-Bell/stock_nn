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
