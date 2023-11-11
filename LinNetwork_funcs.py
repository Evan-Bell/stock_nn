from setup import *


class SimpleNetwork(nn.Module):
    def __init__(self, input_size, representation_size, output_size):
        super(SimpleNetwork, self).__init__()
        self.representation_size = representation_size
        self.input_size = input_size
        self.output_size = output_size

        self.Z1 = nn.Linear(input_size, output_size).double().to(device)
        # self.F1 = nn.Tanh()
        # self.Z2 = nn.Linear(representation_size, output_size).double()

        # self.model = nn.Sequential(
        #     nn.Linear(input_size, representation_size).double(),
        #     nn.Tanh(),
        #     nn.Linear(representation_size, representation_size).double(),
        #     nn.ReLU(),
        #     nn.Linear(representation_size, representation_size).double(),
        #     nn.Sigmoid(),
        #     nn.Linear(representation_size, output_size).double()
        # )

    def forward(self, x):
        '''
        The forward pass defines how to process an input x. This implicitly sets how
        the modules saved in __init__() should be chained together.

        Every PyTorch module has a forward() function, and when defining our own
        modules like we're doing here, we're required to define its forward()

        x: a single batch of input data
        '''

        return self.Z1(x)


class UpDownNetwork(nn.Module):
    def __init__(self, input_size, representation_size, output_size):
        super(UpDownNetwork, self).__init__()
        self.representation_size = representation_size
        self.input_size = input_size
        self.output_size = output_size

        self.Z1 = nn.Linear(input_size, representation_size).double()
        self.F1 = nn.ReLU()
        self.Z2 = nn.Linear(representation_size, output_size).double()
        self.F2 = nn.Sigmoid()

        # self.model = nn.Sequential(
        #     nn.Linear(input_size, representation_size).double(),
        #     nn.Tanh(),
        #     nn.Linear(representation_size, representation_size).double(),
        #     nn.ReLU(),
        #     nn.Linear(representation_size, representation_size).double(),
        #     nn.Sigmoid(),
        #     nn.Linear(representation_size, output_size).double()
        # )

    def forward(self, x):
        '''
        The forward pass defines how to process an input x. This implicitly sets how
        the modules saved in __init__() should be chained together.

        Every PyTorch module has a forward() function, and when defining our own
        modules like we're doing here, we're required to define its forward()

        x: a single batch of input data
        '''

        return self.F2(self.Z2(self.F1(self.Z1(x))))