from setup import *


class ConvNetwork(nn.Module):
    def __init__(self, train_len, features, representation_size = 10):
        super(ConvNetwork, self).__init__()
        self.representation_size = representation_size
        self.train_len = train_len
        self.features = features

        # self.c = nn.Conv2d(1, representation_size, kernel_size = (5,3), padding=(1, 0)).double()
        # self.f = nn.ReLU()
        # self.z = nn.Linear(representation_size*train_len, 3).double()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, representation_size, kernel_size = (features,features), padding=(1, 0)).double()
            ,
            nn.ReLU()
        )

        self.lin_layers = nn.Sequential(
            nn.Linear(representation_size*train_len, features).double()
            ,
        )
        

    def forward(self, x):
        '''
        The forward pass defines how to process an input x. This implicitly sets how
        the modules saved in __init__() should be chained together.

        Every PyTorch module has a forward() function, and when defining our own
        modules like we're doing here, we're required to define its forward()

        x: a single batch of input data
        '''
        t = self.conv_layers(x)
        t = t.view(-1, self.representation_size*self.train_len)
        t = self.lin_layers(t)
        return t.view(-1, 1, t.shape[1])