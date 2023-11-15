from setup import *
from get_data import *

def splitData(X, Y, training_ratio=0.8):
    """
    splits input data into training and testing
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=training_ratio, shuffle=True)

    X_val, garbage1, Y_val, garbage2 = train_test_split(X, Y, train_size=0.4, shuffle=True)

    print('\nSplitting Data at ratio: ' , training_ratio)
    print('X_train: {} \nY_train: {} \nX_test: {} \nY_test: {}'.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class StockDataset(Dataset):
    '''
    A custom dataset class to use with PyTorch's built-in dataloaders.
    This will make feeding images to our models much easier downstream.

    data: np.arrays downloaded from Keras' databases
    vectorize: if True, outputed image data will be (784,)
                   if False, outputed image data will be (28,28)
    '''
    def __init__(self, data, labels, dims):
        self.data = data
        self.labels = labels
        self.dims = dims

    def __getitem__(self, idx):
        image_data = self.data[idx, :]
        image_data = image_data.reshape((1,*self.dims))
        image_label = self.labels[idx]
        return image_data, image_label

    def __len__(self):
        return self.data.shape[0]

def create_trainers(train_data, val_data, test_data, batch_size, dims):
    # Create Dataset objects for each of our train/val/test sets
    train_dataset = StockDataset(*train_data, dims)
    val_dataset = StockDataset(*val_data, dims)
    test_dataset = StockDataset(*test_data, dims)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
    
    # Display dataloader info
    print("\nCreated the following Dataloaders:")
    print(f"train_loader has {len(train_loader)} batches of training data")
    print(f"val_loader has {len(val_loader)} batches of validation data")
    print(f"test_loader has {len(test_loader)} batches of testing data\n")
    return train_loader, val_loader, test_loader


class Data_obj:
    def __init__(self, X, Y, dims, batch_size = -1, train_split = -1):
        self.X = X
        self.Y = Y
        self.dims = dims
        self.batch_size = batch_size
        self.train_split = train_split
        self.data = None
        self.loaders = None

    def make_loaders(self, batch_size = 50, train_split = 0.8):
        if self.train_split != train_split:
            self.train_split = train_split
            self.data = splitData(self.X, self.Y, train_split)

        if self.batch_size != batch_size or self.train_split != train_split:
            self.batch_size = batch_size
            self.loaders = create_trainers(*self.data, batch_size, self.dims)
        return self.loaders
