import setup
import get_data

class StockDataset(Dataset):
    '''
    A custom dataset class to use with PyTorch's built-in dataloaders.
    This will make feeding images to our models much easier downstream.

    data: np.arrays downloaded from Keras' databases
    vectorize: if True, outputed image data will be (784,)
                   if False, outputed image data will be (28,28)
    '''
    def __init__(self, data, labels, vectorize=True):
        self.data = data
        self.labels = labels
        self.vectorize = vectorize

    def __getitem__(self, idx):
        image_data = self.data[idx, :]
        if self.vectorize:
            image_data = image_data.reshape((image_data.shape[0],))
        image_label = self.labels[idx]
        return image_data, image_label

    def __len__(self):
        return self.data.shape[0]

def creater_trainers(train_data, val_data, test_data, batch_size):
    # Create Dataset objects for each of our train/val/test sets
    train_dataset = StockDataset(*train_data)
    val_dataset = StockDataset(*val_data)
    test_dataset = StockDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Display dataloader info
    print("Created the following Dataloaders:")
    print(f"train_loader has {len(train_loader)} batches of training data")
    print(f"val_loader has {len(val_loader)} batches of validation data")
    print(f"test_loader has {len(test_loader)} batches of testing data")
    return train_loader, val_loader, test_loader