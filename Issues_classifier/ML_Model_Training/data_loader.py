import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
import torch

import re

def remove_extra_commas(input_string):
    # Use regular expression to replace multiple commas with a single comma
    cleaned_string = re.sub(',+', ',', input_string)
    return cleaned_string


def modify(input_string):
    input_string = input_string.replace('\n', '').replace(' ', ',')
    input_string = remove_extra_commas(input_string)
    if(input_string[1] == ','):
        input_string = input_string[0] + input_string[2:]
    input_string = ast.literal_eval(input_string.replace(' ',''))
    return input_string


def preprocess_data():
    data = pd.read_csv('GT_bert_concat_data_tf.csv')

    X = data['BERT Embedding'].apply(lambda x: modify(x))
    X = np.asarray(X.values.tolist(), dtype=np.float32)
    #Given data['Is Bug']make them 0 and 1
    data['Is Bug'] = data['Is Bug'].apply(lambda x: 1 if x == True else 0)
    #get it as numpy array
    y = np.asarray(data['Is Bug'], dtype=np.uint8)

    return X,y


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(n_splits=5, batch_size=32):
    '''
    Load the data from the csv file and return DataLoader instances for cross-validation
    '''
    X, y = preprocess_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()

    loaders = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X_tensor[train_index], X_tensor[test_index]
        y_train, y_test = y_tensor[train_index], y_tensor[test_index]

        # Create custom datasets
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loaders.append((train_loader, test_loader))

    return loaders



# def test():

#     # Example usage:
#     cross_validation_loaders = load_data(n_splits=5, batch_size=32)

#     # Iterate over the cross-validation loaders
#     for train_loader, test_loader in cross_validation_loaders:
#         for inputs, labels in train_loader:
#             # Your training logic here
#             pass

#         for inputs, labels in test_loader:
#             # Your testing/evaluation logic here
#             pass


# if __name__ == '__main__':
#     test()