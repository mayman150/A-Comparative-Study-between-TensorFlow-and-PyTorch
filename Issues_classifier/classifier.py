'''
Two approaches for classification: 
(1): Word-based Approach [Comparing the needed word] 
(2): Prediction Approach [Using differnt Models: Naive Base, Logistic Regression]
'''

import pandas as pd
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch.autograd import Variable

torch.manual_seed(1)

def wordBasedChecker(IssueTitle, IssueBody):
    '''
    Input: IssueTitle, IssueBody
    Output: True/False
    '''
    bug_keywords = {
    'error',
    'exception',
    'traceback',
    'crash',
    'issue',
    'problem',
    'unexpected',
    'incorrect',
    'not working',
    'failure',
    'flaw',
    'mistake',
    'fault',
    'glitch',
    'inconsistency',
    'abnormal',
    'unexpected behavior',
    'unhandled',
    'segmentation fault',
    'defect',
    'bug'
    }
    if any(word.lower() in IssueTitle for word in bug_keywords):
        return True
    if any(word.lower() in IssueBody for word in bug_keywords):
        return True
    return False




class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return F.sigmoid(self.linear(x)).view(-1) # change output shape from [n_samples, 1] to [n_samples]




def training_loop(model, epochs=1500):
    criterion = torch.nn.BCELoss()
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters())
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(epochs):
        
        loss = criterion(model(X_train), y_train)
        val_loss = criterion(model(X_test), y_test)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history["loss"].append(loss.data[0])
        history["val_loss"].append(val_loss.data[0])
        history["acc"].append(accuracy(model(X_train), y_train))
        history["val_acc"].append(accuracy(model(X_test), y_test))
    


#TODO: Just added other accuracies: F1-Score, IOU, Precision, Recall
#TODO: Add the plot for the accuracies
#TODO: Implement Naive Bayes
#TODO: Having a function to gather all of them.
# def predictionBasedChecker(Input, Output, Model):
#     '''
#     Input: IssueTitle, IssueBody, Tag

#     Output: True/False
#     '''

#     # Load the model
#     # Load the data
#     # Predict the output
#     # Return the output
#     return False



