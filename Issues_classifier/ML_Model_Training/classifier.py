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
        self.linear_1 = nn.Linear(input_size, 8)
        self.linear_2 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.sigmoid(self.linear_2(x))
        return x.view(-1)
        # return F.sigmoid(self.linear(x)).view(-1) # change output shape from [n_samples, 1] to [n_samples]




