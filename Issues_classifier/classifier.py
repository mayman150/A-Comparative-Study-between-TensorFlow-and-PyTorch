'''
Two approaches for classification: 
(1): Word-based Approach [Comparing the needed word] 
(2): Prediction Approach [Using differnt Models: Naive Base, Logistic Regression]
'''

import pandas as pd
import numpy as np 



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

def predictionBasedChecker(Input, Output, Model):
    '''
    Input: IssueTitle, IssueBody, Tag

    Output: True/False
    '''
    # Load the model
    # Load the data
    # Predict the output
    # Return the output
    return False



