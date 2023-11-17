import numpy as np 
import pandas as pd 
import torch


#accuracy
def accuracy(y_pred, y_true):
    '''
    Input: y_pred, y_true
    Output: Accuracy
    '''
    return np.mean((y_pred > 0.5).data.numpy().astype(int) == y_true.data.numpy().astype(int))


#Precision
def precision(y_pred, y_true):
    '''
    Input: y_pred, y_true
    Output: Precision
    '''
    y_pred_binary = torch.round(y_pred)
    true_positives = torch.sum((y_true == 1) & (y_pred_binary == 1)).float()
    false_positives = torch.sum((y_true == 0) & (y_pred_binary == 1)).float()
    # Avoid division by zero
    precision = true_positives / (true_positives + false_positives + 1e-9)
    return precision.item()  # Convert to Python float


#Recall
def recall(y_pred, y_true):
    '''
    Input: y_pred, y_true
    Output: Recall
    '''
    # Convert predictions to binary (1 if probability >= 0.5, else 0)
    y_pred_binary = torch.round(y_pred)

    # Calculate true positives (correctly predicted positives) and false negatives
    true_positives = torch.sum((y_true == 1) & (y_pred_binary == 1)).float()
    false_negatives = torch.sum((y_true == 1) & (y_pred_binary == 0)).float()
    # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    return recall.item()  # Convert to Python float

#F1 Score
def f1_score(y_pred, y_true):
    '''
    Input: y_pred, y_true
    Output: F1 Score
    '''
    # Calculate precision and recall using previously defined functions
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)

    # Avoid division by zero
    f1 = 2 * (p * r) / (p + r + 1e-9)
    return f1

