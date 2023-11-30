'''
This is an implementation for this API Documentation Index Metric that we are going to use to evaluate the usability of the API
We relied on the Some structural measures of API usability with some modification to fit in our research purpose

'''
import numpy as np 
import pandas as pd


def ADI_per_method(length_documentation_words, thershold=50):
    
    if(length_documentation_words > thershold):
        return 1
    else:
        return length_documentation_words/thershold


def ADI_Methods(df):
    ADI = 0
    
    for i in range(len(df)):
        ADI += ADI_per_method(df['documentation_length'][i])
    
    return ADI/len(df)

