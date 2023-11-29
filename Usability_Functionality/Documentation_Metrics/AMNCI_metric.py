'''
This is an implementation for he API Method Name Confusion Index (AMNCI) that we are going to use to evaluate the usability of the API
We relied on the Some structural measures of API usability with some modification to fit in our research purpose

Measuring S5:Existence of too many methods with nearly identical method names
'''

import numpy as np 
import pandas as pd

def clean_and_convert_to_uppercase(input_string):
    # Remove underscores
    cleaned_string = input_string.replace('_', '')

    # Remove numerical suffix
    if cleaned_string[-1].isdigit():
        # Find the index of the last non-digit character
        last_non_digit_index = next(i for i, char in enumerate(reversed(cleaned_string)) if not char.isdigit())
        
        # Remove numerical suffix
        cleaned_string = cleaned_string[:-last_non_digit_index]

    # Convert to uppercase
    uppercase_string = cleaned_string.upper()

    return uppercase_string



def AMNCI(df):

    # Get the set of all method names
    method_names = df['Name_Method'].tolist()
    method_names = [clean_and_convert_to_uppercase(method_name) for method_name in method_names]
    C = len(set(method_names))
    AMNCI = 1 - C/len(method_names)
    
    return AMNCI