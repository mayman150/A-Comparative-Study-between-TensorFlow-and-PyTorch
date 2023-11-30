'''
This is an implementation for he API Method Name Confusion Index (AMNCI) that we are going to use to evaluate the usability of the API
We relied on the Some structural measures of API usability with some modification to fit in our research purpose

S6: Not grouping conceptually similar API methods together
'''

import re
from collections import Counter
import numpy as np
import pandas as pd


def getSignificantKeywords(method_names, threshold=6):
    
    # Step 1: Remove common prefixes and suffixes
    common_prefixes_suffixes = ['torch', 'nn', 'functional', 'module', 'optim', 'autograd', 'data', 'utils', 'tf', 'layers', 'losses', 'optimizers', 'metrics', 'data', 'estimator', 'feature_column']
    # Remove common prefixes and suffixes and added it in the cleaned_names 
    cleaned_names = []
    for prefix_suffix in common_prefixes_suffixes:
        #add temp for each method_name make 
        for temp in method_names:
            X = temp.lower()
            if X.startswith(prefix_suffix):
                cleaned_names.append(temp[len(prefix_suffix):])
            if X.endswith(prefix_suffix):
                cleaned_names.append(temp[:-len(prefix_suffix)])
                
    # Step 2: Split what remains of each method name on the basis of commonly occurring connectors such
    # as ’_’, ’-’, ’by’, ’of’, ’and’, ’to’, and so on, and on the basis of case change as in ’CamelCase’.
    split_names = []
    for name in cleaned_names:
        split_names.extend(re.findall('[A-Z][^A-Z]*', name))

    cleaned_name = split_names
    split_names = []
    for name in cleaned_name:
        split_names.extend(re.split('_|-|by|of|and|to', name))

    # Step 3: Count occurrences of name fragments
    name_fragment_counts = Counter(split_names)

    # Step 4: Filter significant keywords based on the threshold
    significant_keywords = [fragment for fragment, count in name_fragment_counts.items() if count >= threshold]

    return significant_keywords


def compute_run_lengths(keyword, method_names):
    #method_names should be the same order existed in the documentation! 
    run_lengths = []
    current_run_length = 0

    for method_name in method_names:
        if keyword in method_name:
            current_run_length += 1
        else:
            if current_run_length > 0:
                run_lengths.append(current_run_length)
                current_run_length = 0

    if current_run_length > 0:
        run_lengths.append(current_run_length)

    return run_lengths
    

def compute_total_matching_methods(run_lengths):
    return sum(run_lengths)


def compute_AMGI(keyword, run_lengths, total_matching_methods):
    Rj = len(run_lengths)
    if(Rj == 1):
        return 1
    AMGI_sj = 1 - (Rj - 1) / (total_matching_methods - 1) if total_matching_methods > 1 else 0
    return AMGI_sj

def compute_average_AMGI(method_names):
    total_AMGI = 0
    # api_keywords = getSignificantKeywords(method_names)
    #just for method_names getting strip with dot and then split with dot and get the final one and add it in api_keywords
    api_keywords = []
    for i in range(len(method_names)):
        temp = method_names[i].strip('.')
        temp = temp.split('.')
        try:
            all = temp[-2] +'.'+temp[-1]
        except:
            all = temp[-1]
        api_keywords.append(all)
    res = 0
    for keyword in api_keywords:
        run_lengths = compute_run_lengths(keyword, api_keywords)
        if(len(run_lengths) > 1):
            res +=1
        total_matching_methods = compute_total_matching_methods(run_lengths)
        AMGI_sj = compute_AMGI(keyword, run_lengths, total_matching_methods)
        total_AMGI += AMGI_sj
    average_AMGI = total_AMGI / len(api_keywords) if len(api_keywords) > 0 else 0
    return average_AMGI




# def test_AMGI_calculation():
#     # Sample method names
#     method_names = [
#         'getUserInfo',
#         'setPassword',
#         'updateUserProfile',
#         'isUserAdmin',
#         'addUserRole',
#         'deleteUser'
#     ]
#     print(compute_average_AMGI(method_names))

# # Run the test function
# test_AMGI_calculation()
