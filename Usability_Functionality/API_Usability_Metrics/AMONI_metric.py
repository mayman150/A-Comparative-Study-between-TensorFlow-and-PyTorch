'''
This is an implementation for this API Method Name Overload Index (AMONI) Metric that we are going to use to evaluate the usability of the API
We relied on the Some structural measures of API usability with some modification to fit in our research purpose

Measuring S1: Methods with similar names returning different types of values
'''

import numpy as np 
import pandas as pd 
from ast import literal_eval
import math

def AMONI_Each_G(Return_List, similar_methods_count):
    '''
    Given G_i, the group of methods with similar names, we calculate the AMONI for each group 
    '''
    total_Unique_Return_Types = len(set(Return_List))
    if total_Unique_Return_Types == 1: 
        return 1.0
    if similar_methods_count == 1: 
        if total_Unique_Return_Types <= 1: 
            return 1.0
    AMONI_Inverse = (total_Unique_Return_Types - 1)/(similar_methods_count-1)
    AMONI_ratio = 1 - AMONI_Inverse
    return AMONI_ratio
    
def AMONI(documentation_dataFrame):
    '''
    Given the documentation data frame, we calculate the AMONI for the whole API
    '''
    #documentation_dataFrame should have the following columns: Name_Method, Parameters, Return_Type
    temp_method_names= documentation_dataFrame['function_name'].tolist()
    #Check if the return type is a list or not
    Return_Types = documentation_dataFrame['return_type'].tolist()
    Method_Names = []

    for i in range(len(temp_method_names)):
        try:
            cleaned_method_name =  temp_method_names[i].split('.')[-2] +'.'+ temp_method_names[i].split('.')[-1] 
        except: 
            cleaned_method_name = temp_method_names[i]
        Method_Names.append(cleaned_method_name)
    
    Method_Return_Type = {}
    #Number of Methods with similar names
    Similar_Methods_Count = {}
    for i in range(len(Method_Names)):
        if Method_Names[i] not in Method_Return_Type:
            Method_Return_Type[Method_Names[i]] = [Return_Types[i]]
            Similar_Methods_Count[Method_Names[i]] = 1
        else:
            Method_Return_Type[Method_Names[i]].append(Return_Types[i])
            Similar_Methods_Count[Method_Names[i]] += 1
    #Get the AMONI for each group send int the total number of methods with similar names
    total_AMONI = 0
    for key in Method_Return_Type:
        total_AMONI +=AMONI_Each_G(Method_Return_Type[key], Similar_Methods_Count[key])
        
    return total_AMONI/len(Method_Return_Type)

    
    
    
#test the AMONI function 

# # Sample data
# data = {
#     'Name_Method': ['getUserInfo', 'getUserInfo', 'updateUserProfile', 'isUserAdmin', 'addUserRole', 'deleteUser', 'getUserInfo'],
#     'Parameters': ['username', 'password', 'profileData', None, 'roleName', 'username', 'userId'],
#     'Return_Type': ['UserInfo', 'X', 'bool', 'bool', 'UserRole', 'bool', 'UserInfo']
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Call the AMONI function
# amoni_result = AMONI(df)

# # Print the result
# print("AMONI Result:", amoni_result)
