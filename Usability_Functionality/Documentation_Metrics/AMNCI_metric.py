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
    #GET CF
    C = 0
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            m1 = method_names[i]
            m2 = method_names[j]
            if m1 == m2:
                C += 1
    
    AMNCI = 1 - C/len(method_names)
    
    return AMNCI


# # Sample data
# data = {
#     'Name_Method': ['getUserInfo', 'setPassword', 'updateUserProfile', 'isUserAdmin', 'addUserRole', 'deleteUser', 'getUserInfo'],
#     'Parameters': ['username', 'password', 'profileData', None, 'roleName', 'username', 'userId'],
#     'Return_Type': ['UserInfo', 'bool', 'bool', 'bool', 'UserRole', 'bool', 'UserInfo']
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Call the AMNCI function
# amnci_result = AMNCI(df)

# # Print the result
# print("AMNCI Result:", amnci_result)
