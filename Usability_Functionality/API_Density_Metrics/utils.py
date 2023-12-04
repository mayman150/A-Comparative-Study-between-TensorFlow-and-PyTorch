import ast
from collections import defaultdict
import json
import os



def parse_file(file_path):
    """
    Parse a Python file and return its AST.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        ast.Module: AST of the code.
    """
    with open(file_path, 'r') as file:
        code = file.read()
        return ast.parse(code, filename=file_path)



def gather_asts(directory):
    """
    Gather ASTs of all Python files in a directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of ASTs.
    """
    asts = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)
            asts.append(parse_file(file_path))
    return asts


def merge_asts(asts):
    """
    Merge multiple ASTs into a single AST.

    Args:
        asts (list): List of ASTs.

    Returns:
        ast.Module: Merged AST.
    """
    # Create an empty Module node
    combined_ast = ast.Module(body=[])

    # Add the ASTs of individual files as children of the Module node
    for ast_node in asts:
        combined_ast.body.extend(ast_node.body)

    return combined_ast



def get_unique_calls_from_Func(dct_, LIST):
    """
    Get unique API calls from a dictionary of functions.

    Args:
        dct_ (dict): Dictionary containing function names as keys and a list of function calls as values.
        LIST (list): List of API calls to check for uniqueness.

    Returns:
        set: Set of unique API calls.
    """
    unique_calls = set()
    for func_name in dct_:
        for i in dct_[func_name]:
            if '.' in i:
                try:
                    before, after = i.split('.')
                except:
                    before, after = i.split('.')[:2]
                before = before.lower()
                after = after.lower()
                if before in LIST:
                    unique_calls.add(after)
            else:
                i = i.lower()
                if i in LIST:
                    unique_calls.add(i)
    return unique_calls


def get_unique_calls_from_class(dct_, LIST):
    """
    Get unique API calls from a dictionary of classes and their functions.

    Args:
        dct_ (dict): Dictionary containing class names as keys and function dictionaries as values.
        LIST (list): List of API calls to check for uniqueness.

    Returns:
        set: Set of unique API calls.
    """
    unique_calls = set()
    for class_name in dct_:
        func_dict = dct_[class_name]
        for key, value in func_dict.items():
            if len(value) > 0:
                for i in value:
                    if '.' in i:
                        try:
                            before, after = i.split('.')
                        except:
                            before, after = i.split('.')[:2]
                        before = before.lower()
                        after = after.lower()
                        after = after.replace('_', '')
                        after = after.replace(',', '')
                        if before in LIST:
                            unique_calls.add(after)
                    else:
                        i = i.lower().replace('_', '').replace(',', '')
                        if i in LIST:
                            unique_calls.add(i)
    return unique_calls


def get_unique_calls(dct_class, dct_func, LIST):
    """
    Get unique API calls from a dictionary of classes and functions.

    Args:
        dct_class (defaultdict): Dictionary containing class names as keys and function dictionaries as values.
        dct_func (defaultdict): Dictionary containing function names as keys and a list of function calls as values.
        LIST (list): List of API calls to check for uniqueness.

    Returns:
        tuple: Set of unique API calls and the count of unique API calls.
    """
    unique_calls_set_class = get_unique_calls_from_class(dct_class, LIST)
    unique_calls_set_Func = get_unique_calls_from_Func(dct_func, LIST)
    unique_calls = set()
    unique_calls.update(unique_calls_set_class)
    unique_calls.update(unique_calls_set_Func)
    return unique_calls, len(unique_calls)

