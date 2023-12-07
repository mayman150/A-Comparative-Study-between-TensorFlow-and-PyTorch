import ast
from collections import defaultdict
import json
import os
from utils import parse_file, gather_asts, merge_asts, get_unique_calls
class FuncCallVisitor(ast.NodeVisitor):
    '''
    Class to visit function calls in a Python file.
    '''
    def __init__(self):
        '''
        Initialize the class.

        Attributes:
        - _name (list): List to store the parts of the function name.
        - _calls (list): List to store function calls.
        '''
        self._name = []
        self._calls = []

    @property
    def name(self):
        '''
        Get the name of the function.

        Returns:
        - str: The name of the function as a dot-separated string.
        '''
        return '.'.join(self._name)

    @property
    def calls(self):
        '''
        Get the function calls.

        Returns:
        - list: List of function calls.
        '''
        return self._calls

    @name.deleter
    def name(self):
        '''
        Clear the name of the function.
        '''
        self._name.clear()

    @calls.deleter
    def calls(self):
        '''
        Clear the function calls.
        '''
        self._calls.clear()

    def visit_Name(self, node):
        '''
        Visit a Name node.

        Args:
        - node (ast.Name): Name node.

        Returns:
        - None
        '''
        self._name.append(node.id)

    def visit_Attribute(self, node):
        '''
        Visit an Attribute node.

        Args:
        - node (ast.Attribute): Attribute node.

        Returns:
        - None
        '''
        try:
            self._name.append(node.value.id)
            self._name.append(node.attr)
        except AttributeError:
            # If the Attribute node does not have 'value' or 'attr' attributes,
            # visit its children nodes.
            self.generic_visit(node)

    def visit_Call(self, node):
        '''
        Visit a Call node.

        Args:
        - node (ast.Call): Call node.

        Returns:
        - None
        '''
        # Create a new instance of FuncCallVisitor to visit the function node
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)
        
        # Append the function name and its calls to the list
        self._calls.append(callvisitor.name)
        self._calls.extend(callvisitor.calls)


def get_class_calls(tree):
    """
    Get function calls within class definitions.

    Args:
        tree (ast.Module): AST of the code.

    Returns:
        defaultdict: Dictionary containing class names as keys and function dictionaries as values.
    """
    class_calls = defaultdict(dict)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            func_calls = defaultdict(list)
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.FunctionDef):
                    func_name = subnode.name
                    callvisitor = FuncCallVisitor()
                    callvisitor.visit(subnode)
                    func_calls[func_name].extend(callvisitor.calls)
            class_calls[class_name] = func_calls
    return class_calls


def get_func_calls(tree):
    """
    Get function calls outside class definitions.

    Args:
        tree (ast.Module): AST of the code.

    Returns:
        defaultdict: Dictionary containing function names as keys and a list of function calls as values.
    """
    func_calls = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node)
            func_calls[func_name].extend(callvisitor.calls)
    return func_calls



def process_files(args):
    """
    Process files and print unique API calls.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    asts = gather_asts(args.data_paths)
    combined_tree = merge_asts(asts)

    json_file = args.json_paths + args.type + '_api_calls.json'
    DICT_ = json.load(open(json_file))
    LIST = DICT_['api_calls']
    dct_class = get_class_calls(combined_tree)
    dct_func = get_func_calls(combined_tree)
    print(get_unique_calls(dct_class, dct_func, LIST))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', type=str, default='../Data/Models/TensorFlow/CV/Inception.py')
    parser.add_argument('--json_paths', type=str, default='../Data/json_packages/')
    parser.add_argument('--type', type=str, default='tf')
    args = parser.parse_args()

    process_files(args)
