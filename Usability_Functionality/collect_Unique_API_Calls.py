import ast
from collections import defaultdict
import json
import os

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
    parser.add_argument('--data_paths', nargs='+', type=str, default='./Data/Models/TensorFlow/NLP/GNMT/')
    parser.add_argument('--json_paths', type=str, default='./Data/json_packages/')
    parser.add_argument('--type', type=str, default='tf')
    args = parser.parse_args()

    process_files(args)
