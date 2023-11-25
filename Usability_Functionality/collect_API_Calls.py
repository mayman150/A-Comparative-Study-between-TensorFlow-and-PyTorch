import ast
from collections import defaultdict


class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = []
        self._calls = []

    @property
    def name(self):
        return '.'.join(self._name)

    @property
    def calls(self):
        return self._calls

    @name.deleter
    def name(self):
        self._name.clear()

    @calls.deleter
    def calls(self):
        self._calls.clear()

    def visit_Name(self, node):
        self._name.append(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.append(node.value.id)
            self._name.append(node.attr)
        except AttributeError:
            self.generic_visit(node)

    def visit_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)
        self._calls.append(callvisitor.name)
        self._calls.extend(callvisitor.calls)


def get_func_calls(tree):
    func_calls = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node)
            func_calls[func_name].extend(callvisitor.calls)

    return func_calls

def calculate_Number_API_Calls(func_calls):
    #Given a dictionary of function calls, calculate the number of API calls -> we want to specify the API calls of PyTorch 
    API_CALLS_DICT = {}
    counter = 0
    for key, value in func_calls.items():
        #for each value in the dictionary
        for i in range(len(value)):
            if value[i] in API_CALLS_DICT:
                API_CALLS_DICT[value[i]] += 1
            else:
                API_CALLS_DICT[value[i]] = 1
            counter += 1
    return API_CALLS_DICT, counter

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='test.py')
    args = parser.parse_args()
    tree = ast.parse(open(args.data_path).read())
    dct_= get_func_calls(tree)
    #print for me each key and value
    for key, value in dct_.items():
        print(key, value)
    #print the number of API calls
    API_CALLS_DICT, counter = calculate_Number_API_Calls(dct_)
    print(counter)
