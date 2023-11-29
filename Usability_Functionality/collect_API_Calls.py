import numpy as np 
import pandas as pd
import os
import ast
import argparse

class APIVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_names = []
        self.function_names = []
        self.function_content = []
        self.function_API_calls = {}

    def visit_ClassDef(self, node):
        self.class_names.append(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.function_names.append(node.name)
        self.function_content.append(ast.get_source_segment(node))
        self.function_API_calls[node.name] = 0
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
            # API call on 'self' (assumed to be a PyTorch module)
            self.function_API_calls[self.function_names[-1]] += 1
        self.generic_visit(node)

def count_API_calls_torch(code_content):
    '''
    We can use AST Tree to count the number of API calls
    Extract functions from the class and then count the number of API calls for each function
    '''
    function_names = []
    function_content = []
    function_API_calls = []
    
    tree = ast.parse(code_content)
    for node in ast.walk(tree):
        print(node)
        print(node.__dict__)
        print("children: " + str([x for x in ast.iter_child_nodes(node)]) + "\\n")
    
    # visitor = APIVisitor()
    # visitor.visit(tree)
    # total_API_calls = sum(visitor.function_API_calls.values())
    # print(total_API_calls)
    
def main():
    args = argparse.ArgumentParser()
    # args.add_argument('--data_path', type=str, default='Data/Models/PyTorch/CV/Inception.py')
    args.add_argument('--data_path', type=str, default='test.py')
    args = args.parse_args()
    code_content = open(args.data_path, 'r').read()
    count_API_calls_torch(code_content)

if __name__ == '__main__':
    main()

    
if __name__ == '__main__':
    main()