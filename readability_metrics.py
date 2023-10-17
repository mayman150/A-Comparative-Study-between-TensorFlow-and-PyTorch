import ast
from itertools import combinations
import argparse
from pathlib import Path

def get_if_blocks(block: ast.If, syntax_blocks: set):
    syntax_blocks.add(block)
    for b in block.body:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)
    
    for b in block.orelse:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)


def get_call_name(func: ast.expr, vocabs: set):
    if isinstance(func, ast.Name):
        vocabs.add(str(func.id))
    elif isinstance(func, ast.Attribute):
        vocabs.add(str(func.attr))
    elif isinstance(func, ast.Constant):
        vocabs.add(str(func.value))
    elif isinstance(func, ast.Compare):
        get_call_name(func.left, vocabs)
        for c in func.comparators:
            get_call_name(c, vocabs)
    elif isinstance(func, ast.BoolOp):
        for v in func.values:
            get_call_name(v, vocabs)
    elif isinstance(func, ast.Subscript) or isinstance(func, ast.Return) or isinstance(func, ast.Expr):
        get_call_name(func.value, vocabs)
    elif isinstance(func, ast.Call):
        get_call_name(func.func, vocabs)
    elif isinstance(func, ast.If):
        get_vocabs(func, vocabs)
    else:
        print(func)
        raise ValueError(ast.dump(func))

def get_vocabs(block: ast.If, vocabs: set):
    for b in block.body:
        get_call_name(b, vocabs)
    
    for b in block.orelse:
        get_call_name(b, vocabs)

    get_call_name(block.test, vocabs)
    


def compute_textual_coherence(code_snippet):
    # Parse the source code and build the Abstract Syntax Tree (AST)
    tree = ast.parse(code_snippet)
    # print(ast.dump(tree, indent=2))
    # Extract all syntactic blocks (All branching statements)
    syntactic_blocks = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            get_if_blocks(node, syntactic_blocks)
       
    for b in syntactic_blocks:
        print(ast.dump(b, indent=2))
        print("----------------")
    
    # get vocabs
    vocab_list = list()
    for block in syntactic_blocks:
        vocabs = set()
        get_vocabs(block, vocabs)
        vocab_list.append(vocabs)

    print(vocab_list)
    
# Example usage with the provided code snippet in python having if conditions
def main():
    parser = argparse.ArgumentParser(prog="ITID calculator")
    parser.add_argument("filename")

    args = parser.parse_args()
    code_file = Path(args.filename).read_text()
    compute_textual_coherence(code_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
