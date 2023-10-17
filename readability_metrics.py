import ast
from itertools import combinations
import argparse
from pathlib import Path
import re


def get_if_blocks(block: ast.If, syntax_blocks: set):
    syntax_blocks.add(block)
    for b in block.body:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)
    
    for b in block.orelse:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]


def add_to_vocabs(token: str, vocabs: set):
    uscore_token_list = str(token).split('_')

    # next, split by camel case
    while len(uscore_token_list) > 0:
        uscore_token = uscore_token_list.pop()
        camel_tokens = camel_case_split(uscore_token)
        for c in camel_tokens:
            vocabs.add(c)


def get_call_name(func: ast.expr, vocabs: set):
    if isinstance(func, ast.Name):
        add_to_vocabs(str(func.id), vocabs)
    elif isinstance(func, ast.Attribute):
        add_to_vocabs(str(func.attr), vocabs)
    elif isinstance(func, ast.Constant):
        add_to_vocabs(str(func.value), vocabs)
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
    vocab_overlap = []
    for pair in combinations(vocab_list,2):
        vocab_overlap.append(len(pair[0].intersection(pair[1])) / len(pair[0].union(pair[1])))
    
    tc_max = max(vocab_overlap)
    tc_min = min(vocab_overlap)
    tc_avg = sum(vocab_overlap)/len(vocab_overlap)

    return tc_max, tc_min, tc_avg

# Example usage with the provided code snippet in python having if conditions
def main():
    parser = argparse.ArgumentParser(prog="ITID calculator")
    parser.add_argument("filename")

    args = parser.parse_args()
    code_file = Path(args.filename).read_text()
    tc_mn, tc_mx, tc_avg = compute_textual_coherence(code_file)
    print("TC Min: ",tc_mn)
    print("TC Max: ", tc_mx)
    print("TC avg: ", tc_avg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
