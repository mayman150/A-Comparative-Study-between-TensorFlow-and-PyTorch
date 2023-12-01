import ast
from itertools import combinations
import argparse
from pathlib import Path
from utils import *
import os
from glob import glob


def get_if_blocks(block: ast.If, syntax_blocks: set):
    syntax_blocks.add(block)
    for b in block.body:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)
    
    for b in block.orelse:
        if isinstance(b, ast.If):
            get_if_blocks(b, syntax_blocks)


def add_to_vocabs(token: str, vocabs: set):
    uscore_token_list = str(token).split('_')

    # next, split by camel case
    while len(uscore_token_list) > 0:
        uscore_token = uscore_token_list.pop()
        camel_tokens = camel_case_split(uscore_token)
        for c in camel_tokens:
            vocabs.add(c)


def get_vocabs(block: ast.If, vocabs: set):
    # dump ast and extract string block
    tree_dump = ast.dump(block)
    extracted_strings = re.findall(r'[\'"](.*?)[\'"]', tree_dump)
    for string in extracted_strings:
        if ' ' in str(string):
            continue
        add_to_vocabs(string, vocabs)


def compute_textual_coherence(code_snippet):
    # Parse the source code and build the Abstract Syntax Tree (AST)
    tree = ast.parse(code_snippet)
    # Extract all syntactic blocks (All branching statements)
    syntactic_blocks = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            get_if_blocks(node, syntactic_blocks)

    # get vocabs
    vocab_list = list()
    for block in syntactic_blocks:
        vocabs = set()
        get_vocabs(block, vocabs)
        vocab_list.append(vocabs)

    # print(vocab_list)
    vocab_overlap = []
    for pair in combinations(vocab_list,2):
        vocab_overlap.append(len(pair[0].intersection(pair[1])) / len(pair[0].union(pair[1])))
    
    if len(vocab_overlap) > 0:
        tc_max = max(vocab_overlap)
        tc_min = min(vocab_overlap)
        tc_avg = sum(vocab_overlap)/len(vocab_overlap)

        return {
            "tc_min": tc_min,
            "tc_max": tc_max,
            "tc_avg": tc_avg
        }
    else:
        return {
            "tc_min": 0,
            "tc_max": 0,
            "tc_avg": 0
        }

def get_file_weight(filename: str):
    non_blank_line_count = 0

    with open(filename) as f:
        for line in f:
            if line.strip():
                non_blank_line_count += 1
    
    return 1 / non_blank_line_count


def process_file(filename: str, output_filename: str):
    code_file = sanitize_file(filename)
    function_lines = get_function_def_lines(code_file, filename)
    
    weights: List[float] = list()
    statistics: List[Dict[str, float]] = list()

    no_of_functions = len(function_lines)
    total_line_count = count_file_lines(code_file)

    for code_snippet in extract_function_blocks(code_file, function_lines).values():
        line_count = count_file_lines(code_snippet)
        
        weight = get_weight(no_of_functions, total_line_count, line_count)
        stats = compute_textual_coherence(code_file)

        weights.append(weight)
        statistics.append(stats)
    
    w_avg = get_weighted_mean(weights, statistics)

    write_to_csv(filename, output_filename, total_line_count, w_avg)

# Example usage with the provided code snippet in python having if conditions
def main():
    parser = argparse.ArgumentParser(prog="TC calculator")
    parser.add_argument("source", help="Input python file(s) to be parsed")
    parser.add_argument("output", help="output csv file name")

    args = parser.parse_args()

    source = args.source
    if os.path.isfile(source):
        process_file(source, args.output)
    elif os.path.isdir(source):
        # glob python files recursively
        if source[-1] != '/':
            source += '/'
        path = source + "**/*.py"
        for file in glob(path, recursive=True):
            filepath = Path(file)
            print("Processing file: " + str(filepath)) 
            try:
                process_file(filepath, args.output)
            except:
                print("Unable to process file: " + str(filepath))
    else:
        parser.error("Source must be a file or a directory")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
