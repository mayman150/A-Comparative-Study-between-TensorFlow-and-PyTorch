from collections import namedtuple
from typing import Dict, List
import ast
import re
import pandas as pd
from pathlib import Path


START_END = namedtuple("START_END", ["start", "end"])

def camel_case_split(identifier):
    ''' Split camel case into terms '''
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]


def get_weight(no_of_functions, total_line_count, code_line_count):
    ''' Get the weight of the file for weighted average '''
    return no_of_functions * (code_line_count / total_line_count)

def get_function_def_lines(code_file: str, filename: str):
    ''' Get the line range for each function definition '''
    function_line = dict()
    v = ListVisitor(function_line)
    code = ast.parse(code_file, filename=filename, mode="exec")
    v.visit(code)

    return function_line

class ListVisitor(ast.NodeVisitor):
    def __init__(self, function_line:  Dict[str, START_END], indent=4):
        self.cur = 0
        self.ind = indent
        self.function_line = function_line

    def visit_FunctionDef(self, node):
        # print(f"{'':>{self.cur}}Function {node.name} {node.lineno}->{node.end_lineno}")
        self.function_line[node.name] = START_END(node.lineno, node.end_lineno)
        self.cur += self.ind
        self.generic_visit(node)
        self.cur -= self.ind


def autoappend_newline(string: str):
    ''' append newline character if necessary '''
    if string[-1] != '\n':
        string += '\n'
    return string

def string_is_none_or_whitespace(string: str):
    ''' Check if the string is None, or contains whitespaces only '''
    if string is None:
        return True
    
    assert isinstance(string, str)

    if string.strip() == "":
        return True
    else:
        return False


def string_is_comment(string: str):
    ''' Check if a string is a Python comments '''
    if string is None:
        return True
    
    string = string.strip()
    if string.startswith("#"):
        return True
    else:
        return False
    

def sanitize_file(filename: str):
    ''' Remove whitespace and comments, expand tabs and append EOF when necessary '''
    with open(filename, encoding="utf-8") as f:
        sanitized_str = ""
        is_in_function_doc = False
        for line in f:
            # ignore white space and comments
            if string_is_none_or_whitespace(line) or string_is_comment(line):
                continue

            # in case the file is tab indented, expand tabs
            line = line.expandtabs(tabsize=4)

            if not is_in_function_doc:
                sanitized_str += line
    
    return autoappend_newline(sanitized_str)


def count_file_lines(code_file: str):
    ''' Count number of lines for the code file '''
    return code_file.count("\n")


def count_leading_space(code_line: str):
    ''' Count the number of leading white space for a string '''
    return len(code_line) - len(code_line.lstrip())


def extract_function_blocks(code_file: str,  function_lines: Dict[str, START_END]):
    ''' Extract each code block for each function '''
    function_block: Dict[str, str] = dict()
    max_line_count = count_file_lines(code_file)
    code_file_lines = code_file.split("\n")

    for function_name, line_range in function_lines.items():
        start = max(line_range.start - 1, 0)
        end = min(line_range.end, max_line_count)

        function_line_list = code_file_lines[start: end]

        # remove leading whitespace on the first line when joining the code
        # and remove all subsequent number of leading whitespace
        # otherwise indentation error would occure
        whitespace_c = count_leading_space(function_line_list[0])
        code_block = '\n'.join([l[whitespace_c:] for l in function_line_list])

        function_block[function_name] = autoappend_newline(code_block)

    return function_block    


def get_weighted_mean(weights: List[float], statistics: List[Dict[str, float]]):
    ''' Get weighted mean, given the list of weights and list of statistics '''
    assert len(weights) == len(statistics)
    assert len(weights) > 0

    field_names = tuple(statistics[0].keys())

    # calculate the first 1 / \sum{w_i}
    w = 1 / sum(weights)

    results: Dict[str, float] = dict()

    for fn in field_names:
        fn_result = 0
        for i in range(len(weights)):
            stat = statistics[i].get(fn, 0)
            w_i = weights[i]

            fn_result += stat * w_i

        results[fn] = w * fn_result

    return results

def write_to_csv(current_filepath: str, output_filepath: str, line_count: int, results: Dict[str, float]):
    ''' Write results to csv '''
    fieldnames = ["filename", "line_count"] + list(results.keys())
    file_exists = Path(output_filepath).exists()
    if file_exists:
        file = pd.read_csv(output_filepath, names=fieldnames)
    else:
        file = pd.DataFrame(columns=fieldnames)
    
    file.set_index("filename", inplace=True)

    row = dict()
    row["line_count"] = line_count
    row |= results

    file.loc[current_filepath] = row
    
    file.to_csv(output_filepath, header=not file_exists)
