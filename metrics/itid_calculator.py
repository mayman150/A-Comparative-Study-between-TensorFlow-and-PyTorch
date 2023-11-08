import ast
import nltk
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from typing import Set, List, Dict
from utils import *
import argparse
import math
from glob import glob
import os


nltk.download("words")
nltk.download('wordnet')
dict_word_set = set(words.words())
lemmatizer = WordNetLemmatizer()
lemmatizer_pos_list = ('n', 'v', 'a', 'r', 's')

ITID = namedtuple("ITID", ["terms_in_dict", "total_terms"])

def try_lemmatize_word(token: str):
    # try to lemmatize the word and see if it appears in dictionary
    for pos in lemmatizer_pos_list:
        word = lemmatizer.lemmatize(token, pos=pos)
        if word in dict_word_set:
            return 1
    else:
        return 0


def get_ITID(id: str):
    # Identifier Terms in Dictionary Score in page 7 of paper
    # tokenize id, support camel case and underscore
    # first split by underscore
    uscore_token_list = str(id).split('_')
    
    # next, split by camel case
    token_set: Set[str] = set()
    while len(uscore_token_list) > 0:
        uscore_token = uscore_token_list.pop()
        camel_tokens = camel_case_split(uscore_token)
        token_set = token_set.union(camel_tokens)

    total_terms_count = len(token_set)
    in_dictionary_terms_count = 0
    for token in token_set:
        token = token.strip()
        if token in dict_word_set:
            in_dictionary_terms_count += 1
        else:
            in_dictionary_terms_count += try_lemmatize_word(token)
    
    return ITID(in_dictionary_terms_count, total_terms_count)   


def get_node_alias_name(node: ast.Import | ast.ImportFrom, whitelist_set: set, real_name_set: set):
    for alias in node.names:

        # add user defined "as" names into whitelist set
        # these naming are usually developer dependent, and therefore would not be analyzed.
        if hasattr(alias, "asname") and not string_is_none_or_whitespace(alias.asname):
            whitelist_set.add(alias.asname)

        # add real library name for analysis
        real_name_set.add(alias.name)


def get_call_name(func: ast.expr):
    # get the name of a ast.expr functiongi
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        return func.attr
    elif isinstance(func, ast.Subscript):
        return get_call_name(func.value)
    elif isinstance(func, ast.Call):
        return get_call_name(func.func)
    else:
        raise ValueError(ast.dump(func, indent=2))


def get_itid_score(itid: ITID):
    if itid.total_terms <= 0:
        return float("nan")
    
    score = itid.terms_in_dict / itid.total_terms
    return score

def itid_max(itid_scores: List[ITID]):
    max_score = 0
    for itid in itid_scores:
        score = get_itid_score(itid)
        if score > max_score:
            max_score = score
    
    return max_score

def itid_min(itid_scores: List[ITID]):
    min_score = 1
    for itid in itid_scores:
        score = get_itid_score(itid)
        if score < min_score:
            min_score = score
    
    return min_score

def itid_macro_avg(itid_scores: List[ITID]):
    score_list = []
    for itid in itid_scores:
        score = get_itid_score(itid)
        if not math.isnan(score):
            score_list.append(score)
    
    if len(score_list) > 0:
        return sum(score_list) / len(score_list)
    else:
        return 0


def itid_micro_avg(itid_scores: List[ITID]):
    num = 0
    den = 0
    for itid in itid_scores:
        num += itid.terms_in_dict
        den += itid.total_terms
    
    if den > 0:
        return num / den
    else:
        return 0

def get_itid_stats(code: str):
    tree = ast.parse(code)
    # print(ast.dump(tree, indent=2))
    library_names = set()
    library_alias = set()
    library_module_names = set()
    library_module_alias = set()

    itid_scores: List[ITID] = list()
    for node in ast.walk(tree): # BFS traversal
        if isinstance(node, ast.Import):
            # get library name whitelist and real name list
            get_node_alias_name(node=node, whitelist_set=library_alias, real_name_set=library_names)
        elif isinstance(node, ast.ImportFrom):
            # get imported from whitelist and real name list
            get_node_alias_name(node=node, whitelist_set=library_module_alias, real_name_set=library_module_names)
        elif isinstance(node, ast.Call):
            call_name = get_call_name(node.func)
            # get itid for non-aliased modules
            # modules with alias will be separatly evaluated
            if call_name not in library_module_alias:
                itid = get_ITID(call_name)
            
            # get itid for keywords if any
            for kw in node.keywords:
                kw_name = kw.arg
                kw_itid = get_ITID(kw_name)

                # aggregate the itid score with call name
                itid = ITID(terms_in_dict=itid.terms_in_dict + kw_itid.terms_in_dict, total_terms=itid.total_terms + kw_itid.total_terms)
            
            itid_scores.append(itid)
        elif isinstance(node, ast.Return):
            if node.value is None:
                continue
    
    # run ITID for library modules
    for m_name in library_module_names:
        itid_scores.append(get_ITID(m_name))
    
    return {
        "itid_min": itid_min(itid_scores),
        "itid_max": itid_max(itid_scores),
        "itid_macro_avg": itid_macro_avg(itid_scores),
        "itid_micro_avg": itid_micro_avg(itid_scores)
    }

def process_file(filename: str, output_filename: str):
    code_file = sanitize_file(filename)
    function_lines = get_function_def_lines(code_file, filename)

    # weights and statistics for aggregating results
    weights: List[float] = list()
    statistics: List[Dict[str, float]] = list()

    no_of_functions = len(function_lines)
    total_line_count = count_file_lines(code_file)
    for code_snippet in extract_function_blocks(code_file, function_lines).values():
        line_count = count_file_lines(code_snippet)

        weight = get_weight(no_of_functions, total_line_count, line_count)
        stats = get_itid_stats(code_snippet)

        weights.append(weight)
        statistics.append(stats)

    # get weighted average
    w_avg = get_weighted_mean(weights, statistics)

    write_to_csv(filename, output_filename, total_line_count, w_avg)

def main():
    parser = argparse.ArgumentParser(prog="ITID calculator")
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
        raise FileNotFoundError()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
