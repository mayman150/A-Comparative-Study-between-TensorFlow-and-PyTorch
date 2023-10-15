import ast
from pathlib import Path
import nltk
import re
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from typing import Set, List
import argparse

nltk.download("words")
nltk.download('wordnet')
dict_word_set = set(words.words())
lemmatizer = WordNetLemmatizer()
lemmatizer_pos_list = ('n', 'v', 'a', 'r', 's')

ITID = namedtuple("ITID", ["terms_in_dict", "total_terms"])

def string_is_none_or_whitespace(string: str):
    if string is None:
        return True
    
    assert isinstance(string, str)

    if string.strip() == "":
        return True
    else:
        return False


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]


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
    uscore_token_list = id.split('_')
    
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
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        return func.attr
    else:
        raise ValueError()
        

def get_ast(code: str):
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
    
    # run ITID for library modules
    for m_name in library_module_names:
        itid_scores.append(get_ITID(m_name))
    
    print(itid_scores)

def main():
    parser = argparse.ArgumentParser(prog="ITID calculator")
    parser.add_argument("filename")

    args = parser.parse_args()
    code_file = Path(args.filename).read_text()
    get_ast(code_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
