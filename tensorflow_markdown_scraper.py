import argparse
from pathlib import Path
import os
from glob import glob
import markdown
import bs4
from bs4 import BeautifulSoup
from itertools import product
from typing import Any, List
from pytorch_html_scraper import validate_name, count_documentation_length
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
import csv
from tqdm import tqdm
import re
from queue import PriorityQueue

nltk.download('averaged_perceptron_tagger')

# from stackoverflow: https://stackoverflow.com/questions/48393253/how-to-parse-table-with-rowspan-and-colspan
def table_to_2d(table_tag: bs4.element.Tag) -> List[List[str]]:
    rowspans = []  # track pending rowspans
    rows = table_tag.find_all('tr')

    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        # count columns (including spanned).
        # add active rowspans from preceding rows
        # we *ignore* the colspan value on the last cell, to prevent
        # creating 'phantom' columns with no actual cells, only extended
        # colspans. This is achieved by hardcoding the last cell width as 1. 
        # a colspan of 0 means “fill until the end” but can really only apply
        # to the last cell; ignore it elsewhere. 
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom. 
        rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans 
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            value = cell.get_text()
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                    rowspans[col + dcol] = rowspan
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table


def pytype_translator(noun_list: List[str]):
    ''' Translate common literal description to actual python type '''
    translations = {
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "string": "str",
        "none": "None",
        "dictionary": "dict",
    }

    cadidate_translations = PriorityQueue()
    # always put none type as possibility
    cadidate_translations.put((0, "None"))
    # prioritize first iteration
    first_iter = 1

    for word in noun_list:
        # fast path: prioritize tensor return
        if "tensor" in validate_name(word):
            return "tensor"

        word = str(word).lower()
        
        translate = translations.get(validate_name(word), word)

        # note: all weights are inverted because we want the highest weight first
        # if the word contains any letter, mark as medium cadidate translate
        if re.search('[a-zA-Z]', translate):
            cadidate_translations.put((-1, translate))
        
        # if the translated word is in default names, prioritize
        if translate in translations.keys():
            cadidate_translations.put((-2, translate))
                
        # if the original word is quoted (``), prioritize
        # if it is the first iteration, put it the absolute highest
        if word.startswith('`') and word.endswith('`'):
            cadidate_translations.put((-3 - first_iter, word))
            # reset first iteration counter
            first_iter = 0

    return cadidate_translations.get()[1]


def check_table_is_type(table: List[List[str]], type: str):
    ''' Check if table is a specific type by iterating through the first row of the table '''
    if len(table) > 1:
        row = table[0]
        for item in row:
            if str(type).lower() in str(item).lower():
                return True

    return False


def get_list_of_nouns(tags: list[tuple[Any, str]]):
    ''' Get the list of nouns in a tag list '''

    noun_list = list()

    for i in range(len(tags)):
        pair = tags[i]
        if len(pair) != 2:
            continue

        tag = str(pair[1])

        # get the previous tag and the next tag. Check if they are tagged as '``'
        prev_pair = tags[max(i - 1, 0)]
        next_pair = tags[min(i + 1, len(tags) - 1)]

        added_to_list = False
        if len(prev_pair) == 2 or len(next_pair) == 2:
            prev_tag = str(prev_pair[1]).strip()
            next_tag = str(next_pair[1]).strip()
            # if they are '``', treat it as noun
            if '`' in prev_tag and '`' in next_tag:
                noun_list.append(str('`' + pair[0] + '`').strip())
                added_to_list = True
        
        # check if it is verb
        if not added_to_list and tag.startswith("NN"):
            noun_list.append(str(pair[0]).strip())
    
    return noun_list


def find_type_from_desc(description: str):
    ''' Extract the type from the description. '''
    description = str(description).strip()

    # tokenisze
    tokenized = word_tokenize(description)
    
    # check if parameter is optional
    is_optional = False
    if "optional" in [str(token).strip().lower() for token in tokenized]:
        is_optional = True

    # tag
    tags = pos_tag(tokenized)
    # get list of nouns
    noun_list = get_list_of_nouns(tags)
    # only taking the first noung as the type
    if len(noun_list) >= 1:
        return pytype_translator(noun_list), is_optional
    else:
        return "None", is_optional


def find_table_of_type(soup: BeautifulSoup, type: str):
    ''' Find the table of a given header. Return the table as 2d list, discarding the header. Treat first row as header'''
    tables = soup.find_all("table")

    table = list()
    table_set = False
    for t in tables:
        parsed_table = table_to_2d(t)
        is_args = check_table_is_type(parsed_table, type)
        if is_args:
            # If multiple table of the same type is found, highly possible that this is a parent doc
            # need to raise exception
            if table_set:
                raise ValueError("Multiple table exists")
            # discard header
            table = parsed_table[1:]
            table_set = True

    return table


def load_md_file(path: Path):
    ''' Load the markdown file and convert to html for easier parsing '''
    with open(path, 'r') as f:
        html = markdown.markdown(f.read())
    return html


def get_function(content: str):
    soup = BeautifulSoup(content, "html.parser")
    
    # find funciton name
    function_tag = soup.find("h1")

    if hasattr(function_tag, "text"):
        function_name = str(function_tag.text)
    else:
        function_name = None

    # find args table:
    args = find_table_of_type(soup, "args")

    # find return table
    return_table = find_table_of_type(soup, "return")

    param_names = list()
    param_types = list()
    optional_list = list()
    for arg_row in args:
        if len(arg_row) != 2:
            continue

        param_name = validate_name(str(arg_row[0]).strip())
        param_desc = arg_row[1]

        param_type, is_optional = find_type_from_desc(param_desc)
        param_names.append(validate_name(param_name))
        param_types.append(validate_name(param_type))
        optional_list.append(is_optional)

    # get the return type
    # default to None
    return_type = "None"
    if len(return_table) > 0:
        return_statement = ' '.join(return_table[0])
        return_statement = str(return_statement).strip()
        
        return_type, _ = find_type_from_desc(return_statement)
        return_type = validate_name(return_type)

    # find documentation length
    # exclude "code" and "table" blocks as those has been accounted for.
    documentation_length = count_documentation_length(soup.extract("code").extract("table").text)

    return {
        "function_name": validate_name(function_name), 
        "param_names": param_names,
        "param_types": param_types, 
        "is_optional": optional_list, 
        "return_type": validate_name(return_type),
        "documentation_length": documentation_length
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("markdown_docs", help="location of the markdown document(s). If a directory is supplied, it will iterate through all the markdown documents recursively")
    parser.add_argument("-o", "--output_file", help="output csv location", default="output.csv")

    args = parser.parse_args()
    path = Path(args.markdown_docs)

    if not path.exists():
        parser.error("Unable to locate path: " + str(path))
    
    print("Obtaining all documentation files...")
    content_list = list()
    if os.path.isfile(path):
        content_list.append(load_md_file(path))
    else:
        path = os.path.join(path.absolute(), '') # append slash automatically
        path = os.path.join(path, "**/*.md")
        for file_path in glob(path, recursive=True):
            content_list.append(load_md_file(file_path))

    functions = list()
    for content in tqdm(content_list, desc="Scraping function from markdown:"):
        try:
            f = get_function(content)
            functions.append(f)
        except ValueError:
            # This will only be raised if the md contains multiple definitons
            # this indicates this fine is not a method only definition
            # discard
            continue
    
    print("Writing to output...")
    with open(args.output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=("function_name", "param_names", "param_types", "is_optional", "return_type", "documentation_length"))
        writer.writeheader()
        for f in functions:
            if len(f.get("param_names", list())) == 0 and str(f.get("return_type", None)).lower() == "none":
                continue
            writer.writerow(f)

    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
