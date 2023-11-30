from bs4 import BeautifulSoup
import bs4
import argparse
from pathlib import Path
import os
from glob import glob
import re
import csv
from tqdm import tqdm


def validate_name(string: str):
    ''' Validate and remove any special characters that does not follow python variable name rules '''
    if not isinstance(string, str):
        return string

    pattern = r"^[^a-zA-Z_\[\]]+|[^a-zA-Z_0-9\[\]]+"

    return re.sub(pattern, "", string.strip())


def find_subsection_tag_by_text(paragraph: bs4.element.Tag | bs4.NavigableString, text: str):
    ''' Find the subsequent tag after a tag, located by the first tag's text '''
    def helper(tag, text: str):
        if not hasattr(tag, "text"):
            return False
    
        text = str(text).lower()
        return text.strip() in str(tag.text).lower()

    if not hasattr(paragraph, "find_all"):
        return None

    all_tags = paragraph.find_all(lambda tag: helper(tag, text))
    # find the smallest tag
    if len(all_tags) > 0:
        smallest_tag = min(all_tags, key=lambda t: len(t))

        # find the next block of tag, whcih should be the
        if isinstance(smallest_tag, bs4.element.Tag):
            return smallest_tag.find_next_sibling()


def parse_type(raw_return_type: str):
    ''' Parse the return type and simply it '''
    raw_return_type = str(raw_return_type)

    # match pattern for [] (e.g. "List[Tensor]")
    # in this case only the outer type is returned
    raw_return_first_type = raw_return_type.split('[')
    if len(raw_return_first_type) > 1:
        return validate_name(raw_return_first_type[0])
    
    # check if the return is a type of tensor
    # if so, unify it
    if "tensor" in raw_return_type.lower():
        return "Tensor"

    # check if it is "or" delimited. If so, return the first type
    or_delimited = raw_return_type.split("or")
    if len(or_delimited) > 1:
        return validate_name(or_delimited[0])

    # check if it is comma delimited. If so, return the first type
    comma_delimited = raw_return_type.split(',')
    if len(comma_delimited) > 1:
        return validate_name(comma_delimited[0])

    # else, validate and return
    return validate_name(raw_return_type)


def extract_parameters(parameters_html: bs4.element.Tag):
    ''' Extract the list of parameters for a function '''
    list_tags = parameters_html.find_all("li")
    if len(list_tags) == 0:
        # use the tag as itself
        list_tags = [parameters_html]

    param_names = list()
    param_list_of_types = list()
    optional_list = list()
    for list_tag in list_tags:
        if not hasattr(list_tag, "text"):
            continue

        param_description = str(list_tag.text).strip()
        
        is_optional = False
        param_types = list()
        try:
            # try regex matching
            # designed for pattern "param_name (type1 or type 2) - description"
            match = re.match(r"^([^ ]*) \((?:([^() ]*)(?: or ([^() ]*))?)\) – .*", param_description)
            param_name = validate_name(match.group(1))

            # parameter types should be group 2 - 3
            param_types.append(parse_type(match.group(2)))

            type_option_2 = match.group(3)
            if type_option_2 is not None:
                param_types.append(parse_type(type_option_2))

        except AttributeError:
            # no group was found. Try alternative comma separation method
            # for complext types
            param_and_types = param_description.split(" – ")[0]

            param_name = validate_name(param_and_types.split()[0])

            # find substring within bracket
            raw_parameter_type = param_and_types[param_and_types.find("(")+1:param_and_types.find(")")]
            raw_types = raw_parameter_type.split(',')
            for t in raw_types:
                t = t.strip()
                if t.startswith("or"):
                    t = t.lstrip("or")
                    t = t.strip()
                if t.lower() == "optional":
                    is_optional = True
                else:
                    param_types.append(parse_type(t))

        param_names.append(param_names)
        param_list_of_types.append(param_types)
        optional_list.append(is_optional)

    return param_names, param_list_of_types, optional_list


def try_find_function(section: bs4.element.Tag):
    ''' Try to find the function name, the parameters list and the return type. Defaults to no parameters and none type return. '''

    h1 = section.find("h1")
    if hasattr(h1, "find"):
        function_name = h1.find(string=True, recursive=False)
    else:
        return None, list(), None

    details_paragraph = section.find("dl")
    parameters_html = find_subsection_tag_by_text(details_paragraph, "Parameters")
    keyword_html = find_subsection_tag_by_text(details_paragraph, "Keyword Arguments")
    return_type_html = find_subsection_tag_by_text(details_paragraph, "Return type")

    # find list of parameters and its type
    if isinstance(parameters_html, bs4.element.Tag):
        param_names, param_list_of_types, optional_list = extract_parameters(parameters_html)
    else:
        param_names, param_list_of_types, optional_list = list(), list(), list()

    # find list of keyword arguments and its type
    # merge it with parameter list
    if isinstance(keyword_html, bs4.element.Tag):
        kw_names, kw_list_of_types, kw_optional_list = extract_parameters(keyword_html)
        param_names += kw_names
        param_list_of_types += kw_list_of_types
        optional_list += kw_optional_list

    # find return type
    return_type_name = "None"
    if hasattr(return_type_html, "find"):
        if hasattr(return_type_html, "text"):
            return_type_name = parse_type(str(return_type_html.text).strip())
    else:
        # try to extract the function name using the arrow
        # using the function type string. e.g,: " torch.select(input, dim, index) → Tensor"
        return_type_tag = find_subsection_tag_by_text(section, "→")
        if hasattr(return_type_tag, "text"):
            return_type_name = parse_type(str(return_type_tag.text).strip())

    assert len(param_names) == len(param_list_of_types) == len(optional_list)
    return function_name, param_names, param_list_of_types, optional_list, return_type_name


def get_function(content: str):
    ''' helper function for getting list of functions for each html file '''
    soup = BeautifulSoup(content, "html.parser")

    sections = soup.find_all("div", {"class": "section"})

    functions = list()
    for s in sections:
        function_name, param_names, param_list_of_types, optional_list, return_type_name = try_find_function(s)

        if function_name is not None:
            functions.append({
                "function_name": function_name,
                "param_names": param_names,
                "param_types": param_list_of_types,
                "is_optional": optional_list,
                "return_type": return_type_name
            })

    return functions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("html_loc", help="location of the html files, or a single html file")
    parser.add_argument("-o", "--output_file", help="output file location", default="output.csv")
    args = parser.parse_args()

    path = Path(args.html_loc)

    if not path.exists():
        parser.error("Unable to locate path: " + str(path))
    

    content_list = list()
    if os.path.isfile(path):
        with open(path.absolute(), 'r', encoding="utf-8-sig") as f:
            content_list.append(f.read())
    else:
        path = os.path.join(path.absolute(), '') # append slash automatically
        path = os.path.join(path, "*.html")
        for file_path in glob(path):
            with open(file_path, 'r', encoding="utf-8-sig") as f:
                content_list.append(f.read())
    
    functions = list()
    for content in tqdm(content_list, desc="Scraping functions from HTML"):
        functions += get_function(content)

    print("Writing to output...")
    with open(args.output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=("function_name", "param_names", "param_types", "is_optional","return_type"))
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
