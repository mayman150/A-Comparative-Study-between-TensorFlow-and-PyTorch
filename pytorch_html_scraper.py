from bs4 import BeautifulSoup
import bs4
import argparse
from pathlib import Path
import os
from glob import glob
import re


def find_subsection_tag_by_text(paragraph: bs4.element.Tag | bs4.NavigableString, text: str):
    ''' Find the subsequent tag after a tag, located by the first tag's text '''
    def helper(tag, text: str):
        if not hasattr(tag, "text"):
            return False
    
        text = str(text).lower()
        return text.strip() in str(tag.text).lower()

    all_tags = paragraph.find_all(lambda tag: helper(tag, text))
    # find the smallest tag
    if len(all_tags) > 0:
        smallest_tag = min(all_tags, key=lambda t: len(t))

        # find the next block of tag, whcih should be the
        if isinstance(smallest_tag, bs4.element.Tag):
            return smallest_tag.find_next_sibling()


def extract_parameters(parameters_html: bs4.element.Tag):
    ''' Extract the list of parameters for a function '''
    list_tags = parameters_html.find_all("li")
    if len(list_tags) == 0:
        # use the tag as itself
        list_tags = [parameters_html]

    param_list = list()
    for list_tag in list_tags:
        if not hasattr(list_tag, "text"):
            continue

        param_description = str(list_tag.text)
        
        is_optional = False
        param_types = list()
        try:
            # try regex matching
            # designed for pattern "param_name (type1 or type 2) - description"
            match = re.match(r"^([^ ]*) \((?:([^() ]*)(?: or ([^() ]*))?)\) – .*", param_description)
            param_name = match.group(1)

            # parameter types should be group 2 - 3
            param_types.append(match.group(2))

            if type_option_2 := match.group(3) is not None:
                param_types.append(type_option_2)

        except AttributeError:
            # no group was found. Try alternative comma separation method
            # for complext types
            param_and_types = param_description.split(" – ")[0]

            param_name = param_and_types.split()[0]

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
                    param_types.append(t)

        param_list.append({
            "param_name": param_name,
            "param_type": param_types,
            "is_optional": is_optional
        })

    return param_list


def try_find_function(section: bs4.element.Tag):
    ''' Try to find the function name, the parameters list and the return type. Defaults to no parameters and none type return. '''

    function_name = section.find("h1").find(string=True, recursive=False)

    details_paragraph = section.find("dl", {"class": "py function"})
    parameters_html = find_subsection_tag_by_text(details_paragraph, "Parameters")
    keyword_html = find_subsection_tag_by_text(details_paragraph, "Keyword Arguments")
    return_type_html = find_subsection_tag_by_text(details_paragraph, "Return type")

    # find list of parameters and its type
    if isinstance(parameters_html, bs4.element.Tag):
        param_list = extract_parameters(parameters_html)
    else:
        param_list = list()

    # find list of keyword arguments and its type
    # merge it with parameter list
    if isinstance(keyword_html, bs4.element.Tag):
        kw_args = extract_parameters(keyword_html)
        param_list += kw_args

    # find return type
    return_type_name = "None"
    if hasattr(return_type_html, "find"):
        type_tag = return_type_html.find("em")
        
        if hasattr(type_tag, "text"):
            return_type_name = type_tag.text
    else:
        # try to extract the function name using the arrow
        # using the function type string. e.g,: " torch.select(input, dim, index) → Tensor"
        return_type_tag = find_subsection_tag_by_text(section, "→")
        if hasattr(return_type_tag, "text"):
            return_type_name = return_type_tag.text

    return function_name, param_list, return_type_name


def get_function(content: str):
    soup = BeautifulSoup(content, "html.parser")

    sections = soup.find_all("div", {"class": "section"})

    for s in sections:
        function_name, param_list, return_type_name = try_find_function(s)

        print(function_name)
        print(param_list)
        print(return_type_name)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("html_loc", help="location of the html files, or a single html file")

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
    
    for content in content_list:
        get_function(content)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
