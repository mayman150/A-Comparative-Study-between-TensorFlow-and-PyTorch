import pypandoc
import argparse
from pathlib import Path
import os
from glob import glob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("markdown_docs", help="location of the markdown document(s). If a directory is supplied, it will iterate through all the markdown documents recursively")
    parser.add_argument("-o", "--output_file", help="output csv location", default="output.csv")

    args = parser.parse_args()
    path = Path(args.markdown_docs)

    if not path.exists():
        parser.error("Unable to locate path: " + str(path))
    
    content_list = list()
    if os.path.isfile(path):
        with open(path.absolute(), 'r', encoding="utf-8-sig") as f:
            content_list.append(f.read())
    else:
        path = os.path.join(path.absolute(), '') # append slash automatically
        path = os.path.join(path, "*.md")
        for file_path in glob(path, recursive=True):
            with open(file_path, 'r', encoding="utf-8-sig") as f:
                content_list.append(f.read())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
