# This is mostly copied from [this answer on Stack Overflow](https://stackoverflow.com/a/49452109/3330552)
# (Thanks [Ryan Tuck](https://github.com/ryantuck)!)
# except that I've included the dependencies and set this up to be called from the command line.
#
# Example call from bash to split a single CSV into multiple 100 line CSVs:
#     python3 split_csv /path/to/my/file.csv /path/to/split_files my_split_file 100
#
# Warning: This doesn't have any validation! This will overwrite existing files if you're not careful.

import csv
import os
import sys
import pandas as pd
from pathlib import Path
import time
import argparse

csv.field_size_limit(sys.maxsize)

class WriteBuffer:
    def __init__(self, header: list, file_prefix: str, destination: str) -> None:
        self.__header = header
        self.__write_buffer = list()
        self.__file_prefix = file_prefix
        self.__destination = destination
        self.__file_write_counter = 0

    def append_row(self, row: dict):
        self.__write_buffer.append(row)
    
    def write_to_disk(self):
        target_filename = f'{self.__file_prefix}_{self.__file_write_counter}.csv'
        target_filepath = os.path.join(self.__destination, target_filename)
    
        with open(target_filepath, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.__header)
            writer.writeheader()
            for row in self.__write_buffer:
                try:
                    writer.writerow(row)
                except Exception as e:
                    import pprint
                    pp = pprint.PrettyPrinter(depth=2)
                    pp.pprint(row)
                    raise e
        
        self.__file_write_counter += 1
        self.__write_buffer.clear()

    def __len__(self):
        return len(self.__write_buffer)


def progressbar(it, count=None, prefix="Progress:", size=60, out=sys.stdout): # Python3.6+
    if count is None:
        count = len(it)
    else:
        assert isinstance(count, int)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def split_csv(source_filepath, dest_path, result_filename_prefix, row_limit):
    """
    Split a source CSV into multiple CSVs of equal numbers of records,
    except the last file.
    The initial file's header row will be included as a header row in each split
    file.
    Split files follow a zero-index sequential naming convention like so:
        `{result_filename_prefix}_0.csv`
    :param source_filepath {str}:
        File name (including full path) for the file to be split.
    :param dest_path {str}:
        Full path to the directory where the split files should be saved.
    :param result_filename_prefix {str}:
        File name to be used for the generated files.
        Example: If `my_split_file` is provided as the prefix, then a resulting
                 file might be named: `my_split_file_0.csv'
    :param row_limit {int}:
        Number of rows per file (header row is excluded from the row count).
    :return {NoneType}:
    """
    if row_limit <= 0:
        raise Exception('row_limit must be > 0')

    line_count = max(pd.read_csv(source_filepath).shape)
    
    with open(source_filepath, 'r', encoding="utf-8") as source:
        reader = csv.DictReader(x.replace('\0', '') for x in source)

        buffer = WriteBuffer(header=reader.fieldnames, file_prefix=result_filename_prefix, destination=dest_path)

        for row in progressbar(reader, line_count):
            buffer.append_row(row)

            if len(buffer) >= row_limit:
                buffer.write_to_disk()

        buffer.write_to_disk()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=True)

    parser.add_argument("source_filepath", help="file path of the source csv")
    parser.add_argument("dest_folder", help="file path of the destination folder")
    parser.add_argument("-f", "--filename_prefix", help="prefix of the generated csv file. Defaults to the original file name", required=False)
    parser.add_argument("-r", "--row_limit", help="number of rows per csv. Defaults to 1000", default=1000, type=int, required=False)

    args = parser.parse_args()

    SOURCE_FILEPATH = args.source_filepath
    DEST_PATH = args.dest_folder
    FILENAME_PREFIX = getattr(args, "filename_prefix") or Path(SOURCE_FILEPATH).name
    ROW_LIMIT = args.row_limit
    split_csv(SOURCE_FILEPATH, DEST_PATH, FILENAME_PREFIX, ROW_LIMIT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt")
        exit(0)
