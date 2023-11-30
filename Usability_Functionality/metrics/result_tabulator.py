from .utils import *
import argparse
import os
import pandas as pd
import numpy as np

np.seterr(divide='ignore')

def check_file_exists(parser: argparse.ArgumentParser, arg: str):
    if not os.path.isfile(arg):
        parser.error("Unable to locate file: " + str(arg))
    else:
        return arg

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("result_csv", help="Location of the result csv file", type=lambda arg: check_file_exists(parser, arg))
    args = parser.parse_args()

    result_table = pd.read_csv(args.result_csv)
    result_output_line = list()
    total_lines: int = result_table["line_count"].sum()
    total_rows: int = len(result_table.index)
    
    print("total_lines: " + str(total_lines))
    result_output_line.append(str(total_lines))
    result_table["weights"] = result_table["line_count"].div(total_lines)
    weighted_column_names = list()
    for c in result_table.columns:
        # skip predefined columns
        if c in ("line_count", "weights", "filename"):
            continue
        
        # calculate raw average
        raw_average = result_table[c].sum() / total_rows

        print("raw_average" + c + ":" + str(raw_average))
        weighted_column_name = c + "_$weighted_c$"
        weighted_column_names.append(weighted_column_name)
        # pre_calculate w_i ln x_i for geometric mean
        result_table[weighted_column_name] = result_table["weights"].mul(np.log(result_table[c]))

    result_table.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in weighted_column_names:
        # calculate geometric mean
        numerator = result_table[c].sum()
        denominator = result_table["weights"].sum()

        orig_column_name = str(c).rstrip("_$weighted_c$")
        result = np.exp(numerator / denominator)

        print("Geomean_" + orig_column_name + ": " + str(result))
        result_output_line.append(str(result))
    
    print("CSV (Geomean): " + ','.join(result_output_line))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)