from utils import *
import argparse
import os
import pandas as pd
import numpy as np
from glob import glob
import csv
from operator import itemgetter


np.seterr(divide='ignore')

def check_folder_exists(parser: argparse.ArgumentParser, arg: str):
    if not os.path.isdir(arg):
        parser.error("Unable to locate directory: " + str(arg))
    else:
        return Path(arg)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--metric_type", choices=["tc", "itid"], required=True, help="Choose the type of metric to run. Can be either 'tc' for Textual Coherance, or 'itid' for Identifier Terms in Dictionary.")
    parser.add_argument("-l", "--library_type", choices=["torch", "tf"], required=True, help="Choose the library. Can be either 'tf' for TensorFlow, or 'torch' for PyTorch.")
    parser.add_argument("result_dir", help="Location of the directory containint result csv file", type=lambda arg: check_folder_exists(parser, arg))
    parser.add_argument("-o", "--output_dir", help="Location of the output directory. Will try to auto create directory if not exists", type=Path, default=Path("./"))
    args = parser.parse_args()

    # glob all csv files in directory
    glob_cmd = args.result_dir / (args.metric_type + "_" + args.library_type + "_*.csv")

    print("Getting all csv with header: " + str(args.metric_type + "_" + args.library_type + "_*.csv"))
    output = list()
    for csv_file in glob(str(glob_cmd)):
        row = calculate_geomean(Path(csv_file))
        output.append(row)

    output_file: Path = args.output_dir / Path(args.metric_type + "_" + args.library_type + "_output.csv")

    if len(output) == 0:
        print("No csv file found in directory")
        exit(1)

    if not args.output_dir.exists():
        os.mkdir(args.output_dir)

    with open(output_file, "w") as f:
        writer = csv.DictWriter(f=f, fieldnames=output[0].keys())
        writer.writeheader()
        output.sort(key=itemgetter("filename"))
        writer.writerows(output)

    print("Output written to: " + str(output_file.absolute()))


def calculate_geomean(result_csv: Path):
    result_table = pd.read_csv(result_csv)
    result_output_line = list()
    total_lines: int = result_table["line_count"].sum()
    total_rows: int = len(result_table.index)
    
    print("total_lines: " + str(total_lines))
    result_output_line.append(str(total_lines))
    result_table["weights"] = result_table["line_count"].div(total_lines)
    weighted_column_names = list()

    model_name = result_csv.stem.split('_')[-1]
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
    
    result_output_line.insert(0, model_name)

    return dict(zip(result_table.columns, result_output_line))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)