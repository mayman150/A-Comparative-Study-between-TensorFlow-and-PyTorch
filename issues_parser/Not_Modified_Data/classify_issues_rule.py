import argparse
import pandas as pd


def tf_rule(all_issues: pd.DataFrame):
    # filter based on having tags of "type:bug"
    contain_values = all_issues["Tags"].str.contains("type:bug")
    all_issues["Is Bug"] = contain_values


def torch_rule(all_issues: pd.DataFrame):
    # filter based on having "Describe the bug" in body
    body_rows = all_issues["Issue Body"].str.split('\n')

    def check_is_issue(row: list):
        if not isinstance(row, list) or len(row) == 0:
            return False
        
        first_line = str(row[0])
        if "describe the bug" in first_line.lower():
            return True
        return False

    is_bugs = list()
    for row in body_rows.items():
        is_bugs.append(check_is_issue(row[1]))
        
    all_issues["Is Bug"] = is_bugs


def main():
    parser = argparse.ArgumentParser(allow_abbrev=True)

    parser.add_argument("csv_files", nargs="+", help="Locations of the csv files of issues.")
    parser.add_argument("-o", "--output_file", metavar="", default="output.csv", help="Output csv file path")
    parser.add_argument("-l", "--library_name", choices=["torch", "tf"], required=True, help="Choose the type of library")
    args = parser.parse_args()

    file_list = []
    for csv_file in args.csv_files:
        file = pd.read_csv(csv_file, index_col="Issue Number")
        file_list.append(file)
    
    all_issues = pd.concat(file_list)

    if args.library_name == "tf":
        tf_rule(all_issues)
    else:
        torch_rule(all_issues)
    
    all_issues.to_csv(args.output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)