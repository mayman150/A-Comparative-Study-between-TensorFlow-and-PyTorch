from github import Github
from datetime import datetime, timedelta
from utils import get_issues, write_issues_to_csv
from decouple import config
import argparse


def __main__():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pytorch", action="store_true", help="Scrape for pytorch.")
    parser.add_argument("-t", "--tensorflow", action="store_true", help="Scrape for tensorflow.")
    parser.add_argument("-f", "--file_suffix", help="suffix of the file. Will append 'PyTorch_ / Tensorflow_' in front for corresponding file. Will always be in the same directory", default="_issues.csv")
    parser.add_argument("--state", help="state of the issue. Can be 'open', 'closed', or 'all'", default="all")
    args = parser.parse_args()

    if not (args.pytorch or args.tensorflow):
        parser.error("At least one option, --pytorch or --tensorflow, must be selected")

    # authenticate with the token
    g = Github(config("ACCESS_TOKEN", cast=str))

    #Pytorch
    if args.pytorch:
        issues = get_issues(g, 'pytorch/pytorch', state=args.state)
        # write issues to a CSV file
        write_issues_to_csv(g, issues, 'PyTorch_' + args.file_suffix, ['Issue Number', 'Issue Title',  'Time created', 'Time closed', 'Number of Assignees', 'Number of Comments', 'Tags'])

    #Tensorflow
    if args.tensorflow:
        issues = get_issues(g, 'tensorflow/tensorflow', state=args.state)
        # write issues to a CSV file
        write_issues_to_csv(g, issues, 'Tensorflow_' + args.file_suffix, ['Issue Number', 'Issue Title',  'Time created','Time closed' ,  'Number of Assignees', 'Number of Comments', 'Tags'])

if __name__ == "__main__":
    __main__()