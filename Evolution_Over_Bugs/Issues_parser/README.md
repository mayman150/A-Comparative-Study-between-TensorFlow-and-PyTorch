# Parsing Issues in TensorFlow and PyTorch

This code provides comprehensive information on each issue, including the following features:

1. Issue Number
2. Issue Title
3. Time Created
4. Time Closed
5. Number of Assignees
6. Number of Comments
7. Tags

## Execution Instructions

### Scraping Issues
1. Add your Github Secret in the Scraper Code. Use this link to figure out how you obtain this secret from your Github: https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens?apiVersion=2022-11-28
2. create .env file and add your access token. It should look something like it:
```python
ACCESS_TOKEN="Your Github Access Token"
```

To execute the script for both PyTorch and TensorFlow, run the following command in your terminal:

```bash
python3 scraper.py -p -t --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE
```

Where the option `-p` specifies to scrape for PyTorch while `-t` specified for TensorFlow. It is possible to specify only one of them at a time.

The output will be a csv with a name format of `PyTorch_/Tensorflow_{Your chosen suffix}.csv`

### Splitting CSV
For splitting the files, you can just run the following command: 

```bash
python3 csv_splitter.py --source_filepath YOUR_SOURCE_FILE_PATH --dest_folder YOUR_DESTINATION_FOLDER --filename_prefix your_prefix_file --row_limit MAX_NUMBER_OF_ROWS_FOR_EACH_FILE
```

This CSV splitter is so that we can bypass the 50MB large file limit on GitHub. For example, we have a large file `tf_issue.csv` that is over 50 MB and we want to split it into smaller files, and store into the folder `output/`:
```bash
python3 csv_splitter.py --source_filepath tf_issue.csv --dest_folder output/
```

If the splitted file is still too large, we can specify a smaller row limit. Row limit is the maximum number of rows per each splitted csv file:
```bash
python3 csv_splitter.py --source_filepath tf_issue.csv --dest_folder output/ --row_limit 1000
```

Now the files in `output/` will look like this:
```bash
output/
├── tf_issue.csv_0.csv
├── tf_issue.csv_1.csv
├── tf_issue.csv_2.csv
└── tf_issue.csv_3.csv
```

If a different prefix is preferred, it is possible to specify with `--filename_prefix`:

```bash
python3 csv_splitter.py --source_filepath tf_issue.csv --dest_folder output/ --filename_prefix another_prefix.csv
```

The files in `output/` will look like this:
```bash
output/
├── another_prefix.csv_0.csv
├── another_prefix.csv_1.csv
├── another_prefix.csv_2.csv
└── another_prefix.csv_3.csv
```
