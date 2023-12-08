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
1. Add your Github Secret in the Scraper Code. Use this link to figure out how you obtain this secret from your Github: https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens?apiVersion=2022-11-28
2. create .env file and add your ACCESS_TOKEN = "Your Github Access Token".

To execute the script, run the following command in your terminal:

```bash
Pytorch: 
python3 scraper.py -p --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE

TensorFlow:
python3 scraper.py -f --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE
```

For splitting the files, you can just run the following command: 

```bash
python3 csv_splitter --source_filepath YOUR_SOURCE_FILE_PATH --dest_folder YOUR_DESTINATION_FOLDER --filename_prefix your_prefix_file --row_limit MAX_NUMBER_OF_ROWS_FOR_EACH_FILE
