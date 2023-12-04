## Parsing Issues in TensorFlow and PyTorch

This code provides comprehensive information on each issue, including the following features:

1. Issue Number
2. Issue Title
3. Time Created
4. Time Closed
5. Number of Assignees
6. Number of Comments
7. Tags

### Execution Instructions

To execute the script, run the following command in your terminal:

```bash
python3 scraper.py --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE

### File Splitting 
For splitting the files, you can just run the following command: 

```bash
python3 csv_splitter --source_filepath YOUR_SOURCE_FILE_PATH --dest_folder YOUR_DESTINATION_FOLDER --filename_prefix your_prefix_file --row_limit MAX_NUMBER_OF_ROWS_FOR_EACH_FILE
