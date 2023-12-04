# Evolution over bugs in both PyTorch and TensorFlow closed issues

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

To be able to answer the question why researchers in Aritificial intelligence whould prefer to use PyTorch over TensorFlow, we need to get the insights how bugs are evolving over time in both frameworks. 

## Reproducing The Results

There are multiple steps to reproduce the results: 
#### 1) You need to scrape the issues in both Pytorch and TensorFlow
Just Run the following command 
```bash
python3 scraper.py --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE
```
And you are expecting to have the files for Scraped_Data. We splitted the Data to be able to add it in the github. 

#### 2) Manually Classifying whether the issue is buggy or not, you will find the data in Issue_classifier/ML_Model_Training/Mapped_Data/Manually_classified_data. 
Check more in the **prepare_data** notebook for more details and insights about the data. We found out that the format for TensorFlow issue is somehow different from PyTorch Issue, which directed us to do a classification for each one alone.
Generally, we balanced data such that we have 171 buggy TensorFlow issues, and 171 not buggy TensorFlow issues. 
**For PyTorch**
   - we have 194 buggy pytorch issues + 194 unbuggy Pyotrch Issues
   - 171 buggy tensorflow issues + 171 unbuggy PyTorch Issues.
More detail discussed in the report.


#### 3) Obtaining Bert Embeddings for Issue Title and Issue Body for Ground Truth (Manually Classified Issues from Section 2)

We experimented with two primary combinations: Issue Title + Issue Body + Tags and Issue Title + Tags. In the `ML_Model_Training/Mapped_Data` directory, you'll find three main sections:

1. **Data_With_Embeddings:**
   - `GT_bert_concat_data`: BERT embeddings for Issue Title + Issue Body + Tags combination.
   - `GT_Title_bert_concat_data`: BERT embeddings for Issue Title + Tags combination.

To reproduce the results, follow these steps:

```bash
        python3 textual_mapper.py --csv_file INPUT_FILE --output_csv OUTPUT_FILE
```
Replace INPUT_FILE with the path to your input CSV file and OUTPUT_FILE with the desired path for the output CSV file. This command generates BERT embeddings for the specified combination and saves the results in the output CSV file.

For other mapping types, they are already automated for you in models_sklearn.ipynb

#### 3) Tried different models to choose the best to predict whether issues are buggy or not. 
You can reproduce the results in this sheet (https://docs.google.com/spreadsheets/d/1cvkvzK7qmmBwF825iB4jhYmMa-C5ubVbUPCEmv1OLck/edit?usp=sharing) in models_sklearn.ipynb
#### 4) For the analysis, check evolution_over_bugs_analysis.ipynb 
