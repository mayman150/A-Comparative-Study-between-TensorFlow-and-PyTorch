# Evolution over bugs in both PyTorch and TensorFlow closed issues

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

To be able to answer the question why researchers in Aritificial intelligence whould prefer to use PyTorch over TensorFlow, we need to get the insights how bugs are evolving over time in both frameworks. 

## Table of Contents

- [Reproducing The Results](#Reproducing_The_Results)
- [Insights](#Insights)
- [Acknowledgments](#acknowledgments)



## Reproducing The Results

There are multiple steps to reproduce the results: 
#### 1) You need to scrape the issues in both Pytorch and TensorFlow
Just Run the following command 
```bash
python3 scraper.py --data_dir Folder_TO_SAVE_CSV_FILES --file_suffix YOUR_SUFFIX_FILE_FOR_BOTH --state TYPE_OF_ISSUE_YOU_WANT_TO_SCRAPE
```
        
And you are expecting to have the files for Scraped_Data. We splitted the Data to be able to add it in the github. 
        
#### 2) Manually Classifying whether the issue is buggy or not, you will find the data in Issue_classifier/ML_Model_Training/Mapped_Data/Manually_classified_data. 
       Check more in the **prepare_data** notebook for more details and insights about the data. We found out that the format for TensorFlow issue is somehow different from PyTorch Issue, which directed us to do a classification for each one alone. But Generally, we balanced data such that we have 171 buggy TensorFlow issues, and 171 not buggy TensorFlow issues. For PyTorch, we have 194 buggy pytorch issues and 171 buggy tensorflow issues. We manually calssified parts of the issues to have a set of 388
PyTorch issues and 342 TensorFlow issues. More detail discussed in the report.

#### 3) Tried different models to choose the best to predict whether issues are buggy or not. 
You can reproduce the results in this sheet (https://docs.google.com/spreadsheets/d/1cvkvzK7qmmBwF825iB4jhYmMa-C5ubVbUPCEmv1OLck/edit?usp=sharing) in models_sklearn.ipynb
#### 4) For the analysis, check evolution_over_bugs_analysis.ipynb 
