# Evolution over bugs in both PyTorch and TensorfFlow closed issues

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

To be able to answer the question why researchers in Aritificial intelligence whould prefer to use PyTorch over TensorFlow, we need to get the insights how bugs are evolving over time in both frameworks. 

## Table of Contents

- [Installation](#installation)
- [Reproducing The Results](#Reproducing The Results)
- [Insights](#Insights)
- [Acknowledgments](#acknowledgments)

## Installation

To be able to run the scripts, just use the following command:
```bash
pip install -r requirements.txt
```

## Reproducing The Results

There are multiple steps to reproduce the results: 
### 1) You need to scrape the issues in both Pytorch and TensorFlow
       Just Run the following command 
       ```bash
        python3 ./Issues_parser/scraper.py
        ```
        And you are expecting to have the files for Scraped_Data. We splitted the Data to be able to add it in the github. 
### 2) Manually Classifying whether the issue is buggy or not, you will find the data in Issue_classifier/ML_Model_Training/Manually_classified_data. 
### 3) In the 
