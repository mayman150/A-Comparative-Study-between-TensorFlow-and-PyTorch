# A Comparative Study between TensorFlow and PyTorch

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview
This repository hosts a comprehensive comparative study between two popular deep learning frameworks, TensorFlow and PyTorch. The investigation spans across two key dimensions: (1) Usability Functionality and (2) Understanding Pytorch and TensorFlow bugs overtime.

## Code-Dependencies
```bash
1. Python 3.x with numpy, pandas, matplotlib
2. Pytorch 2.x
```
We reproduced the results with Python 3.10.13 on Ubuntu 22.04.

## Installation 

We recommend using Anaconda and create an isolated Python environment.

```bash
conda create -n proj_663 python=3.10.13
conda activate proj_663
pip install -r requirements.txt
```
## Reproducing the Results 

To facilitate the replication of our findings and encourage contributions for further research, we provide detailed steps on how to reproduce the results for both dimensions explored in our study.

**Note that** our data can be found in our github repos. For each dimension of the project folder, you will find the needed data. 


### Usability Functionality

In exploring the usability functionality for client code and API documentation, we aim to gain insights through quantitative comparisons between PyTorch and TensorFlow. The following dimensions are considered:

1. **API Usability on Documentation:** Evaluate the ease of use and clarity of the documentation provided by both frameworks.

2. **API Density on Client Code:** Analyze the density of API calls within client code, specifically focusing on models implemented in both PyTorch and TensorFlow.

3. **Code Readability on Client Code:** Assess the readability of client code implemented in both frameworks.

To replicate our results, detailed instructions are available in the **Usability_Functionality** folder.

### Evolution Over Bugs

Understanding the evolution of bugs provides valuable insights into the temporal trends of issues within the source code of PyTorch and TensorFlow. Additionally, analyzing closed issues offers insights into the size and activity of the community, potentially influencing researchers' preference for PyTorch over TensorFlow.

To delve into these insights and reproduce our findings, you will find in the README.me of **Evolution_Over_Bugs** folder.


**Our results** can be found in the following sheet: https://docs.google.com/spreadsheets/d/1cvkvzK7qmmBwF825iB4jhYmMa-C5ubVbUPCEmv1OLck/edit?usp=sharing



**Feel free to explore subdirectories** within each section for more in-depth discussions and contribute to our ongoing research efforts.
