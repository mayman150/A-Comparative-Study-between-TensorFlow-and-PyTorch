# IMPORTANT
## classify_issues_openai.py
This is a LEGACY implementation of issue classifier that uses openai API to classify issue

However due to our lack of money to pay for the API, we discarded this approach

This code is here to show what a potential classifier that utilize the GPT model could look it

It is UNTESTED and NOT GUARENTEED TO WORK

## classify_issues_rule.py

This is an old implementation that uses github issue template for classifying issues.

However, we discovered that the template is not strictly enforced, making its recall very bad.

That said, it does have a very high accuracy, making it a good helper function when generating training data.