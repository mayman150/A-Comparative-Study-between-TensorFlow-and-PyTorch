from github import Github
from datetime import datetime, timedelta
from utils import get_issues, write_issues_to_csv



def __main__():
    # authenticate with the token
    ACCESS_TOKEN = "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"
    g = Github(ACCESS_TOKEN)
    # set the date two years ago

    #Pytorch
    issues = get_issues(g, 'pytorch/pytorch', state='closed')
    # write issues to a CSV file
    write_issues_to_csv(g, issues, 'PyTorch_issues.csv', ['Issue Number', 'Issue Title', 'Issue Body', 'Time created', 'Time closed', 'Number of Assignees', 'Number of Comments', 'Tags'])

    #Tensorflow
    issues = get_issues(g, 'tensorflow/tensorflow', state='closed')
    # write issues to a CSV file
    write_issues_to_csv(g,issues, 'Tensorflow_issues.csv', ['Issue Number', 'Issue Title', 'Issue Body', 'Time created','Time closed' , 'Number of Assignees', 'Number of Comments', 'Tags'])


if __name__ == "__main__":
    __main__()