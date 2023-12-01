from github import Github
from datetime import datetime, timedelta
from utils import get_issues, write_issues_to_csv



def __main__():
    # authenticate with the token
    ACCESS_TOKEN = "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"
    g = Github(ACCESS_TOKEN)
    # set the date two years ago

    #Pytorch
    issues = get_issues(g, 'pytorch/pytorch', state='all')
    # write issues to a CSV file
    write_issues_to_csv(g, issues, 'PyTorch_issues_final2.csv', ['Issue Number', 'Issue Title',  'Time created', 'Time closed', 'Number of Assignees', 'Number of Comments', 'Tags'])

    #Tensorflow
    issues = get_issues(g, 'tensorflow/tensorflow', state='all')
    # write issues to a CSV file
    write_issues_to_csv(g,issues, 'Tensorflow_issues_final2.csv', ['Issue Number', 'Issue Title',  'Time created','Time closed' ,  'Number of Assignees', 'Number of Comments', 'Tags'])


if __name__ == "__main__":
    __main__()