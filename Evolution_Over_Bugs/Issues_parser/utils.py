import csv 
from github import Github
from github import RateLimitExceededException
from github.GithubException import GithubException
import time
import calendar
import logging
from pprint import pprint
from pathlib import Path


def writing_in_file(issues, file_name):
    """ًWriting the Label issues in a file_name
    Args:
        issues (list): List of issues.
        file_name    : (str): Name of the CSV file.
    Returns:
        None
    """
    states = set()

    for issue in issues:
        for label in issue.labels:
            states.add(label.name)
    
    with open(file_name, 'w') as f:
        for item in states:
            f.write("%s " % item)
    
    return states


def get_issues(g, repo_name, state):
    """Get issues of a repository.
    Args:
        repo_name (str): Name of the repository.
        state (str): State of the issues. Can be 'open', 'closed' or 'all'.
    Returns:
        list: List of issues.
    """
    while True: 
        try:
            repo = g.get_repo(repo_name)
            issues = repo.get_issues(state=state)
            break

        #Handle if we exceed the rate limit
        except RateLimitExceededException as e:
            search_rate_limit = g.get_rate_limit().search
            logging.info('search remaining: {}'.format(search_rate_limit.remaining))
            reset_timestamp = calendar.timegm(search_rate_limit.reset.timetuple())
            # add 10 seconds to be sure the rate limit has been reset
            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 10
            print("Rate Limit Exceeded", sleep_time)
            time.sleep(sleep_time)
            continue 
        
    return issues


def write_issues_to_csv(g, issues, filename, fieldnames):
    """Write issues to a CSV file.
    Args:
        issues (list): List of issues.
        filename (str): Name of the CSV file.
        fieldnames (list): List of field names.
    """
    output_file_path = Path(filename)

    # create output folder if not exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        writer.writeheader()

        # iterate over issues
        iter_obj = iter(issues)
        while True: # To wait if we exceed the rate limit
            try:
                issue = next(iter_obj)
                isPullRequest = False
                while True: #To wait if any problem happened in pull request and exceeded time-out
                    try:
                        if issue.pull_request:
                            isPullRequest = True
                            break
                        else:
                            break
                    except GithubException as e:
                        print(e)
                        continue
                if isPullRequest:
                    continue

                writer.writerow({'Issue Number': issue.number,
                                'Issue Title': issue.title,
                                'Time created': issue.created_at,
                                'Time closed': issue.closed_at,
                                # 'Issue response time': issue.closed_at - issue.created_at,
                                'Number of Assignees': len(issue.assignees),
                                'Number of Comments': issue.comments,
                                'Tags': issue.labels,                                
                                })
            
            except StopIteration:
                break
            
            #Handle if we exceed the rate limit
            except RateLimitExceededException: 
                search_rate_limit = g.get_rate_limit().search
                logging.info('search remaining: {}'.format(search_rate_limit.remaining))
                reset_timestamp = calendar.timegm(search_rate_limit.reset.timetuple())
                # add 10 seconds to be sure the rate limit has been reset
                sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 10
                print("Rate Limit Exceeded", sleep_time)
                time.sleep(sleep_time)
                continue

            except Exception as e:
                print("Error with Issue: ")
                pprint({'Issue Number': issue.number,
                        'Issue Title': issue.title,
                        'Time created': issue.created_at,
                        'Time closed': issue.closed_at,
                        # 'Issue response time': issue.closed_at - issue.created_at,
                        'Number of Assignees': len(issue.assignees),
                        'Number of Comments': issue.comments,
                        'Tags': issue.labels,                                
                        })
                print("Error: ", str(e.args))
                print("Skipping Issue")
                continue
    