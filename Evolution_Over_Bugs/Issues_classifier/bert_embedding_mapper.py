'''
Three Options: 
(1): Bag Of Words
(2): Textual Embedding
(3): BERT Embedding
'''
import pandas as pd

from tqdm import tqdm
import argparse

from sentence_transformers import SentenceTransformer

def bertEmbeddingMapper(IssueTitle, IssueBody):
    '''
    Input: IssueTitle, IssueBody
    Output: BERT Embedding Representation
    Comment: Using HuggingFace pre-Trained Model
    '''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    title_embedding = model.encode(IssueTitle)
    body_embedding = model.encode(IssueBody)
    return title_embedding, body_embedding

def bertEmbeddingMapper(Issues):
    '''
    Input: IssueTitle, IssueBody
    Output: BERT Embedding Representation
    Comment: Using HuggingFace pre-Trained Model
    '''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    res_embedding = model.encode(Issues)
    return res_embedding



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--csv_file", help="location of the ground truth CSV file", required=True)
    parser.add_argument("-o", "--output_csv", help="output file location", default="GT_bert_data.csv")
    parser.add_argument("--option", help="BERT Embedding", default="Title_Only")

    args = parser.parse_args()

    #Read CSV File 
    df = pd.read_csv(args.csv_file)
    df_result = df.copy()
    #Have a numpy array of all the issue titles and issue bodies
    IssueTitles = df['Issue Title'].to_numpy()
    Tags = df['Tags'].to_numpy()
    IssueBodies = df['Issue Body'].to_numpy()
    
    embeddings  = []
    if args.option == "Title_Only":
        for i in tqdm(range(len(IssueTitles))):
            res_embedding = bertEmbeddingMapper(str(IssueTitles[i]) + ' ' + str(Tags[i]))
            embeddings.append(res_embedding)
    elif args.option == "all":
        for i in tqdm(range(len(IssueTitles))):
            all_issues = str(IssueTitles[i]) + ' ' +str(Tags[i]) + ' ' +str(IssueBodies[i])
            res_embedding = bertEmbeddingMapper(all_issues)
            embeddings.append(res_embedding)
    
    #add the list in the dataframe
    df_result['BERT Embedding'] = embeddings

    df_result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()