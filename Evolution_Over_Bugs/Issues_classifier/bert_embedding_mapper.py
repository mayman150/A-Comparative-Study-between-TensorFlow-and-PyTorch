'''
Three Options: 
(1): Bag Of Words
(2): Textual Embedding
(3): BERT Embedding
'''
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tqdm import tqdm
import argparse

# from text_preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


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

    parser.add_argument("--csv_file", help="location of the ground truth CSV file")
    parser.add_argument("-o", "--output_csv", help="output file location", default="GT_bert_data.csv")

    args = parser.parse_args()

    #Read CSV File 
    df = pd.read_csv(args.csv_file)
    df_result = df.copy()
    #Have a numpy array of all the issue titles and issue bodies
    IssueTitles = df['Issue Title'].to_numpy()
    IssueBodies = df['Issue Body'].to_numpy()
    embeddings  = []
    for i in tqdm(range(len(IssueTitles))):
        all_issues = str(IssueTitles[i]) + ' ' + str(IssueBodies[i])
       
        res_embedding = bertEmbeddingMapper(all_issues)
        embeddings.append(res_embedding)
    
    #add the list in the dataframe
    df_result['BERT Embedding'] = embeddings

    df_result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()