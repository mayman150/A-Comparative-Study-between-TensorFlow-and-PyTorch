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

from text_preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


from sentence_transformers import SentenceTransformer


def bagOfWordsMapper(IssueTitle):
    '''
    Input: IssueTitle
    Output: Bag of Word Representation
    '''
    # preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]

    pre_processed_title = preprocess_text(IssueTitle)
    count_vector = CountVectorizer()
    bag_of_words = count_vector.fit_transform([pre_processed_title])
    return bag_of_words.toarray()[0]

def textualEmbeddingMapper(IssueTitle):
    '''
    Input: IssueTitle
    Output: Textual Embedding Representation
    Comment: Using HuggingFace pre-Trained Model
    '''
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    
    def text_to_embedding(sentence):
        # Tokenize input sentence
        tokenized_sentence = word_tokenize(sentence.lower())

        # Get the word vectors for each word in the sentence
        word_vectors = [model.wv[word] for word in tokenized_sentence if word in model.wv]

        # Average the word vectors to get the sentence vector
        sentence_vector = sum(word_vectors) / len(word_vectors) if word_vectors else None

        return sentence_vector

    return text_to_embedding(IssueTitle)



def bertEmbeddingMapper(IssueTitle, IssueBody):
    '''
    Input: IssueTitle, IssueBody
    Output: BERT Embedding Representation
    Comment: Using HuggingFace pre-Trained Model
    '''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    title_embedding = model.encode(IssueTitle)
    body_embedding = model.encode(IssueBody)
    return np.concatenate((title_embedding, body_embedding), axis=0)



def main():
    #test bag of words
    print(bagOfWordsMapper("This is Mohamed, man. My email is mohamedayman@15069@gmail.com. COntact me whenever u re avaialble."))

if __name__ == "__main__":
    main()