from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en")
text_sentiment_columns = ['Text', 'Sentiment']
df1 = pd.read_csv("data/amazon_cells_labelled.txt", delimiter="\t", header=None)
df2 = pd.read_csv("data/imdb_labelled.txt", delimiter="\t", header=None)
df3 = pd.read_csv("data/yelp_labelled.txt", delimiter="\t", header=None)
frames = [df1, df2, df3]

def concat_datasets():
    df = pd.concat(frames)
    df.columns = text_sentiment_columns
    return df

def clean_text():
    df = concat_datasets()
    text_list = df["Text"].values
    lower_text_list = []
    for text in text_list:
        text_lower = text.lower()
        lower_text_list.append(text_lower)
    clean_text_list = []
    for text in lower_text_list:
        text = nlp(str(text))
        token = [token.orth_ for token in text if not token.is_punct]
        clean_text_list.append(token)
    return clean_text_list

def bag_of_words():
    text = clean_text()
    for phrase in text:
        print(phrase)
        vectorizer = CountVectorizer()
        bag = vectorizer.fit_transform(phrase).todense()
        print(bag)
    return bag

if __name__ == '__main__':
    bag_of_words()