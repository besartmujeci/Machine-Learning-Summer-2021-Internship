import pandas as pd
import streamlit as st
from queue import PriorityQueue
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk
from nltk.corpus import stopwords
from textblob import Word

nltk.download("stopwords")
nltk.download("wordnet")

PATH = "CleanedRecommenderData.csv"


class Recommender:
    def __init__(self, text, num_recs, progress_bar, path=PATH):
        self.data = self.read_data(path)
        self.text = text
        self.num_recs = int(num_recs)
        self.progress_bar = progress_bar

    def recommend(self):
        recommender = PriorityQueue()

        placeholder = st.empty()
        placeholder.markdown("![Alt Text](https://media2.giphy.com/"
                             "media/tXL4FHPSnVJ0A/source.gif)")

        self.text = self.clean_new_text(self.text)
        self.text_append_to_data()
        vectors = self.tf_idf()

        placeholder.empty()
        to_recommend = vectors[-1]
        vectors = vectors[:-1]

        for i, text_vec in enumerate(vectors):
            recommender.put(
                (-1 * cosine_similarity(to_recommend, text_vec), i))
            self.progress_bar.progress(i / len(self.data))

        self.data = self.data[:-1]
        return [self.data["URL"][recommender.get()[1]] for _ in
                range(self.num_recs)]

    def text_append_to_data(self):
        df_post_to_get = pd.DataFrame([["", "", "", "", "", "", self.text]],
                                      columns=list(self.data.columns))
        self.data = self.data.append(df_post_to_get, ignore_index=True)
        self.data.reset_index(drop=True, inplace=True)

    def tf_idf(self):
        tf_idf = TfidfVectorizer()
        return tf_idf.fit_transform(self.data["Post"].values.astype('U'))

    @staticmethod
    def read_data(path):
        data = pd.read_csv(path)
        data = data.drop(data[data["Post"] == "nan"].index)
        data.reset_index(drop=True, inplace=True)
        return data

    @staticmethod
    def clean_new_text(text):
        text = text.replace("\n", " ")
        text = re.sub(r'http\S+', '', str(text))
        text = text.replace("\\n", " ")
        text = re.sub(r'\S+.jpg\S+', '', text)
        text = re.sub(r'\S+.png\S+', '', text)
        text = " ".join(x for x in text.split() if
                        x not in stopwords.words("english"))
        text = " ".join(x.lower() for x in text.split())
        text = text.replace("[^\\w\\s]", "")
        text = re.sub(r'\d+', '', text)
        text = text.replace("  ", " ")
        text = " ".join(
            [Word(word).lemmatize() for word in text.split()])

        return text
