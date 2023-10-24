import subprocess
import nltk
import spacy
import string
import numpy as np
import pandas as pd 
import re 
from tqdm import tqdm 
import matplotlib.pyplot as plt
tqdm.pandas() 

import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')
# spacy.download('en_core_web_sm')
stop_words = set(stopwords.words('english'))

from sklearn.metrics.pairwise import cosine_similarity

# @st.cache_resource
# def download_en_core_web_sm():
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])


# download_en_core_web_sm()


def plot_graph_to_check_hashtags_impact(data_df, hashtag_col, imp_col):
    df = data_df.explode(hashtag_col).copy(deep=True)
    print(f'Total Different Types of Hashtags Used: {df[hashtag_col].nunique()}')
    
    # Filter the DataFrame to include only the relevant hashtags (at least 10 percentile of the total number of hashtags)
    percentile = df[hashtag_col].value_counts().quantile(0.90)
    filtered_hashtags = df[hashtag_col].value_counts()[df[hashtag_col].value_counts() >= percentile].index.tolist() 
    df_filtered = df[df[hashtag_col].isin(filtered_hashtags)]

    # Calculate the mean engagement rate for each hashtag
    mean_imp_by_hashtag = df_filtered.groupby(hashtag_col)[imp_col].mean().reset_index()

    # Sort the DataFrame by mean engagement rate in descending order
    mean_imp_by_hashtag = mean_imp_by_hashtag.sort_values(by=imp_col, ascending=False)
    # print(mean_imp_by_hashtag[[hashtag_col, imp_col]])
    top_five_hashtags = mean_imp_by_hashtag[[hashtag_col]][:5]
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(mean_imp_by_hashtag[hashtag_col], mean_imp_by_hashtag[imp_col], color='skyblue')
    plt.xlabel(hashtag_col)
    plt.ylabel(f'Mean {imp_col}')
    plt.title('Impact of Hashtags')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    return top_five_hashtags


def hash_day(data_df):
    mean_imp_by_day_df = data_df.groupby('Posting Day')['Post Impressions'].mean().reset_index()
    mean_imp_by_day_df = mean_imp_by_day_df.sort_values(by='Post Impressions', ascending=False)
    plt.figure(figsize=(5, 3))
    plt.bar(mean_imp_by_day_df['Posting Day'], mean_imp_by_day_df['Post Impressions'], color='skyblue')
    plt.xlabel('Posting Day')
    plt.ylabel(f'Mean Post Impressions')
    plt.title('Impact of Hashtags')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.show()


def preprocess_caption(caption):
    tokens = word_tokenize(caption.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join(text.split())
    return text

def get_cosine_similarity(user_query, caption):
    # download_en_core_web_sm()
    try:
        nlp = spacy.load("en_core_web_sm")
        embedding1 = nlp(clean_text(user_query)).vector
        embedding2 = nlp(clean_text(caption)).vector
        return cosine_similarity([embedding1], [embedding2])[0][0]
    except Exception as e:
        st.write(e)
        
def predict(user_query):
    # preprocess data
    try:
        xls = pd.ExcelFile('LinkedIn Analytics.xlsx')
        df1 = pd.read_excel(xls, 'April-August 2023')
        df2 = pd.read_excel(xls, 'August-Oct 2022')
        df = pd.concat([df1, df2])
        data = df.copy(deep=True)
        if len(user_query.strip()) <= 0:
            return ""
            # preprocess data    
        data = data.copy(deep=True)
        # data['preprocessed_captions'] = data['Post Caption'].apply(preprocess_caption)
        data['Posting Date '] = pd.to_datetime(data['Posting Date '], errors='coerce')
        data['Posting Day'] = data['Posting Date '].dt.strftime('%A')
        data['Hashtags used'] = data['Hashtags used'].apply(lambda x: re.split(r'[ \xa0\n]+', x))
        data['Hashtags used'] = data['Hashtags used'].apply(lambda x: [a.lower() for a in x if a.startswith('#')])
        print("Hashtags created")
        # data = data[['Post Caption', 'preprocessed_captions', 'Posting Date ', 'Hashtags used', 'Post Impressions', 'Posting Day']]
        # Train Word2Vec model
        # model = gensim.models.Word2Vec(data['preprocessed_captions'], vector_size=100, window=5, min_count=1, sg=0)
        # Save the trained Word2Vec model to a file
        # model.save("word_embeddings.gensim")
        # Preprocess the user query
        # user_query = preprocess_caption(user_query)
        # Get the vector representation of the user query
        # user_query_vector = model.wv[user_query]
        # Calculate cosine similarity between the user query and post captions
        data['cosine_similarity'] = data['Post Caption'].apply(lambda x: get_cosine_similarity(user_query, x))
        print("assigned similarity score")
        # Sort by cosine similarity in descending order to get the most relevant captions
        data = data.sort_values(by='cosine_similarity', ascending=False)
        # print(data.head(10))
        res = {}
        similar_df_thres = data[data['cosine_similarity'] > 0.5]

        if similar_df_thres.shape[0] == 0:
            res['status'] = "Note: While historical data hasn't provided best matches for the caption, these plots may offer a glimpse into impression predictions for the caption."
            st.write(res['status'])
            similar_df_thres = data.head(10)
        else:
            res['status'] = f"Historical data came up {similar_df_thres.shape[0]} related posts, Check out these plots for possible impression predictions."
            st.write(res['status'])
        top_five_hashtags = plot_graph_to_check_hashtags_impact(similar_df_thres, 'Hashtags used', 'Post Impressions')
        hash_day(similar_df_thres)
        res['Top Hashtags'] = top_five_hashtags['Hashtags used'].to_list()
        return res
    except:
        return ""
