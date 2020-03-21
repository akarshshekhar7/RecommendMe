#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:35:03 2020

@author: akarsh
"""

import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


pd.set_option('display.max_columns', 2000)
df = pd.read_csv('movie_metadata.csv')


df = df[['Title','Genre','Director','actor_1_name','actor_2_name','actor_3_name','Plot']]

#replace \ with ,
df['Plot'] =df['Plot'].replace('\|',',',regex=True)
df['Genre'] =df['Genre'].replace('\|',',',regex=True)

#change type of columns
df['Director'] = df['Director'].astype(str)
df['actor_1_name'] = df['actor_1_name'].astype(str)
df['actor_2_name'] = df['actor_2_name'].astype(str)
df['actor_3_name'] = df['actor_3_name'].astype(str)

df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))

# putting the genres in a list of words

df['Director'] = df['Director'].map(lambda x: x.split(' '))

df['actor_1_name'] = df['actor_1_name'].map(lambda x: x.split(','))

df['actor_2_name'] = df['actor_2_name'].map(lambda x: x.split(','))

df['actor_3_name'] = df['actor_3_name'].map(lambda x: x.split(','))

# merging together first and last name for each actor and director, so it's considered as one word 
# and there is no mix up between people sharing a first name
for index, row in df.iterrows():
    row['actor_1_name'] = [x.lower().replace(' ','') for x in row['actor_1_name']]
    row['actor_2_name'] = [x.lower().replace(' ','') for x in row['actor_2_name']]
    row['actor_3_name'] = [x.lower().replace(' ','') for x in row['actor_3_name']]
    row['Director'] = ''.join(row['Director']).lower()
    
# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    
    # instantiating Rake, by default is uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column
    row['Key_words'] = list(key_words_dict_scores.keys())
    
df.drop(columns = ['Plot'], inplace = True)

df.set_index('Title', inplace = True)

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)    

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies


