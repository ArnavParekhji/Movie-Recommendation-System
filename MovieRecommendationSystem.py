import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import pandas as pd
import numpy as np

df = pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')
df = df[['Title','Genre','Director','Actors','Plot']]

count = CountVectorizer()

df['Keywords'] = ""
df['bag_of_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    r = Rake()
    r.extract_keywords_from_text(plot)
    keywords_dict_scores = r.get_word_degrees()
    keywords = list(keywords_dict_scores.keys())
    keywordString = ""
    for keyword in keywords:
        keywordString = keywordString + " " + keyword
    row['Keywords'] = keywordString

    actorsString = ""
    directorsString = ""
    genreString = ""
    
    actors = row['Actors']
    actors = actors.lower()
    directors = row['Director']
    directors = directors.lower()
    genres = row['Genre']
    genres = genres.lower()
    
    anames = actors.split(',')
    dnames = directors.split(',')
    gnames = genres.split(',')
    
    for aname in  anames:
        aname = aname.strip()
        an = aname.split(' ')
        alcnames = ""
        for i in range(0, len(an)):
            alcnames = alcnames + an[i]
        actorsString = actorsString + " " + alcnames

    for dname in  dnames:
        dname = dname.strip()
        dn = dname.split(' ')
        dlcnames = ""
        for i in range(0, len(dn)):
            dlcnames = dlcnames + dn[i]
        directorsString = directorsString + " " + dlcnames

    for gname in  gnames:
        gname = gname.strip()
        gn = gname.split(' ')
        glcnames = ""
        for i in range(0, len(gn)):
            glcnames = glcnames + gn[i]
        genreString = genreString + " " + glcnames


    row['Actors'] = ""
    row['Actors'] = actorsString
    row['Director'] = ""
    row['Director'] = directorsString
    row['Genre'] = ""
    row['Genre'] = genreString

    row['bag_of_words'] = genreString + actorsString + directorsString + keywordString


df.drop(columns = ['Plot', 'Actors', 'Genre', 'Director', 'Keywords'], inplace = True)
df.set_index('Title', inplace = True)


count_matrix = count.fit_transform(df['bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)

def recommendations(title, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
    return recommended_movies

print(recommendations('Star Wars: Episode V - The Empire Strikes Back'))








