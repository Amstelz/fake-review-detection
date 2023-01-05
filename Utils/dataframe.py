import sqlite3
import pandas as pd 
import numpy as np
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pythainlp.tokenize import word_tokenize
from collections import OrderedDict
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    conn = sqlite3.connect("../Data/yelpResData.db")
    conn.text_factory = lambda x: str(x, 'gb2312', 'ignore')
    cursor = conn.cursor()
    
    # Create Review DataFrame
    cursor.execute(
        "SELECT reviewID, reviewerID, restaurantID, date, rating, usefulCount as reviewUsefulCount, reviewContent, flagged FROM review where flagged in ('Y','N')")
    review_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Create Reviewer DataFrame
    cursor.execute("SELECT * FROM reviewer")
    reviewer_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Create Restaurant DataFrame
    cursor.execute("SELECT restaurantID, rating as restaurantRating FROM restaurant")
    restaurant_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

    # Merge all DataFrames
    review_reviewer_df = review_df.merge(reviewer_df, on='reviewerID', how='inner')
    df = review_reviewer_df.merge(restaurant_df, on='restaurantID', how='inner')
    return df

def data_cleaning(df):
    # Removing \n from date field
    for i in range(len(df['date'])):
        if df['date'][i][0] == '\n':
            df['date'][i] = df['date'][i][1:]

    # Making yelpJoinDate Format Uniform
    df['yelpJoinDate'] = df['yelpJoinDate'].apply(
        lambda x: datetime.strftime(datetime.strptime(x, '%B %Y'), '01/%m/%Y'))

    # Pre-processing Text Reviews
    # Remove Stop Words
    stop = stopwords.words('english')
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop))

    # Remove Punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: ' '.join(word for word in tokenizer.tokenize(x)))

    # Lowercase Words
    df['reviewContent'] = df['reviewContent'].apply(
        lambda x: x.lower())
    return df

def get_str_from_df(df:pd.DataFrame, round:int = 10, text_column:str = 'Text'):
    pd.set_option('max_colwidth', None)
    content_legnth = len(df)
    n = content_legnth // round
    str = ''
    for i in range(round):
        if i == round - 1:
            content = df.iloc[i*n:]
        else:
            content = df.iloc[i*n:(i+1)*n]
        # str += content['Text'].values[0]
        str += content[text_column].to_string(index=False)
    return str.replace('\n', '')


def get_most_frequent(text_string:str, remove_stop_word:bool):        
    frequency = {}

    words = re.findall(r'\b[A-Za-z][a-z]{2,9}\b', text_string)
    stop = stopwords.words('english')
    
    for word in words:
        if remove_stop_word:
            if word not in stop:
                count = frequency.get(word,0)
                frequency[word] = count + 1
        else:
            count = frequency.get(word,0)
            frequency[word] = count + 1

    most_frequent = dict(sorted(frequency.items(), key=lambda elem: elem[1], reverse=True))

    return most_frequent

def print_most_frequent(most_frequent:dict):
    top_count = 0
            
    for idx, (words, frequency) in enumerate(most_frequent.items()):
        if idx == 0:
            top_count = frequency
        print(f'{words}: {frequency}: {round(top_count/frequency, 2)}')

def string_to_list(string):
    return  [i[1:-1]for i in string[1:-1].split(', ')]

def onehot(df, col, col_name, type):
    for i in range(len(df[col][0])):
        df[f'{col_name[i]} ({type})'] = [data[i] for data in df[col]]
    df.drop(col, axis=1, inplace=True)
    return df

def under_sampling(df:pd.DataFrame, target:str, big_sample:any,  small_sample:any) -> pd.DataFrame:
    print("Under-Sampling Data")

    sample_size = len(df[(df[target] == big_sample)])

    small_sample_df = df[df[target] == small_sample]
    big_sample_df = df[df[target] == big_sample]

    small_sample_us_df = small_sample_df.sample(sample_size)
    under_sampled_df = pd.concat([small_sample_us_df, big_sample_df], axis=0)

    under_sampled_df.reset_index(drop=True, inplace=True)

    print("Under-Sampling Complete")
    return under_sampled_df

def over_sampling(df:pd.DataFrame, target:str, big_sample:any,  small_sample:any) -> pd.DataFrame:
    print("Over-Sampling Data")

    sample_size = len(df[(df[target] == small_sample)])

    small_sample_df = df[df[target] == small_sample]
    big_sample_df = df[df[target] == big_sample]

    big_sample_os_df = big_sample_df.sample(sample_size, replace=True)
    over_sampled_df = pd.concat([small_sample_df, big_sample_os_df], axis=0)

    over_sampled_df.reset_index(drop=True, inplace=True)

    print("Over-Sampling Complete")
    return over_sampled_df

def feature_engineering_thai(df):
    mnr_df1 = df[['reviewerID', 'date']].copy()
    mnr_df2 = mnr_df1.groupby(by=['date', 'reviewerID']).size().reset_index(name='reviewPerDay')
    mnr_df2['scaledReviewPerDay'] = mnr_df2['reviewPerDay'] / mnr_df2['reviewPerDay'].max()
    mnr_df2.drop(columns=['reviewPerDay'], inplace=True)
    df = df.merge(mnr_df2, on=['reviewerID', 'date'], how='inner')

    # Review Length
    df['reviewsLength'] = df['reviewContent'].apply(
        lambda x: len(word_tokenize(x, engine="newmm")))

    # Review Deviation
    df['reviewsDeviation'] = abs(df['rating'] - df['restaurantRating']) / 4

    # Maximum cosine similarity
    review_data = df

    res = OrderedDict()

    # Iterate over data and create groups of reviewers
    for row in review_data.iterrows():
        if row[1].reviewerID in res:
            res[row[1].reviewerID].append(row[1].reviewContent)
        else:
            res[row[1].reviewerID] = [row[1].reviewContent]

    individual_reviewer = [{'reviewerID': k, 'reviewContent': v} for k, v in res.items()]
    df2 = dict()
    df2['reviewerID'] = pd.Series([])
    df2['maximumContentSimilarity'] = pd.Series([])
    vector = TfidfVectorizer(min_df=0)
    count = -1
    for reviewer_data in individual_reviewer:
        count = count + 1
        # Handle Null/single review gracefully -24-Apr-2019
        try:
            tfidf = vector.fit_transform(reviewer_data['reviewContent'])
        except:
            pass
        cosine = 1 - pairwise_distances(tfidf, metric='cosine')

        np.fill_diagonal(cosine, -np.inf)
        max = cosine.max()

        # To handle reviewier with just 1 review
        if max == -np.inf:
            max = 0
        df2['reviewerID'][count] = reviewer_data['reviewerID']
        df2['maximumContentSimilarity'][count] = max

    df3 = pd.DataFrame(df2, columns=['reviewerID', 'maximumContentSimilarity'])

    # left outer join on original datamatrix and cosine dataframe -24-Apr-2019
    df = pd.merge(review_data, df3, on="reviewerID", how="left")

    df.drop(index=np.where(pd.isnull(df))[0], axis=0, inplace=True)
    return df

def feature_engineering(df):
    mnr_df1 = df[['reviewerID', 'date']].copy()
    mnr_df2 = mnr_df1.groupby(by=['date', 'reviewerID']).size().reset_index(name='reviewPerDay')
    mnr_df2['scaledReviewPerDay'] = mnr_df2['reviewPerDay'] / mnr_df2['reviewPerDay'].max()
    mnr_df2.drop(columns=['reviewPerDay'], inplace=True)
    df = df.merge(mnr_df2, on=['reviewerID', 'date'], how='inner')

    # Review Length
    df['reviewsLength'] = df['reviewContent'].apply(
        lambda x: len(x.split()))

    # Review Deviation
    df['reviewsDeviation'] = abs(df['rating'] - df['restaurantRating']) / 4

    # Maximum cosine similarity
    review_data = df

    res = OrderedDict()

    # Iterate over data and create groups of reviewers
    for row in review_data.iterrows():
        if row[1].reviewerID in res:
            res[row[1].reviewerID].append(row[1].reviewContent)
        else:
            res[row[1].reviewerID] = [row[1].reviewContent]

    individual_reviewer = [{'reviewerID': k, 'reviewContent': v} for k, v in res.items()]
    df2 = dict()
    df2['reviewerID'] = pd.Series([])
    df2['maximumContentSimilarity'] = pd.Series([])
    vector = TfidfVectorizer(min_df=0)
    count = -1
    for reviewer_data in individual_reviewer:
        count = count + 1
        # Handle Null/single review gracefully -24-Apr-2019
        try:
            tfidf = vector.fit_transform(reviewer_data['reviewContent'])
        except:
            pass
        cosine = 1 - pairwise_distances(tfidf, metric='cosine')

        np.fill_diagonal(cosine, -np.inf)
        max = cosine.max()

        # To handle reviewier with just 1 review
        if max == -np.inf:
            max = 0
        df2['reviewerID'][count] = reviewer_data['reviewerID']
        df2['maximumContentSimilarity'][count] = max

    df3 = pd.DataFrame(df2, columns=['reviewerID', 'maximumContentSimilarity'])

    # left outer join on original datamatrix and cosine dataframe -24-Apr-2019
    df = pd.merge(review_data, df3, on="reviewerID", how="left")

    df.drop(index=np.where(pd.isnull(df))[0], axis=0, inplace=True)
    return df