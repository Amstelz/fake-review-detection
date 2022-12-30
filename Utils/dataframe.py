import sqlite3
import pandas as pd 
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def load_data():
    conn = sqlite3.connect("../Data/yelpResData.db")
    conn.text_factory = lambda x: str(x, 'gb2312', 'ignore')
    cursor = conn.cursor()
    
    # Create Review DataFrame
    cursor.execute(
        "SELECT reviewID, reviewerID, restaurantID, date, rating, usefulCount as reviewUsefulCount, reviewContent, flagged FROM review")
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

def under_sampling(df):
    print("Under-Sampling Data")

    sample_size = len(df[(df['flagged'] == 'Y')])

    authentic_reviews_df = df[df['flagged'] == 'N']
    fake_reviews_df = df[df['flagged'] == 'Y']

    authentic_reviews_us_df = authentic_reviews_df.sample(sample_size)
    under_sampled_df = pd.concat([authentic_reviews_us_df, fake_reviews_df], axis=0)

    print("Under-Sampling Complete")
    return under_sampled_df

def over_sampling(df):
    print("Over-Sampling Data")

    sample_size = len(df[(df['flagged'] == 'N')])

    authentic_reviews_df = df[df['flagged'] == 'N']
    fake_reviews_df = df[df['flagged'] == 'Y']

    fake_reviews_os_df = fake_reviews_df.sample(sample_size, replace=True)
    over_sampled_df = pd.concat([authentic_reviews_df, fake_reviews_os_df], axis=0)

    print("Over-Sampling Complete")
    return over_sampled_df