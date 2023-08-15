import pandas as pd
import numpy as np
import re
import nltk
# run if you get an error saying that you don't have the stopwords, wordnet, or averaged_perceptron_tagger modules
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import ast

# viewing an entire dataframe from a csv file



# remove stop words
STOP_WORDS = stopwords.words('english')


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOP_WORDS])


# Remove Punctuation, Links, Usernames, Hashtags
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove http links
    text = re.sub(r"www\S+", "", text)  # remove www links
    text = re.sub(r"\@\w+", "", text)  # remove usernames
    text = re.sub(r"\#\w+", "", text)  # remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text


# Lemmatize text
lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# tokenize text
def tokenization(text):
    return text.split()

# removing emojis

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Apply Preprocessing Steps
def preprocess_tweets(df, column):
    df[column] = df[column].apply(remove_emoji)  # remove emojis
    df[column] = df[column].apply(lambda x: x.lower())  # lowercase
    df[column] = df[column].apply(remove_stopwords)  # remove stopwords
    df[column] = df[column].apply(clean_text)  # remove urls, at tags, hashtags, punctuation
    df[column] = df[column].apply(lemmatize_words)  # lemmatize
    df[column] = df[column].apply(tokenization)  # tokenize
    return df

def main():
    df = pd.read_csv("C:/Users/mohit/PycharmProjects/pythonProject/training.1600000.processed.noemoticon.csv",
                     encoding="ISO-8859-1")

    # print(df.head())
    # first column is the sentiment of the tweet, 0 is negative and 4 is positive, second column is the id of the tweet, third column is the date and time of the tweet, fourth column is the query, fifth column is the username of the person who tweeted, sixth column is the tweet itself

    # viewing a single column from a dataframe from a csv file and converting it into a list of strings for further processing and analysis of the data in the column and then printing the list of strings to the console to see the data in the column and to see if the data is in the correct format for further processing and analysis of the data in the column
    # the columns are not labled, so we have to label them ourselves and then access them by their labels
    df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]
    # print(df["tweet"])
    # converting the column into a list of strings
    tweets = df["tweet"].tolist()
    # Run Preprocessing
    df = preprocess_tweets(df, "tweet")
    '''
    # tagging parts of speech
    
    def pos_tagging(text):
        return nltk.pos_tag(text)
    
    
    df["tweet"] = df["tweet"].apply(pos_tagging)
    
    print(df["tweet"])

    # saving df to a csv file
    df.to_csv("C:/Users/mohit/PycharmProjects/pythonProject/processed_tweets.csv", index=False)

    '''

    # Assuming df is your DataFrame and 'tweets' is the column with the lists
    df = df[df['tweet'].apply(lambda x: bool(x))]

    #converting df[tweet] to a list of lists

    #df["tweet"] = df["tweet"].apply(lambda x: x.strip('"'))
    #saving df to a csv file
    print(df["tweet"])
    df.to_csv("C:/Users/mohit/PycharmProjects/pythonProject/processed_tweets.csv", index=False)
if __name__ == "__main__":
    main()