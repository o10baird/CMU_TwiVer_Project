import numpy as np
import pandas as pd
import re
import click
from pathlib import Path
import preprocessor as p


def clean_tweet(tweet):
    #print(tweet)
    #print('Now Here')
    for index, temp in tweet.items():
        temp = str(tweet).lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()
    return tweet

@click.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path))
def clean_tweets(filename):
    file = filename
    path = Path.cwd()
    data_Dir = path
    file_Dir = path.joinpath(filename)

    #print(data_Dir)
    tweets = pd.read_csv(file_Dir, dtype='str' , names=['id', 'tweet', 'rating']) # This is a dataframe of actors on twitter and their information
    processed = filename.stem + "_final.tsv"
    out_path = data_Dir.joinpath(processed)
    #print(out_path)
    #print("Here")
    #print(tweets.head(10))
    tweets['hashtag'] = tweets['tweet_text'].apply(lambda x: re.findall(r"#(\w+)", x))
    for i,v in enumerate(tweets['tweet']):
        tweets.loc[v,'tweet'] = p.clean(i)
    tweets_processed = tweets.apply(lambda x: clean_tweet(x) if x.name =="tweet" else x)
    #print(tweets_processed.head(10))
    #print(tweets.head(10))
    tweets_processed.to_csv(out_path, header=False)
    print("Tweets cleaned")

if __name__ == "__main__":
    clean_tweets()
    ##  python data/process_tweets.py data/nagorno2.csv  Ran this to make this happen