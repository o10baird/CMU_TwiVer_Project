import pandas as pd
import click
from pathlib import Path

@click.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path))
def process_tweet(filename):
    file = filename
    path = Path.cwd()
    data_Dir = path
    file_Dir = path.joinpath(filename)
    #print(data_Dir)
    tweets = pd.read_csv(file_Dir, dtype='str') # This is a dataframe of actors on twitter and their information
    tweets.head(10)
    processed = filename.stem + "_processed.csv"
    out_path = data_Dir.joinpath(processed)
    #print(out_path)
    tweets_processed = tweets.loc[:, ['id', 'text']]
    print("# of tweets: ", len(tweets_processed))
    tweets_processed = tweets_processed.drop_duplicates(subset=['text'])
    print("# of tweets: ", len(tweets_processed))
    total_Dir = path.joinpath('ukraine_turked')
    total_file = total_Dir.joinpath('ukraine_turked_total')
    total_turked = pd.read_csv(total_file, dtype='str' , names=['id', 'tweet', 'label', 'time'])
    total_id = list(total_turked['id'])
    tweets_processed = tweets_processed[tweets_processed.id.isin(total_id) ==False]
    print("# of tweets: ", len(tweets_processed))
    tweets_processed.to_csv(out_path, header=False, index=False)
    print("Tweets processed")

if __name__ == "__main__":
    process_tweet()
    ##  python data/process_tweets.py data/nagorno2.csv  Ran this to make this happen
