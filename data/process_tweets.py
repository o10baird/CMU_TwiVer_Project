import pandas as pd
import click
from pathlib import Path

@click.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path))
def process_tweet(filename):
    file = filename
    path = Path.cwd()
    data_Dir = path.joinpath('data')
    file_Dir = path.joinpath(filename)
    tweets = pd.read_csv(file_Dir, dtype='str') # This is a dataframe of actors on twitter and their information
    tweets.head(10)
    processed = filename.stem + "_processed.csv"
    out_path = data_Dir.joinpath(processed)
    print(out_path)
    tweets_processed = tweets.loc[:, ['id', 'text']]
    tweets_processed.to_csv(out_path, header=False, index=False)
    print("Tweets processed")

if __name__ == "__main__":
    process_tweet()
    ##  python data/process_tweets.py data/nagorno2.csv  Ran this to make this happen
