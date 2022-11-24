import pandas as pd
import click
from pathlib import Path
from tqdm import tqdm


def process_tweet(filename, data_Dir, total_id):
    tweets = pd.read_csv(filename, dtype='str') # This is a dataframe of actors on twitter and their information
    print(tweets.head(10))
    processed = filename.stem + "_processed.csv"
    out_path = data_Dir.joinpath(processed)
    #tweets_processed = tweets['id', 'text']
    tweets_processed = tweets.loc[:, ['id', 'text']]
    tweets_processed = tweets_processed.drop_duplicates(subset=['text'])
    tweets_processed = tweets_processed.loc[:, ['id', 'text']]
    tweets_processed = tweets_processed[tweets_processed.id.isin(total_id) ==False]
    tweets_processed.to_csv(out_path, header=False, index=False)
    print(file, " Tweets processed")
    print("# of tweets: ", len(tweets_processed))
    new_total_id = total_id.append(tweets_processed['id'].to_list())
    return new_total_id

if __name__ == "__main__":
    path = Path.cwd()
    data_Dir = path.joinpath('new_data')
    total_Dir = path.joinpath('ukraine_turked')
    total_file = total_Dir.joinpath('ukraine_turked_total')
    total_turked = pd.read_csv(total_file, dtype='str' , names=['id', 'tweet', 'label', 'time'])
    total_id = list(total_turked['id'])
    for file in tqdm(data_Dir.glob('**/*')):
        if file.is_file():
            total_id = process_tweet(file, data_Dir, total_id)