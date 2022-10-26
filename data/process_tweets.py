import pandas as pd
import urllib.parse as urlparse
import sys
from pathlib import Path


tweets_filename = sys.argv[1]
path = Path.cwd()
path_root = path.parent
data_Dir = path_root.joinpath('data')
file_Path = data_Dir.joinpath("ukraine.csv")
print(file_Path)
tweets = pd.read_csv(file_Path) # This is a dataframe of actors on twitter and their information
tweets.head(10)
out_path = data_Dir.joinpath("ukraine_processed.csv")
tweets_processed = tweets.loc[:, ['id', 'text']]
tweets_processed.to_csv(out_path, header=False, index=False)