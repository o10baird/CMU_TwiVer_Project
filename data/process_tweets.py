

'''
This is a little script I put together to use TWARC2
for scraping tweets from the command line.

Link to TWARC2 Documentation --> https://twarc-project.readthedocs.io/en/latest/twarc2_en_us/

Make sure to update your .env file to use TWARC2. You'll need to 
do the following...
1. Install TWARC2. I've updated the requirements.txt so you should just have to re-run the pip install -r requirements.txt command.
2. Run twarc2 configure to configure the tool with your Twitter API keys. You'll need to include API Key, API Secret, and Bearer Token.
3. Update this script to build out whatever queries you want to run.

I've included some documentation and comments below
'''
import twarc
import subprocess
import pandas as pd
import urllib.parse as urlparse
# Import the necessary package to process data in JSON format
#import sys
# We use the file saved from last step as example
#tweets_filename = sys.argv[1]
#tweets_file = open(tweets_filename, "r")

#import csv
#f2 = open(sys.argv[1].replace(".txt", "_processed.csv"), 'w')
#fout = csv.writer(f2)

# Prep the data
tweets = pd.read_csv('ukraine.csv') # This is a dataframe of actors on twitter and their information
tweets.head(10)
#actors_list = actors['conceptFrom'].to_list() # Create a list from the twitter handles
#actors_handles = [handle for handle in actors_list if handle.startswith('@')] # Filter out the handles that don't start with @

# Commands
run_num = 0
#for handle in actors_handles: # Loop through the list of handles
#    strip_handle = handle.replace('@', '') # Strip the @ from the handle so that you can have a valid filename
#    encoded_handle = urlparse.encode(handle) # Encode the handle so that it can be used in the query. Honestly though, the whole query needs to be encoded I think. Still researching
#    command = f"twarc2 timeline --limit 5 {encoded_handle} test_files\{strip_handle}.jsonl" # Build the command
#    subprocess.run(command.split()) # Run the command
#    run_num += 1
#    print(f"Run {run_num} of {len(actors_handles)}complete") # Print the run number so you can see how far along you are

#for line in tweets_file:
#    try:
#        # Read in one line of the file, convert it into a json object 
#        tweet = json.loads(line.strip())
#        if 'text' in tweet: # only messages contains 'text' field is a tweet
#            d2w = []
#            d2w.append(tweet['id']) # This is the tweet's id
#            #print tweet['created_at'] # when the tweet posted
#            d2w.append(tweet['text']) # content of the tweet
                        
            #print tweet['user']['id'] # id of the user who posted the tweet
            #print tweet['user']['name'] # name of the user, e.g. "Wei Xu"
#            d2w.append(tweet['user']['screen_name']) # name of the user account, e.g. "cocoweixu"
#            fout.writerow(d2w)
            #hashtags = []
            #for hashtag in tweet['entities']['hashtags']:
            #    hashtags.append(hashtag['text'])
            #print hashtags

#    except:
        # read in a line is not in JSON format (sometimes error occured)
#        continue
