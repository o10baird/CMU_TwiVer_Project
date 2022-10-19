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

# Prep the data
actors = pd.read_csv('actors.csv') # This is a dataframe of actors on twitter and their information
actors_list = actors['conceptFrom'].to_list() # Create a list from the twitter handles
actors_handles = [handle for handle in actors_list if handle.startswith('@')] # Filter out the handles that don't start with @

# Commands
run_num = 0
for handle in actors_handles: # Loop through the list of handles
    strip_handle = handle.replace('@', '') # Strip the @ from the handle so that you can have a valid filename
    encoded_handle = urlparse.encode(handle) # Encode the handle so that it can be used in the query. Honestly though, the whole query needs to be encoded I think. Still researching
    command = f"twarc2 timeline --limit 5 {encoded_handle} test_files\{strip_handle}.jsonl" # Build the command
    subprocess.run(command.split()) # Run the command
    run_num += 1
    print(f"Run {run_num} of {len(actors_handles)}complete") # Print the run number so you can see how far along you are

