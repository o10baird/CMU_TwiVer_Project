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
# import twarc
import subprocess
import pandas as pd
import urllib.parse as urlparse

import subprocess
import pandas as pd
import urllib.parse as urlparse
from datetime import date, timedelta


# Commands
run_num = 0


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

week_start = date(2022, 2, 10)
week_end = date(2022, 2, 17)
for single_date in daterange(week_start, week_end):
    print(single_date.strftime("%Y-%m-%d"))
    command = f"twarc2 searches --archive --limit 10000 --start-time {single_date.strftime('%Y-%m-%d')} --end-time {single_date.strftime('%Y-%m-%d')} --combine-queries new_data/ukraine_{single_date.strftime('%Y-%m-%d')}.json"

month_start = date(2021,12,23)
month_end = date(2022,1,23)
for single_date in daterange(month_start, month_end):
    print(single_date.strftime("%Y-%m-%d"))
    command = f"twarc2 searches --archive --limit 10000 --start-time {single_date.strftime('%Y-%m-%d')} --end-time {single_date.strftime('%Y-%m-%d')} --combine-queries new_data/ukraine_{single_date.strftime('%Y-%m-%d')}.json"