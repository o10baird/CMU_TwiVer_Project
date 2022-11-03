### TWARC2 Pulls

# This is a reference document to document what TWARC pulls were made 

twarc2 csv nagorno.json nagorno.csv

## Pull 1
''' twarc2 searches --archive --limit 10000 --start-time 2021-11-02 --end-time 2022-02-23 --combine-queries ukraine.txt ukraine.json '''

Pulled 10081 tweets using the search terms ukraine, russia, Putin, invade

## Pull 2
''' twarc2 searches --archive --limit 10000 --start-time 2020-08-26 --end-time 2020-09-26 --combine-queries nagorno.txt nagorno.json '''

Pulled 10081 tweets using the search terms ukraine, russia, Putin, invade


## Pull 3 

''' twarc2 searches --archive --limit 10000 --start-time 2020-08-26 --end-time 2020-09-26 --combine-queries nagorno.txt nagorno2.json '''

''' twarc2 csv nagorno2.json nagorno2.csv '''

Pulled 10081 tweets using the search terms ukraine, russia, Putin, invade

### Fresh pulls

## Pull Ukraine month before
''' twarc2 searches --archive --limit 10000 --start-time 2021-12-23 --end-time 2022-01-23 --combine-queries ukraine.txt ukraine_month.json '''

twarc2 csv ukraine_month.json ukraine_month.csv

## Pull Ukraine week before
''' twarc2 searches --archive --limit 10000 --start-time 2022-02-10 --end-time 2022-02-17 --combine-queries ukraine.txt ukraine_week.json '''

twarc2 csv ukraine_week.json ukraine_week.csv

## Pull Ukraine day before
''' twarc2 searches --archive --limit 10000 --start-time 2022-02-22 --end-time 2022-02-23 --combine-queries ukraine.txt ukraine_day.json '''

twarc2 csv ukraine_day.json ukraine_day.csv

## Pull Nagorno month before
''' twarc2 searches --archive --limit 10000 --start-time 2020-07-26 --end-time 2020-08-26 --combine-queries nagorno.txt nagorno_month.json '''

twarc2 csv nagorno_month.json nagorno_month.csv

## Pull Nagorno week before
''' twarc2 searches --archive --limit 10000 --start-time 2020-09-13 --end-time 2020-09-19 --combine-queries nagorno.txt nagorno_week.json '''

twarc2 csv nagorno_week.json nagorno_week.csv

## Pull Nagorno day before
''' twarc2 searches --archive --limit 10000 --start-time 2022-02-22 --end-time 2022-02-23 --combine-queries nagorno.txt nagorno_day.json '''

twarc2 csv nagorno_day.json nagorno_day.csv