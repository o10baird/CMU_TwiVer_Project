---
created: 2022-11-15T22:10:33 (UTC -05:00)
tags: []
source: https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
author: 
---

# Tweets Cleaning with Python – Catris Code

> ## Excerpt
> A tweet can contain a lot of things, from plain text, mentions, hashtags, links, punctuations to many other things. When you’re working on a data science or machine learning project, you may …

---
A tweet can contain a lot of things, from plain text, mentions, hashtags, links, punctuations to many other things. When you’re working on a data science or machine learning project, you may want to remove these things first before you process the tweets further. I am going to show you the steps needed to be performed to clean tweets.

I usually deal with tweets in English or Bahasa Indonesia, so the tutorial below is only going to process characters used in these languages. If you are dealing with tweets in languages that use non Latin alphabets, you may need different processes.

This is the basics of text preprocessing in a tweet that I usually do.

## Text Preprocessing: Step by Step Examples

Let’s start with the following tweet, which I took from National Geographic’s official Twitter account. This tweet is going to be the data we are working on, but you can always try with a different tweet if you want to.

```
# This is the tweet that we are going to work with.
tweet = "Get ready for #NatGeoEarthDay! Join us on 4/21 for an evening of music and celebration, exploration and inspiration https://on.natgeo.com/3t0wzQy."
```

### Lowercasing all the letters

This step is important to make sure that all your letters are in uniform.

```
temp = tweet.lower()
temp
```

‘get ready for #natgeoearthday! join us on 4/21 for an evening of music and celebration, exploration and inspiration https://on.natgeo.com/3t0wzqy.

### Removing hashtags and mentions

Hashtags and mentions are common in tweets. There are cases where you want to remove them so you only get the clean content of a tweet without all these elements. You can remove these hashtags and mentions using regex.

```
import re

temp = re.sub("@[A-Za-z0-9_]+","", temp)
temp = re.sub("#[A-Za-z0-9_]+","", temp)
temp
```

‘get ready for ! join us on 4/21 for an evening of music and celebration, exploration and inspiration https://on.natgeo.com/3t0wzqy.’

### Removing links

Links are usually not necessary for text processing, so it’s better to remove them from your text.

```
temp = re.sub(r"http\S+", "", temp)
temp = re.sub(r"www.\S+", "", temp)
temp
```

‘get ready for ! join us on 4/21 for an evening of music and celebration, exploration and inspiration ‘

### Removing punctuations

Depending on your needs, you may not need punctuations such as period, comma, exclamation mark, question mark, etc.

```
temp = re.sub('[()!?]', ' ', temp)
temp = re.sub('\[.*?\]',' ', temp)
temp
```

‘get ready for join us on 4/21 for an evening of music and celebration, exploration and inspiration ‘

### Filtering non-alphanumeric characters

The previous step may have removed the punctuations, including all the non-alphanumeric characters, but just to be sure, we can remove all letters except the alphabets (a-z) and numbers (0-9). The sign ^ below means except.

```
temp = re.sub("[^a-z0-9]"," ", temp)
temp
```

‘get ready for join us on 4 21 for an evening of music and celebration exploration and inspiration ‘

### Tokenization

The term tokenization sounds very sophisticated, but it’s actually not. In tokenization, you basically tokenize your text into tokens. And what is a token? In this case, you split your text into smaller components, for example a paragraph into a list of sentences, or a sentence into a list of words.

Library such as [nltk](https://www.nltk.org/) provides functions such as `word_tokenize()` or `sent_tokenize()` to help you with this. However, if you just want a simple tokenizing step where you split your text into words into a list, then you can do it as simple as the following code. The result will give you a list of words from your text.

\[‘get’, ‘ready’, ‘for’, ‘join’, ‘us’, ‘on’, ‘4’, ’21’, ‘for’, ‘an’, ‘evening’, ‘of’, ‘music’, ‘and’, ‘celebration’, ‘exploration’, ‘and’, ‘inspiration’\]

### Stop words removal

What is a stop word? Stop words are words that are considered unimportant to the meaning of a text. These words may seem important to us, humans, but to machine these words may be considered nuisance to the processing steps.

It’s also important to keep in mind that stop words are largely language-dependent. In English, you have stop words such as for, to, and, or, in, out, etc. In other languages, such as Indonesian, you may need different sets of stop words to work with.

Here I first defined a list of stop words in English. Then, I match each token with each stop word. If a token isn’t found in the list of stop words, the token gets saved, otherwise it’s not saved. In the end, you join all the words into one text again.

```
stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]

temp = [w for w in temp if not w in stopwords]
temp = " ".join(word for word in temp)
```

‘get ready join us 4 21 evening music celebration exploration inspiration’

___

Those were all the basic processes you need to know when you need to preprocess a text containing a tweet. Now we move to the next section.

___

## Text Preprocessing: From Start to Finish

I hope you understand the steps I have explained above. Now we can combine all those lines of code into one function that we can call and pass an argument to. The function then returns a clean text that is ready for you to work with.

Keep in mind that the order of steps here are not absolute. You can arrange them around depending on your text and your needs. The code below is what I found to be the most effective on the data I usually work with, but in case you find another pattern of data, you can always work them out differently.

```
import numpy as np
import re

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp
```

Now, let’s apply the function to a set of tweets.

```
tweets = ["Get ready for #NatGeoEarthDay! Join us on 4/21 for an evening of music and celebration, exploration and inspiration https://on.natgeo.com/3t0wzQy.",
"Coral in the shallows of Aitutaki Lagoon, Cook Islands, Polynesia https://on.natgeo.com/3gkgq4Z",
"Don't miss our @reddit AMA with author and climber Mark Synnott who will be answering your questions about his historic journey to the North Face of Everest TODAY at 12:00pm ET! Start submitting your questions here: https://on.natgeo.com/3ddSkHk @DuttonBooks"]

results = [clean_tweet(tw) for tw in tweets]
results
```

\[‘get ready join us 4 21 evening music celebration exploration inspiration’,

‘coral shallows aitutaki lagoon cook islands polynesia’,

‘dont miss our ama with author climber mark synnott who will be answering your questions about his historic journey north face everest today at 12 00pm et start submitting your questions here’\]

___

## Text Preprocessing: In Summary

This whole article may seem long and complicated, but I assure you I can summarize all the steps above to the following basic processes.

1.  Lowercasing all your letters
2.  Removing hashtags, mentions, and links
3.  Punctuations removal (including filtering non-alphanumeric characters if necessary)
4.  Tokenization
5.  Stop words removal

There are other processes worth mentioning such as lemmatization or stemming that I didn’t explain here, but they may require higher computing powers that can slow down your computer. I have learned that the processes I use above sometimes already give decent results, and sacrificing running time to perform lemmatization and stemming doesn’t always lead to better outcome.

However, it also always comes down to your data and your use cases, so you can always try different approaches. After all, we always need to keep experimenting to improve.

I hope you find this article useful. Thanks for reading!

___

## Further readings

-   Language Processing and Python [http://www.nltk.org/book/ch01.html](http://www.nltk.org/book/ch01.html)
-   Tokenization in Lexical Analysis [https://en.wikipedia.org/wiki/Lexical\_analysis#Tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
-   NLTK’s list of English stop words [https://gist.github.com/sebleier/554280](https://gist.github.com/sebleier/554280)
