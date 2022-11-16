## Acknowledgement
This data is a subset of Multi-Domain Sentiment Dataset(https://www.cs.jhu.edu/~mdredze/datasets/sentiment/). Please do not distribute this dataset outside the class.

## Data
The structure of this data set is show as follows

.
   |-all.review.vec.txt
   |-test
   |---negative
   |---positive
   |-train
   |---negative
   |---positive


Pretrained word embedding:
-------------
"all.review.vec.txt" is a pre-trained word embedding on a larger amazon review corpus. The first line is "56050 100", which indicates that there are 56050 words in total and the dimension size for each word is 100. Each following line is the vector for a specific word. The line format is:

    <word> <dim0> <dim1> <dim2> ... <dim99>

Training/Testing data:
-------------
On top level there are wo folders containing the training and testing split of texts. Inside each train/test split there are the splits between positive and negative reviews.
