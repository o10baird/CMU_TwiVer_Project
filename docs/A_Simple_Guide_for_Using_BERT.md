---
created: 2022-11-15T22:08:57 (UTC -05:00)
tags: []
source: https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
author: Thilina Rajapakse
---

# A Simple Guide On Using BERT for Binary Text Classification. | by Thilina Rajapakse | The Startup | Medium

> ## Excerpt
> An A-to-Z guide on how you can use Google’s BERT for binary text classification tasks with Python and Pytorch. Simple and practical with example code provided.

---
## The A-to-Z guide on how you can use Google’s BERT for binary text classification tasks. I’ll be aiming to explain, as simply and straightforwardly as possible, how to fine-tune a BERT model (with PyTorch) and use it for a binary text classification task.

![](https://miro.medium.com/max/700/1*X3TNZbdu06LWNCpHixIFxQ.jpeg)

Photo by [Andy Kelly](https://unsplash.com/@askkell?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/robot?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## Update Notice II

Please consider using the [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) library as it is easy to use, feature-packed, and regularly updated. The article still stands as a reference to BERT models and is likely to be helpful with understanding how BERT works. However, [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) offers a lot more features, much more straightforward tuning options, all the while being quick and easy to use! The links below should help you get started quickly.

1.  [Binary Classification](https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=---------16------------------)
2.  [Multi-Class Classification](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a?source=---------15------------------)
3.  [Multi-Label Classification](https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?source=---------13------------------)
4.  [Named Entity Recognition (Part-of-Speech Tagging)](https://towardsdatascience.com/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?source=---------14------------------)
5.  [Question Answering](https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?source=---------12------------------)
6.  [Sentence-Pair Tasks and Regression](https://medium.com/swlh/solving-sentence-pair-tasks-using-simple-transformers-2496fe79d616?source=---------9------------------)
7.  [Conversational AI](https://towardsdatascience.com/how-to-train-your-chatbot-with-simple-transformers-da25160859f4?source=---------6------------------)
8.  [Language Model Fine-Tuning](https://medium.com/skilai/language-model-fine-tuning-for-pre-trained-transformers-b7262774a7ee?source=---------4------------------)
9.  [ELECTRA and Language Model Training from Scratch](https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=---------2------------------)
10.  [Visualising Model Training](https://medium.com/skilai/to-see-is-to-believe-visualizing-the-training-of-machine-learning-models-664ef3fe4f49?source=---------10------------------)

## Update Notice I

**_In light of the update to the library used in this article (HuggingFace updated the_** `**_pytorch-pretrained-bert_**` **_library to_** `[**_pytorch-transformers_**](https://github.com/huggingface/pytorch-transformers)`**_), I have written a new_** [**_guide_**](https://medium.com/@chaturangarajapakshe/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca) **_as well as a new_** [**_repo_**](https://github.com/ThilinaRajapakse/pytorch-transformers-classification)**_. If you are starting out with Transformer models, I recommend using those as the code has been cleaned up both on my end and in the Pytorch-Transformers library, greatly streamlining the whole process. The new repo also supports XLNet, XLM, and RoBERTa models out of the box, in addition to BERT, as of September 2019._**

## 1\. Intro

## Let’s talk about what we are going to (and not going to) do.

_Before we begin, let me point you towards the_ [_github repo_](https://github.com/ThilinaRajapakse/BERT_binary_text_classification) _containing all the code used in this guide. All code in the repo is included in the guide here, and vice versa. Feel free to refer to it anytime, or clone the repo to follow along with the guide._

If your internet wanderings have led you here, I guess it’s safe to assume that you have heard of BERT, the powerful new language representation model, open-sourced by Google towards the end of 2018. If you haven’t, or if you’d like a refresher, I recommend giving their [paper](https://arxiv.org/pdf/1810.04805.pdf) a read as I won’t be going into the technical details of how BERT works. If you are unfamiliar with the Transformer model (or if words like “attention”, “embeddings”, and “encoder-decoder” sound scary), check out this _brilliant_ [article](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar. You don’t necessarily need to know everything about BERT (or Transformers) to follow the rest of this guide, but the above links should help if you wish to learn more about BERT and Transformers.

Now that we’ve gotten what we _won’t_ do out of the way, let’s dig into what we _will_ do, shall we?

-   Getting BERT downloaded and set up. We will be using the PyTorch version provided by the amazing folks at Hugging Face.
-   Converting a dataset in the .**_csv_** format to the .**_tsv_** format that BERT knows and loves.
-   Loading the **._tsv_** files into a notebook and converting the text representations to a feature representation (think numerical) that the BERT model can work with.
-   Setting up a pretrained BERT model for fine-tuning.
-   Fine-tuning a BERT model.
-   Evaluating the performance of the BERT model.

_One last thing before we dig in, I’ll be using three Jupyter Notebooks for data preparation, training, and evaluation. It’s not strictly necessary, but it felt cleaner to separate those three processes._

## 2\. Getting set up

## Time to get BERT up and running.

1.  Create a virtual environment with the required packages. You can use any package/environment manager, but I’ll be using Conda.  
    `conda create -n bert python pytorch pandas tqdm`  
    `conda install -c anaconda scikit-learn   `_(Note: If you run into any missing package error while following the guide, go ahead and install them using your package manager. A google search should tell you how to install a specific package.)_
2.  Install the PyTorch version of BERT from Hugging Face.  
    `pip install pytorch-pretrained-bert`
3.  To do text classification, we’ll obviously need a text classification dataset. For this guide, I’ll be using the Yelp Reviews Polarity dataset which you can find [here](https://course.fast.ai/datasets) on fast.ai. (Direct [download link](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz) for any lazy asses, I mean busy folks.)  
    Decompress the downloaded file and get the **_train.csv_**, and **_test.csv_** files. For reference, the path to my **_train.csv_** file is `<starting_directory>/data/train.csv`

## 3\. Preparing data

## Before we can cook the meal, we need to prepare the ingredients! (Or something like that. <Insert proper analogy here>)

Most datasets you find will typically come in the **_csv_** format and the Yelp Reviews dataset is no exception. Let’s load it in with pandas and take a look.

As you can see, the data is in the two **_csv_** files `train.csv` and `test.csv`. They contain no headers, and two columns for the label and the text. The labels used here feel a little weird to me, as they have used 1 and 2 instead of the typical 0 and 1. Here, a label of 1 means the review is bad, and a label of 2 means the review is good. I’m going to change this to the more familiar 0 and 1 labelling, where a label 0 indicates a bad review, and a label 1 indicates a good review.

Much better, am I right?

BERT, however, wants data to be in a **_tsv_** file with a specific format as given below (Four columns, and no header row).

-   Column 0: An ID for the row
-   Column 1: The label for the row (should be an int)
-   Column 2: A column of the same letter for all rows. BERT wants this so we’ll give it, but we don’t have a use for it.
-   Column 3: The text for the row

Let’s make things a little BERT-friendly.

For convenience, I’ve named the test data as dev data. The convenience stems from the fact that BERT comes with data loading classes that expects **_train_** and **_dev_** files in the above format. We can use the train data to train our model, and the dev data to evaluate its performance. BERT’s data loading classes can also use a **_test_** file but it expects the **_test_** file to be unlabelled. Therefore, I will be using the train and dev files instead.

Now that we have the data in the correct form, all we need to do is to save the **_train_** and **_dev_** data as **_.tsv_** files.

That’s the eggs beaten, the chicken thawed, and the veggies sliced. Let’s get cooking!

## 4\. Data to Features

## The final step before fine-tuning is to convert the data into features that BERT uses. Most of the remaining code was adapted from the HuggingFace example _run\_classifier.py, found_ [_here_](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)_._

Now, we will see the reason for us rearranging the data into the **_.tsv_** format in the previous section. It enables us to easily reuse the example classes that come with BERT for our own binary classification task. Here’s how they look.

The first class, **InputExample**, is the format that a single example of our dataset should be in. We won’t be using the `text_b` attribute since that is not necessary for our binary classification task. The other attributes should be fairly self-explanatory.

The other two classes, **DataProcessor** and **BinaryClassificationProcessor,** are helper classes that can be used to read in **.tsv** files and prepare them to be converted into _features_ that will ultimately be fed into the actual BERT model.

The **BinaryClassificationProcessor** class can read in the `**_train.tsv_**` and `**_dev.tsv_**` files and convert them into lists of **InputExample** objects.

So far, we have the capability to read in **_tsv_** datasets and convert them into **InputExample** objects. BERT, being a neural network, cannot directly deal with text as we have in **InputExample** objects. The next step is to convert them into **InputFeatures.**

BERT has a constraint on the maximum length of a sequence after tokenizing. For any BERT model, the maximum sequence length after tokenization is 512. But we can set any sequence length equal to or below this value. For faster training, I’ll be using 128 as the maximum sequence length. A bigger number may give better results if there are sequences longer than this value.

An **InputFeature** consists of purely numerical data (with the proper sequence lengths) that can then be fed into the BERT model. This is prepared by tokenizing the text of each example and truncating the longer sequence while padding the shorter sequences to the given maximum sequence length (128). I found the conversion of **InputExample** objects to **InputFeature** objects to be quite slow by default, so I modified the conversion code to utilize the _multiprocessing_ library of Python to significantly speed up the process.

We will see how to use these methods in just a bit.

_(Note: I’m switching to the training notebook.)_

First, let’s import all the packages that we’ll need, and then get our paths straightened out.

In the first cell, we are importing the necessary packages. In the next cell, we are setting some paths for where files should be stored and where certain files can be found. We are also setting some configuration options for the BERT model. Finally, we will create the directories if they do not already exist.

Next, we will use our **BinaryClassificationProcessor** to load in the data, and get everything ready for the tokenization step.

Here, we are creating our **BinaryClassificationProcessor** and using it to load in the train examples. Then, we are setting some variables that we’ll use while training the model. Next, we are loading the pretrained tokenizer by BERT. In this case, we’ll be using the _bert-base-cased_ model.

The `convert_example_to_feature` function expects a tuple containing _an example, the label map, the maximum sequence length, a tokenizer, and the output mode_. So lastly, we will create an examples list ready to be processed (tokenized, truncated/padded, and turned into **InputFeatures**) by the `convert_example_to_feature` function.

Now, we can use the multi-core goodness of modern CPU’s to process the examples (relatively) quickly. My Ryzen 7 2700x took about one and a half hours for this part.

**_Your notebook should show the progress of the processing rather than the ‘HBox’ thing I have here. It’s an issue with uploading the notebook to Gist._**

_(Note: If you have any issues getting the multiprocessing to work, just copy paste all the code up to, and including, the multiprocessing into a python script and run it from the command line or an IDE. Jupyter Notebooks can sometimes get a little iffy with multiprocessing. I’ve included an example script on github named_ `_converter.py_`_)_

Once all the examples are converted into features, we can pickle them to disk for safekeeping (I, for one, do **not** want to run the processing for another one and a half hours). Next time, you can just unpickle the file to get the list of features.

Well, that was a **lot** of data preparation. You deserve a coffee, I’ll see you for the training part in a bit. (Unless you already had your coffee while the processing was going on. In which case, kudos to efficiency!)

## 5\. Fine-tuning BERT (finally!)

## Had your coffee? Raring to go? Let’s show BERT how it’s done! (Fine tune. Show how it’s done. Get it? I might be bad at puns.)

Not much left now, let’s hope for smooth sailing. (Or smooth.. cooking? I forgot my analogy somewhere along the way. Anyway, we now have all the ingredients in the pot, and all we have to do is turn on the stove and let thermodynamics work its magic.)

HuggingFace’s pytorch implementation of BERT comes with a function that automatically downloads the BERT model for us (have I mentioned I love these dudes?). I stopped my download since I have terrible internet, but it shouldn’t take long. It’s only about 400 MB in total for the base models. Just wait for the download to complete and you are good to go.

Don’t panic if you see the following output once the model is downloaded, I know it looks panic inducing but this is actually the expected behavior. The **_not initialized_** things are not meant to be initialized. Intentionally.

```
INFO:pytorch_pretrained_bert.modeling:Weights of BertForSequenceClassification not initialized from pretrained model: ['classifier.weight', 'classifier.bias']INFO:pytorch_pretrained_bert.modeling:Weights from pretrained model not used in BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
```

_(Tip: The model will be downloaded into a temporary folder. Find the folder by following the path printed on the output once the download completes and copy the downloaded file to the_ `**_cache/_**` _directory. The file should be a compressed file in_ **_.tar.gz_** _format. Next time, you can just use this downloaded file without having to download it all over again. All you need to do is comment out the line that downloaded the model, and uncomment the line below it.)_

We just need to do a tiny bit more configuration for the training. Here, I’m just using the default parameters.

Setting up our DataLoader for training..

Training time!

Now we’ve trained the BERT model for one epoch, we can evaluate the results. Of course, more training will likely yield better results but even one epoch should be sufficient for proof of concept (hopefully!).

In order to be able to easily load our fine-tuned model, we should save it in a specific way, i.e. the same way the default BERT models are saved. Here is how you can do that.

-   Go into the `**_outputs/yelp_**` directory where the fine tuned models will be saved. There, you should find 3 files; `**_config.json_**`**_,_** `**_pytorch_model.bin_**`**_,_** `**_vocab.txt_**`**_._**
-   Archive the two files (I use 7zip for archiving) **_config.json,_** and **_pytorch\_model.bin_** into a **_.tar_** file.
-   Compress the **_.tar_** file into **_gzip_** format. Now the file should be something like `**_yelp.tar.gz_**`
-   Copy the compressed file into the `**_cache/_**` directory.

We will load this fine tuned model in the next step.

## 6\. Evaluation

## Time to see what our fine-tuned model can do. (We’ve cooked the meal, let’s see how it tastes.)

_(Note: I’m switching to the evaluation notebook)_

Most of the code for the evaluation is very similar to the training process, so I won’t go into too much detail but I’ll list some important points.

-   BERT\_MODEL parameter should be the name of your fine-tuned model. For example, `**_yelp.tar.gz_**`**_._**
-   The tokenizer should be loaded from the vocabulary file created in the training stage. In my case, that would `**_outputs/yelp/vocab.txt_**` (or the path can be set as `OUTPUT_DIR + vocab.txt` )
-   This time, we’ll be using the `BinaryClassificationProcessor` to load in the `_dev.tsv_` file by calling the `get_dev_examples` method.
-   Double check to make sure you are loading the fine-tuned model and not the original BERT model. 😅

Here’s my notebook for the evaluation.

With just one single epoch of training, our BERT model achieves a 0.914 Matthews correlation coefficient (Good measure for evaluating unbalanced datasets. Sklearn doc [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)). With more training, and perhaps some hyperparameter tuning, we can almost certainly improve upon what is already an impressive score.

## 7\. Conclusion

BERT is an incredibly powerful language representation model that shows great promise in a wide variety of NLP tasks. Here, I’ve tried to give a basic guide to how you might use it for binary text classification.

As the results show, BERT is a very effective tool for binary text classification, not to mention all the other tasks it has already been used for.

**Reminder: Github repo with all the code can be found** [**here**](https://github.com/ThilinaRajapakse/BERT_binary_text_classification)**.**
