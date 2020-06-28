"""

    Helper Functions.

    Author: JHB-EN2.

    Description: These helper functions are to be used to plot data for EDA 

"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob

# Import datasets
# train_data = pd.read_csv('/resources/datasets/train.csv')


# Functions
def getVocab(df):
    """
    count total vocabulary from Dataframe message
    
    Parameters
    ----------
        df (DataFrame): input dataframe
    Returns
    -------
        vocab (list): list of all words that occur atleast once in the tweets
    Examples
    --------
        >>> vocab = getVocab(df = df['column_name'])
    """
    vocab = list()
    for tweet in df:
        for token in tweet:
            vocab.append(token)
    return vocab
def wordFrequencyDict(df, target, vocab):
    """
    function that returns dictionary of word frequencies from a dataframe
    Parameters
    ----------
        df (DataFrame): dataframe containing X_features
        target (str): column to calculate
    Returns
    -------
        word_frequency_dict (dict): frequency
    Examples
    --------
        >>> word_frequency_dict = word_frequency_dict(train_data, 'target')
        >>> word_frequency_dict['Anti'][-5:]
        [('philippine', 0.004),
        ('guaranteed', 0.004),
        ('experimental', 0.004),
        ('drug', 0.004),
         ('validated', 0.004)]
    """

    word_frequency_dict = {}
    for label in df[target].unique():
        data = df[df[target] == label]
        class_vocab = getVocab(data)
        length_of_vocab = len(class_vocab)
        ordered_class_words = Counter(class_vocab).most_common()
        ordered_class_words_freq = list()
        for paired_tuple in ordered_class_words:
            word, count = paired_tuple
            word_frequency = round((count / length_of_vocab) * 100, 3)
            ordered_class_words_freq.append((word, word_frequency))
            word_frequency_dict[label] = ordered_class_words_freq
    return word_frequency_dict
def getClassWords(word_frequency_dict):
    """
    doctstring
    Parameters
    ----------
        word_frequency_dict (dict): list of word frequencies 
    Returns
    -------
        class_words (dict): dictionary of most frequently occuring words by class
    Examples
    --------
        >>> getClassWords(word_frequency_dict)
        >>> class_words['News'][:5]
        ['urlweb', 'change', 'climate', 'trump', 'global']
    """
    frequency_threshold = 0.01
    class_words = {}
    for label, value in word_frequency_dict.items():
        words = list()
        for paired_tuple in value:
            word, freq = paired_tuple
            if freq > frequency_threshold:
                words.append(word)
        class_words[label] = words
    return class_words
def getOrder(class_words, df):
    """
    
    """
    pro_spec_words = list(set(class_words['Pro']) - set(class_words['Neutral']).union(set(class_words['Anti'])).union(set(class_words['News'])))
    neutral_spec_words = list(set(class_words['Neutral']) - set(class_words['Pro']).union(set(class_words['Anti'])).union(set(class_words['News'])))
    anti_spec_words = list(set(class_words['Anti']) - set(class_words['Pro']).union(set(class_words['Neutral'])).union(set(class_words['News'])))
    news_spec_words = list(set(class_words['News']) - set(class_words['Pro']).union(set(class_words['Neutral'])).union(set(class_words['Anti'])))

    # label_specific_words = dict(
    #     Pro = pro_spec_words, Neutral = neutral_spec_words, Anti = anti_spec_words, News = news_spec_words
    #     )
    vocab = getVocab(df)

    class_specific_words = pro_spec_words + neutral_spec_words + anti_spec_words + news_spec_words
    class_specific_words[:5], len(class_specific_words)

    ordered_words = Counter(vocab).most_common()
    return ordered_words
def topNWords(ordered_words, n = 5000):
    """
    count total vocabulary from Dataframe message
    
    Parameters
    ----------
        df (DataFrame): input dataframe
    Returns
    -------
        vocab (list): list of all words that occur atleast once in the tweets 
    """
    most_common = list()
    for word in ordered_words[:n]:
        most_common.append(word[0])
    return most_common
def removeInfrequentWords(tweet, include):
    """
    Function that goes through the words in a tweet,
    determines if there are any words that are not in
    the top n words and removes them from the tweet
    and return the filtered tweet.
    Parameters
    ----------
        tweet (list): list tokens to be flitered
        top_n_words (int): number of tweets to keep
    Returns
    -------
        filt_tweet (list): list of top n words
    Examples
    --------
        >>> bag_of_words = [('change', 12634),
                            ('climate', 12609),
                            ('rt', 9720),
                            ('urlweb', 9656),
                            ('global', 3773)],
        >>> removeInfrequentWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'],2)
            ['change', 'climate']    
    """
    
    filt_tweet = list()
    for token in tweet:
        if token in include:
            filt_tweet.append(token)
    return filt_tweet
def allVocab(df, target = 'tweets_clean'):
    all_vocab = list()
    for tweet in df[target]:
        for token in tweet:
            all_vocab.append(token)
    return all_vocab
def removeCommonWords(tweet, very_common_words):
    """
    removes the most common words from a list of given words
    Parameters
    ----------
        tweet (list): list of words to be cleaned
    Returns
    -------
        filt_tweet (list): list of cleaned words
    Examples
    --------
        >>> very_common_words = ['change', 'climate', 'rt', 'urlweb', 'global']
        >>> removeCommonWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'])
            ['equation']
    """
    filt_tweet = list()
    for token in tweet:
        if token not in very_common_words:
            filt_tweet.append(token)
    return filt_tweet
def lengthOfTweet(tweet):
    """
    return the length of each tweet in the dataset
    
    Parameters
    ----------
        tweet (list): list of a tweet
    Returns
    -------
        length (int): length of each tweet
    
    """
    length = len(tweet)
    return length
def plotDist(df, target = 'len_of_tweet'):
    sns.distplot(df[target])
    plt.show()
def plotCounts(df):
    sns.countplot(data = df, x = 'target', palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})
    plt.title('Count of Sentiments\n')
    plt.xlabel('\nSentiment')
    plt.ylabel('Count\n')
    plt.show()
def plotWordCloud(data, label):
    """
    the plot of the most common use of words that appear bigger than words that
    appear infrequent in a text document by each sentiment
    
    Parameters
    ----------
        data (DataFrame): input of dataframe
        label (int): sentiment variable from dataframe
        
    Returns
    -------
        fig (matplotlib.figure.Figure): Final plot to be displayed
    Examples
    --------
        data = train_data[train_data['target'] == 'Pro']
        plotWordCloud(data, label = 'Sentiment = Pro')
    """
    words = list()
    for tweet in data['tweets_clean']:
        for token in tweet:
            words.append(token)
    words = ' '.join(words)

    from wordcloud import WordCloud
    wordcloud = WordCloud(contour_width=3, contour_color='steelblue').generate(words)

    # Display the generated image:
    plt.figure(figsize = (10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(label, fontsize = 30)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
def getPolarityScores(tweet):
    """
    return the polarity score of each tweet in the dataset
    
    Parameters
    ----------
        tweet (list): list of a tweet
    Returns
    -------
        scores (dict): dictionary of polarity scores
    
    """
    tweet = ' '.join(tweet)
    # analyse the sentiment
    sid = SentimentIntensityAnalyzer()
    # get polarity score of each data
    scores = sid.polarity_scores(tweet) 
    # return the polarity scores
    return scores
def applyScores(data):
    nltk_scores = dict(compound = list(), negative = list(), neutral = list(), positive = list())
    for tweet in data['tweets_clean']:
        output = getPolarityScores(tweet)
        nltk_scores['compound'].append(output['compound'])
        nltk_scores['negative'].append(output['neg'])
        nltk_scores['neutral'].append(output['neu'])
        nltk_scores['positive'].append(output['pos'])

    # concatenate the output from above into the main dataset
    if 'compound' in data.columns:
    
        # drop the columns if this has been executed before
        data.drop(['compound', 'negative', 'neutral', 'positive'], axis = 1, inplace = True)
    
        # concatenate a DataFrame version of the nltk_scores dictionary
        data = pd.concat([data, pd.DataFrame(nltk_scores)], axis = 1)
    else:
        # concatenate directly if this is the first execution
        data = pd.concat([data, pd.DataFrame(nltk_scores)], axis = 1)
    return data
def getPolaritySubjectivity(data):
    sentiment_scores = [TextBlob(' '.join(tweet)).sentiment for tweet in data['tweets_clean']]

    # Add output to dataframe
    pol = list()
    subj = list()
    for scores in sentiment_scores:
        pol.append(scores.polarity)
        subj.append(scores.subjectivity)

    data['polarity'] = pol
    data['subjectivity'] = subj
    return data
def violinPlots(data, target = 'target'):
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    for i, column in enumerate(['compound', 'polarity', 'subjectivity']):
        g = sns.violinplot(data = data, target = 'target', y = column, ax = axes[i], palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})
        g.set_title(column)
        if column == "compound":
            g.set_ylabel('Scores')
        else:
            g.set_ylabel(' ')
        g.set_xlabel(' ')
    plt.show()
    return fig
def plotScatter(x, y, df, title):
    """
    display the scatter plot
    
    Parameters
    ----------
        x (str): variable string from dataframe
        y (str): variable string from dataframe
        dict (dict): input of word list
        
    Returns
    -------
        g (plot): display a scatter plot with points of each labelled sentiments
    
    """
    plt.figure(figsize = (8, 5))
    sns.scatterplot(df = df, x = x, y = y, hue = 'target', legend = False, palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})
    plt.title(title, fontsize = 20)
    
    # add annotations one by one with a loop
    for line in range(0,df.shape[0]):
        plt.text(df[x][line], df[y][line], df['target'][line], 
                horizontalalignment='left', size='large', color='black')
    
    plt.show()
def plotAltScatter(df, X = 'compound', y_ = 'polarity', title_ = 'Compound Vs Polarity\n'):
    data = df.groupby('target')[['negative', 'positive', 'neutral', 'compound', 'polarity', 'subjectivity']].mean().reset_index()
    plotScatter(x = X, y = y_, df = data, title = title_)
    plt.xlabel('\nCompound Score')
    plt.ylabel('Polarity\n')
    plt.show()
def arrowScatter(df):
    plt.figure(figsize = (8, 5))
    sns.scatterplot(data = df, x = 'subjectivity', y = 'polarity', color = 'teal', hue = 'target', alpha = 1/3)
    plt.arrow(0, 0.1, 0.99, 1, fc = 'black', ec = '#CCCC00')
    plt.arrow(0, 0.1, 0.99, -1, fc = 'black', ec = '#CCCC00')
    plt.title('Subjectiviy vs Polarity\n')
    plt.xlabel('\nSubjectivity')
    plt.ylabel('Polarity')
    plt.show()
def histPlot(df):
    columns = ['polarity', 'compound']
    fig, axes = plt.subplots(1, len(columns), figsize = (18, 5), sharey = True)
    for i, column in enumerate(columns):
        sns.distplot(df[column], ax = axes[i])
    plt.show()
def polCompNeutralPlot(df):
    # variables of columns from dataframe that are conditioned to zero
    # polarity_mask = df['polarity'] == 0
    # compound_mask = df['compound'] == 0

    #plotting histigram and line on it
    sum_of_pol_and_comp = df['polarity'].add(df['compound'])
    sns.distplot(sum_of_pol_and_comp)
    plt.show()
def polPlusCompScatter(df):
    data = df.groupby('target')[['pol_plus_comp', 'subjectivity']].mean().reset_index()
    plotScatter(x = 'subjectivity', y = 'pol_plus_comp', df = data, title="Polarity & Compound vs Subjectivity")
    plt.show()
def midCloud(df, lower = -0.15, interval = 0.3, sentiment = [-1,0,1,2]):
    data = df[(df['sentiment']==sentiment) & (df['compound'] < lower + interval) & (df['compound'] > lower)]
    plotWordCloud(data, label = f'Neutral where: {lower} < Compound < {lower+interval}')
def upperCloud(df, upper = 0.6, sentiment = [-1,0,1,2]):
    data = df[(df['sentiment']==sentiment)&(df['compound'] > upper)]
    plotWordCloud(data, label = f'compound > {upper}')
def lowerCloud(df, lower = -0.6, sentiment = [-1,0,1,2]):
    data = df[(df['sentiment']==sentiment)&(df['compound'] > lower)]
    plotWordCloud(data, label = f'compound > {lower}')
