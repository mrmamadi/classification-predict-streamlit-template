"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
######################################################################################################
##################################-------------EVERYONE-------------##################################
######################################################################################################

### General git commands
# creates a branch -- git branch branch_name
# switches branches -- git checkout branch_name
# creates and switches branch -- git checkout -b branch_name
# display branches -- git branch
# delete branch -- git branch -d branch_name

### GIT INSTRUCTIONS

# for every task that you work on you must follow this process
# 1. switch to the development branch using "git checkout dev"
# 2. Create a new feature branch using "git checkout -b issue_16"
# 3. Resolve issues then save changes
# 4. stage changes to feature branch using "git add ."
# 5. commit changes THIS IS IMPORTANT with "git commit -m "fixes issue x" where x is the issue number"
# 6. Switch to development branch "git checkout dev"
# 7. Merge the feature branch with "git merge issue_16"
# ### IF YOU HAVE A MERGE CONFLICT, refer to this link: https://docs.google.com/presentation/d/1MyZAy63pEExvF-z9mr3nFHORw-6uo_IQhJlV0656G5U/edit#slide=id.g8a00cae286_0_30
# 8. Delete the branch using "git branch -d issue_16"
# 9. create a pull request by using "git push origin dev"
# 10. The code administrator will review your changes and complete the merge to the development branch

# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Cleaning
import preprocessing as prep
import eda

# Data analysis
from collections import Counter
from wordcloud import WordCloud
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Vectorizer - ADD A SECOND VECTORIZER (MELVA/KGAOGELO)
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load data
raw = pd.read_csv("resources/datasets/train.csv")

######################################################################################################
##################################----------EVERYONE-END------------##################################
######################################################################################################

### Data Preparation
@st.cache(allow_output_mutation=True)
def prepareData(df):
    """
    function to apply data cleaning steps
    Parameters
    ----------
        df (DataFrame): DataFrame to be cleaned
    Returns
    -------
        prep_df (DataFrame): cleaned DataFrame
    """
    prep_df = df.copy()
    target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
    prep_df['target'] = prep_df['sentiment'].map(target_map)
    prep_df = prep.typeConvert(prep_df)
    prep_df['urls'] = prep_df['message'].map(prep.findURLs)
    prep_df = prep.strip_url(prep_df)
    prep_df['handles'] = prep_df['message'].map(prep.findHandles)
    prep_df['hash_tags'] = prep_df['message'].map(prep.findHashTags)
    prep_df['tweets'] = prep_df['message'].map(prep.removePunctuation)

    return prep_df

interactive = prepareData(raw)

# Feature Engineering
@st.cache(allow_output_mutation=True)
def feat_engine(df):
    feat = df.copy()
    feat['tweets'] = feat['tweets'].map(prep.tweetTokenizer)
    feat['tweets'] = feat['tweets'].map(prep.removeStopWords)
    feat['tweets'] = feat['tweets'].map(prep.lemmatizeTweet)
    return feat
@st.cache(allow_output_mutation=True)
def build_corpus(df):
    corp_df = df.copy()
    vocab = eda.getVocab(corp_df['tweets'])
    word_frequency_dict = eda.wordFrequencyDict(df,'target',vocab)
    class_words = eda.getClassWords(word_frequency_dict)
    pro_spec_words, neutral_spec_words, anti_spec_words, news_spec_words, label_specific_words,class_specific_words, ordered_words = eda.getOrder(class_words,df)
    df = eda.applyScores(df)
    return df, vocab, word_frequency_dict, class_words, pro_spec_words, neutral_spec_words, anti_spec_words, news_spec_words, label_specific_words,class_specific_words, ordered_words

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Creates side bare header image

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    # Reorder the list to change the page order
    options = ["Information",  "Insights", "Prediction"] # These are the four main pages "EDA",
    selection = st.sidebar.selectbox("Choose Page", options)

    ### Building out the "Information" page
    if selection == "Information":
        info = open(r"resources/markdown/info.md").read()
        width = 700

        ### Building "Information" sub pages
        info_options = ["General Information", "Problem Landscape", "Contributors"]
        info_selection = st.selectbox("",info_options)

        if info_selection == "General Information":
            st.image(r"resources/imgs/base_app/info-banner1.jpg", use_column_width = True)
            st.title("Tweet Classifer")
            st.subheader("Climate change belief classification")
            # You can read a markdown file from supporting resources folder
            st.markdown(info[0:2290])
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.write(raw[['sentiment', 'message']])

        if info_selection == "Problem Landscape":
            st.image(r"resources/imgs/base_app/info-banner1.jpg", use_column_width = True)
            ps = open(r"resources/markdown/problem_statement.md").read()
            st.markdown(info[2300:])

        if info_selection == "Contributors":
# CREDITS
            # Team Name
            st.markdown(f"""### **The Blobs** - *classification-jhb-en2*
            """)


            # Team members
# Titus
            st.markdown(
f"""#### <a href="https://github.com/titusndondo">Titus Ndondo</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor3.jpg", width=128)
# Rirhandzu
            st.markdown(
f"""#### <a href="https://github.com/Rirhandzu95">Rirhandzu Mahlaule</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor4.jpg", width=128)
# Kgaogelo
            st.markdown(
f"""#### <a href="https://github.com/mrmamadi">Kgaogelo Mamadi</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/theblobs.png", width=128)
# Stanley
            st.markdown(
f"""#### <a href="https://github.com/Martwuene">Stanley Machuene Kobo</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor6.jpeg", width=128)
# Zanele
            st.markdown(
f"""#### <a href="https://github.com/Zaneleg">Zanele Gwamanda</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor7.jpeg", width=128)
# Bulelani
            st.markdown(
f"""#### <a href="https://github.com/BNkosi">Bulelani Nkosi</a>
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor2.jpg", width=128)

            st.markdown(
f"""#### <a href="https://www.linkedin.com/in/ebrahim-noormahomed-b88404141/">Ebrahim Noormahomed</a> - Supervisor
            """,unsafe_allow_html=True)
            st.image(r"resources/imgs/base_app/contributor1.jpg", width=128)

######################################################################################################
##################################----------------EDA-PAGE--------------##############################
######################################################################################################


    # # Building out the "EDA" page
    # if selection == "EDA":
    #     # Create a new Page
    #     eda_sections = ["Word Frequencies", "Text Analysis", "Sentiment Analysis"]
    #     eda_section = st.selectbox("", eda_sections) # if eda_section == "xxx":
    #     st.write('add stuff here')

    #     # Data preperation (Do not build complex functions, consider only using functions on the page you need them)

    #     # Building the Word Frequencies page
    #     if eda_section == "Word Frequencies":
    #         st.write("fill eda")
    #     # Building the Text Analysis Page
    #     if eda_section == "Text Analysis":
    #         st.write("fill eda")
    #     # Building the Sentiment Analysis Page
    #     if eda_section == "Sentiment Analysis":
    #         st.write("fill eda")
# TASKS:
# 1. Build out the Word  Frequencies Page
# 2. Visualize the top n words per sentiment
# 3. Write some eda about it
# 4. Build Text Analysis page
# 5.
######################################################################################################
##################################-------------EDA-PAGE-END-------------##############################
######################################################################################################

#====================================================================================================#

######################################################################################################
##################################------------INSIGHTS-PAGE-------------##############################
######################################################################################################

    # Building the insights page
    if selection == "Insights":
        # Import data
        ins_data = interactive.copy()
        # Data 
        ins_data = feat_engine(ins_data)
            
        ins_data, vocab, word_frequency_dict, class_words, pro_spec_words, neutral_spec_words, anti_spec_words, news_spec_words, label_specific_words,class_specific_words, ordered_words = build_corpus(ins_data)
        
        insights_pages = ["Instructions", "Overview", "Neutral", "News", "Anti", "Pro"]
        ins_page = st.selectbox("",insights_pages)
        
        # Building out Instructions Page
        if ins_page == "Instructions":
            st.write(f"""
1. `Filter` uncommon words from the dataset. This step will make the less frequent words more relevant
2. `Filter` very common words from the dataset. This will reduce the noice from the larger words
3. `Filter` less positive sentiments by increasing `Lower Positive threshhold`.
4. `Increase or decrease` the Neutral sentiment interval by increasing or decreasing the `Neutral Lower Threshold`
5. `Filter` less negative sentiments by decreasing the `Negative Upper Threshhold`
6. `Handles` and `Hashtags` are static for the whole class
""")
        
        # Building out the Overview page
        if ins_page == "Overview":
            st.write("""Full dataset""")
            # Import Data
            over_data = ins_data.copy()
            # st.write(over_data['tweets'])
            
            # Step 1: Filter infrequent words
            n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=12000)
            top_n_words = eda.topNWords(ordered_words, n=n_1)
            to_include = top_n_words + class_specific_words
            over_data['tweets_clean'] = over_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
            # st.write(over_data['tweets_clean'])

            # Step 2: Filter very frequent words
            n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
            very_common_words = eda.topNWords(ordered_words, n = n_2)
            over_data['tweets_clean'] = over_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
            # st.write(over_data['tweets_clean'])
            st.markdown("### What do people talk about?")
            # Plotting the general wordcloud
            eda.plotWordCloud(data=over_data, label = "Overview\n", column = 'tweets_clean')
            st.pyplot()
            st.markdown("These are the words most commonly used in tweets in the dataset. It is unsurprising that the words present appear to show the sentiments of the `Pro` class. This is because the `Pro class makes up Over 50% of the dataset. It is clear that the majority of the sample demands science based action to be taken  in fighting climate change.")

            # Plotting the general positive sentiments
            n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
            data_pos_gen = over_data[over_data['compound'] > n_3]
            eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            st.markdown("Positive Sentiments is a subset of the data that has a `compound sentiment` score that is greater than `zero`. `Compound sentiment` is a value that ranges from `-1` to `1`, where values that are less than `0` express a negative sentiment or felling towards the subject matter.")
            st.markdown("This word cloud confirms that the people who feel positive about the future once again favour a science based approach to combatting climate change like the `Paris Agreement`. It is noted that they make mention of the Great Barrier Reef, most likely to news reports of scientific findings that could aid in its recovery")

            # Plotting the general  neutral sentiments
            n_4 = st.slider("Neutral Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

            data_neu_gen = over_data[(over_data['compound'] > n_4) & (over_data['compound'] < n_4*-2)]
            eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            st.markdown("Neutral sentiments is a subset representing tweets that no sentiment towards climate change either way. They are the most objective tweets")
            st.markdown("This subset talks about taking action and the factors that affect climate change such as travel. They are facts based and more concerned with politics and scientific evidence linking climate change to human activity.")

            # Plotting the general negative sentiments
            n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
            data_neg_gen = over_data[over_data['compound'] < n_5]
            eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            st.markdown("The negative sentiments are a subset that expresses a strong negative view. It appears that tweets in this category express their outrage at climate change deniers. They consider Climate change to be a disaster that threatens our survival. It is interesting that this group speaks more of assigning blame. This explains refernces to China as they may be blaming China for being the worst emmiter of greenhouse gasses.")
            
            st.markdown("### Who are the key influencers in the discussion on climate change?")
            # Who do they talk about
            eda.plotWordCloud(data=over_data, label = "Handles\n", column = 'handles')
            st.pyplot()
            st.markdown("Here we examin the twitter handles mentioned in the tweets. Of note are the President Donald Trump and Senator Bernie Sanders. These public figures represent extremes in the debate with Trump being a notorious climate change denier while Bernie Sanders has expressed his support of the Green New Deal and is a favorite amongst the Pro climate change camp")

            st.markdown("### What are the most influencial topics?")
            # What do they talk about?
            eda.plotWordCloud(data=over_data, label = "Hashtags\n", column = 'hash_tags')
            st.pyplot()
            st.markdown("Hashtags represent summaries of the main topics in a tweet. They are an excellent way to summarize a conversation. It is immedeately clear that the main topics are political. People speak of voting and taking action. It is clear that the tweets express the wish to change the status quo.")

        # Building out the neutral page
        if ins_page == "Neutral":
            st.write("""Neutral subset""")
            # Import Data
            neu_data = ins_data[ins_data['sentiment']==0].copy()
            # st.write(neu_data['tweets'])
            
            # Step 1: Filter infrequent words
            n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=12000)
            top_n_words = eda.topNWords(ordered_words, n=n_1)
            to_include = top_n_words + class_specific_words
            neu_data['tweets_clean'] = neu_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
            # st.write(neu_data['tweets_clean'])

            # Step 2: Filter very frequent words
            n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
            very_common_words = eda.topNWords(ordered_words, n = n_2)
            neu_data['tweets_clean'] = neu_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
            # st.write(neu_data['tweets_clean'])
            
            # Plotting the general wordcloud
            eda.plotWordCloud(data=neu_data, label = "Overview\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general positive sentiments
            n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
            data_pos_gen = neu_data[neu_data['compound'] > n_3]
            eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general Neutral sentiments
            n_4 = st.slider("neu_data Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

            data_neu_gen = neu_data[(neu_data['compound'] > n_4) & (neu_data['compound'] < n_4*-2)]
            eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
            st.pyplot()



            # Plotting the general negative sentiments
            n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
            data_neg_gen = neu_data[neu_data['compound'] < n_5]
            eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            
            # Who do they talk about
            eda.plotWordCloud(data=neu_data, label = "Handles\n", column = 'handles')
            st.pyplot()

            # What do they talk about?
            eda.plotWordCloud(data=neu_data, label = "Hashtags\n", column = 'hash_tags')
            st.pyplot()

        # Building out the News 
        if ins_page == "News":
            st.write("""News subset""")
            # Import Data
            news_data = ins_data[ins_data['sentiment']==2].copy()
            # st.write(news_data['tweets'])
            
            # Step 1: Filter infrequent words
            n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=10000)
            top_n_words = eda.topNWords(ordered_words, n=n_1)
            to_include = top_n_words + class_specific_words
            news_data['tweets_clean'] = news_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
            # st.write(news_data['tweets_clean'])

            # Step 2: Filter very frequent words
            n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
            very_common_words = eda.topNWords(ordered_words, n = n_2)
            news_data['tweets_clean'] = news_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
            # st.write(news_data['tweets_clean'])
            
            # Plotting the general wordcloud
            eda.plotWordCloud(data=news_data, label = "Overview\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general positive sentiments
            n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
            data_pos_gen = news_data[news_data['compound'] > n_3]
            eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general Neutral sentiments
            n_4 = st.slider("news_data Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

            data_neu_gen = news_data[(news_data['compound'] > n_4) & (news_data['compound'] < n_4*-2)]
            eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
            st.pyplot()



            # Plotting the general negative sentiments
            n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
            data_neg_gen = news_data[news_data['compound'] < n_5]
            eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            
            # Who do they talk about
            eda.plotWordCloud(data=news_data, label = "Handles\n", column = 'handles')
            st.pyplot()

            # What do they talk about?
            eda.plotWordCloud(data=news_data, label = "Hashtags\n", column = 'hash_tags')
            st.pyplot()

        # Building out the pro page
        if ins_page == "Pro":
            st.write("""Pro subset""")
            # Import Data
            pro_data = ins_data[ins_data['sentiment']==1].copy()
            # st.write(pro_data['tweets'])
            
            # Step 1: Filter infrequent words
            n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=10000)
            top_n_words = eda.topNWords(ordered_words, n=n_1)
            to_include = top_n_words + class_specific_words
            pro_data['tweets_clean'] = pro_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
            # st.write(pro_data['tweets_clean'])

            # Step 2: Filter very frequent words
            n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
            very_common_words = eda.topNWords(ordered_words, n = n_2)
            pro_data['tweets_clean'] = pro_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
            # st.write(pro_data['tweets_clean'])
            
            # Plotting the general wordcloud
            eda.plotWordCloud(data=pro_data, label = "Overview\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general positive sentiments
            n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
            data_pos_gen = pro_data[pro_data['compound'] > n_3]
            eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general Neutral sentiments
            n_4 = st.slider("Neutral Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

            data_neu_gen = pro_data[(pro_data['compound'] > n_4) & (pro_data['compound'] < n_4*-2)]
            eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
            st.pyplot()



            # Plotting the general negative sentiments
            n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
            data_neg_gen = pro_data[pro_data['compound'] < n_5]
            eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            
            # Who do they talk about
            eda.plotWordCloud(data=pro_data, label = "Handles\n", column = 'handles')
            st.pyplot()

            # What do they talk about?
            eda.plotWordCloud(data=pro_data, label = "Hashtags\n", column = 'hash_tags')
            st.pyplot()

        # Building out the Anti 
        if ins_page == "Anti":
            st.write("""Neutral subset""")
            # Import Data
            anti_data = ins_data[ins_data['sentiment']==-1].copy()
            # st.write(anti_data['tweets'])
            
            # Step 1: Filter infrequent words
            n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=10000)
            top_n_words = eda.topNWords(ordered_words, n=n_1)
            to_include = top_n_words + class_specific_words
            anti_data['tweets_clean'] = anti_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
            # st.write(anti_data['tweets_clean'])

            # Step 2: Filter very frequent words
            n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
            very_common_words = eda.topNWords(ordered_words, n = n_2)
            anti_data['tweets_clean'] = anti_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
            # st.write(anti_data['tweets_clean'])
            
            # Plotting the general wordcloud
            eda.plotWordCloud(data=anti_data, label = "Overview\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general positive sentiments
            n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
            data_pos_gen = anti_data[anti_data['compound'] > n_3]
            eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
            st.pyplot()

            # Plotting the general Neutral sentiments
            n_4 = st.slider("anti_data Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

            data_neu_gen = anti_data[(anti_data['compound'] > n_4) & (anti_data['compound'] < n_4*-2)]
            eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
            st.pyplot()



            # Plotting the general negative sentiments
            n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
            data_neg_gen = anti_data[anti_data['compound'] < n_5]
            eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
            st.pyplot()
            
            # Who do they talk about
            eda.plotWordCloud(data=anti_data, label = "Handles\n", column = 'handles')
            st.pyplot()

            # What do they talk about?
            eda.plotWordCloud(data=anti_data, label = "Hashtags\n", column = 'hash_tags')
            st.pyplot()
        
            #vocab = eda.getVocab(df = ins_data[ins_data'tweets'])

######################################################################################################
##################################----------INSIGHTS-PAGE-END-----------##############################
######################################################################################################

#====================================================================================================#

######################################################################################################
##################################------------PREDICTION-PAGE-----------##############################
######################################################################################################

    # Building out the "Prediction" page
    if selection == "Prediction":

        ins_data = interactive.copy()

        ins_data = feat_engine(ins_data)

        ins_data, vocab, word_frequency_dict, class_words, pro_spec_words, neutral_spec_words, anti_spec_words, news_spec_words, label_specific_words,class_specific_words, ordered_words = build_corpus(ins_data)

        top_n_words = eda.topNWords(ordered_words, n=5000)

        very_common_words = eda.topNWords(ordered_words, n = 20)


        st.info("Prediction with ML Models")

        # Creating a selection box to choose different models
        models = ['Support Vector', 'Nearest Neighbours',
         'AdaBoost', 'Naive Bayes', 'Decision Tree'] #,'Logistic Regression'
        classifiers = st.selectbox("Choose a classifier", models)

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):

            tweet = prep.removePunctuation(tweet_text)
            tweet = prep.tweetTokenizer(tweet)
            tweet = prep.removeStopWords(tweet)
            tweet = prep.lemmatizeTweet(tweet)
            tweet = prep.removeInfrequentWords(tweet,
            top_n_words = top_n_words + class_specific_words)
            tweet = prep.removeCommonWords(tweet,
            bag=very_common_words)
            tweet = [' '.join(tweet)]

            if classifiers == 'Support Vector':
                predictor = joblib.load(open(os.path.join("resources/support_vector.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            elif classifiers == 'Logistic Regression':
                predictor = joblib.load(open(os.path.join("resources/logistic_regression.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            elif classifiers == 'Nearest Neighbours':
                predictor = joblib.load(open(os.path.join("resources/nearest_neighbors.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            elif classifiers == 'AdaBoost':
                predictor = joblib.load(open(os.path.join("resources/adaboost.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            elif classifiers == 'Naive Bayes':
                predictor = joblib.load(open(os.path.join("resources/naive_bayes.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            elif classifiers == 'Decision Tree':
                predictor = joblib.load(open(os.path.join("resources/decision_tree.pkl"),"rb"))
                prediction = predictor.predict(tweet)

            # elif classifiers == 'Random Forest':
            #     predictor = joblib.load(open(os.path.join("resources/random_forest.pkl"),"rb"))
            #     prediction = predictor.predict(tweet)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            if prediction == -1:
                result = 'Anti - the tweet does not believe in man-made climate change'
            elif prediction == 0:
                result = 'Neutral - the tweet neither supports nor refutes the belief of man-made climate change'
            elif prediction == 1:
                result = 'Pro - the tweet supports the belief of man-made climate change'
            else:
                result = 'News - the tweet links to factual news about climate change'

            st.success("Text Categorized as: {}".format(result))

######################################################################################################
##################################----------PREDICTION-PAGE-END---------##############################
######################################################################################################

    st.sidebar.image(r"resources/imgs/base_app/theblobs.png", width=100)
    st.sidebar.image(r"resources/imgs/EDSA_logo.png", width=225)
# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
