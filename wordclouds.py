import streamlit as st
import eda
import base_app as ba


def wordcloudpages(df, page_selection):
    """

    Interactive wordcloud page
    
    Author: JHB-EN2.

    Description: This function is used to define the interactive wordcloud page

    Parameters
    ----------
        df (DataFrame): DataFrame containing climate change tweets and sentiments
    Returns
    -------
        None
    Examples
    --------
        >>> if selection == "Wordclouds":
        >>>     wordclouds(interactive)

        `anaconda prompt`
        --> streamlit run base_app.py
            You can now view your Streamlit app in your browser.
            Local URL: http://localhost:0000
            Network URL: http://000.000.000.000:000


    """
    st.write(f"""
    1. `Filter` your wordcloud's vocabulary by `word frequency`. This step will make the smaller words more relevant
    2. `Filre

    """)
    df = ba.build_corpus(df)[0]
    class_specific_words= ba.build_corpus(df)[9]
    ordered_words = ba.build_corpus(df)[10]
    # Building out the Overview page
    if page_selection == "Overview":
        st.write("""Full dataset""")
        # Import Data
        over_data = df.copy()
        # st.write(over_data['tweets'])
        
        # Step 1: Filter infrequent words
        n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=10000)
        top_n_words = eda.topNWords(ordered_words, n=n_1)
        to_include = top_n_words + class_specific_words
        over_data['tweets_clean'] = over_data['tweets'].map(lambda tweet: eda.removeInfrequentWords(tweet, include = to_include))
        # st.write(over_data['tweets_clean'])

        # Step 2: Filter very frequent words
        n_2 = st.sidebar.slider('Step 2: Filter very common words', min_value = 0, max_value=40, step =2, value = 20)
        very_common_words = eda.topNWords(ordered_words, n = n_2)
        over_data['tweets_clean'] = over_data['tweets'].map(lambda tweet: eda.removeCommonWords(tweet, very_common_words))
        # st.write(over_data['tweets_clean'])
        
        # Plotting the general wordcloud
        eda.plotWordCloud(data=over_data, label = "Overview\n", column = 'tweets_clean')
        st.pyplot()

        # Plotting the general positive sentiments
        n_3 = st.slider("Positive Threshhold", min_value = 0.0, max_value=0.95, step =0.05, value = 0.75)
        data_pos_gen = over_data[over_data['compound'] > n_3]
        eda.plotWordCloud(data=data_pos_gen, label = "Positive Sentiments\n", column = 'tweets_clean')
        st.pyplot()

        # Plotting the general  neutral sentiments
        n_4 = st.slider("Neutral Lower Threshhold", min_value = -1.0, max_value=-0.05, step =0.05, value = -0.15)

        data_neu_gen = over_data[(over_data['compound'] > n_4) & (over_data['compound'] < n_4*-2)]
        eda.plotWordCloud(data=data_neu_gen, label = "Neutral Sentiments\n", column = 'tweets_clean')
        st.pyplot()



        # Plotting the general negative sentiments
        n_5 = st.slider("Negative Upper Threshold", min_value = -0.95, max_value = 0.0, step = 0.05, value = -0.60)
        data_neg_gen = over_data[over_data['compound'] < n_5]
        eda.plotWordCloud(data=data_neg_gen, label = "Negative Sentiments\n", column = 'tweets_clean')
        st.pyplot()
        
        # Who do they talk about
        eda.plotWordCloud(data=over_data, label = "Handles\n", column = 'handles')
        st.pyplot()

        # What do they talk about?
        eda.plotWordCloud(data=over_data, label = "Hashtags\n", column = 'hash_tags')
        st.pyplot()

    # Building out the neutral page
    if page_selection == "Neutral":
        st.write("""Neutral subset""")
        # Import Data
        neu_data = df[df['sentiment']==0].copy()
        # st.write(neu_data['tweets'])
        
        # Step 1: Filter infrequent words
        n_1 = st.sidebar.slider('Step 1: Filter infrequent words', min_value = 0, max_value = 13000, step = 1000, value=10000)
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
    if page_selection == "News":
        st.write("""News subset""")
        # Import Data
        news_data = df[df['sentiment']==2].copy()
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
    if page_selection == "Pro":
        st.write("""Pro subset""")
        # Import Data
        pro_data = df[df['sentiment']==1].copy()
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
    if page_selection == "Anti":
        st.write("""Neutral subset""")
        # Import Data
        anti_data = df[df['sentiment']==-1].copy()
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

        st.write("""# BLANK""")