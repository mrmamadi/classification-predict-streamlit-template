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
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

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
	prep_df = prep.typeConvert(prep_df)
	prep_df['urls'] = prep_df['message'].map(prep.findURLs)
	prep_df = prep.strip_url(prep_df)
	prep_df['handles'] = prep_df['message'].map(prep.findHandles)
	prep_df['hash_tags'] = prep_df['message'].map(prep.findHashTags)
	prep_df['tweets'] = prep_df['message'].map(prep.removePunctuation)
	target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
	prep_df['target'] = prep_df['sentiment'].map(target_map)
	return prep_df

interactive = prepareData(raw)

### Feature Extraction
@st.cache(allow_output_mutation=True)
def featureCreation(df, uncommon = 10000, common = 20):
	"""
	Create features from input dataframe
	Parameters
	----------
		df (DataFrame): input DataFrame
		uncommon (int): keep top n words Default = 10000
		common (int): remove n most common words Default = 20
	Returns
	-------
		feat_df (DataFrame): output DataFrame
	"""
	feat_df = df.copy()
	# Prepare data for feature manipulation
	feat_df['tweets'] = feat_df['tweets'].map(prep.tweetTokenizer)
	feat_df['tweets'] = feat_df['tweets'].map(prep.removeStopWords)
	feat_df['tweets'] = feat_df['tweets'].map(prep.lemmatizeTweet)

	# Create a vocabulary
	vocab = list()
	for tweet in feat_df['tweets']:
		for token in tweet:
			vocab.append(token)

	# Create a frequency list
	ordered_words = Counter(vocab).most_common()
	# creat a list of most common words
	top_n_words = prep.topNWords(ordered_words, n = uncommon)

	# Remove infrequent words
	feat_df['tweets_clean'] = feat_df['tweets'].map(lambda tweet: prep.removeInfrequentWords(tweet, top_n_words))

	# Create list of very common words
	very_common_words = prep.topNWords(ordered_words, n = common)

	# Remove common words
	feat_df['tweets_clean'] = feat_df['tweets_clean'].map(lambda tweet: prep.removeCommonWords(tweet, very_common_words))

	# Count the tweet length
	feat_df['len_of_tweet'] = feat_df['tweets_clean'].map(prep.lengthOfTweet)
	
	# Add target labels
	target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
	feat_df['target'] = feat_df['sentiment'].map(target_map)

	# Calculate tweet sentiment
	nltk_scores = dict(compound = list(), negative = list(), neutral = list(), positive = list())
	for tweet in feat_df['message']:
		output = prep.getPolarityScores(tweet)
		nltk_scores['compound'].append(output['compound'])
		nltk_scores['negative'].append(output['neg'])
		nltk_scores['neutral'].append(output['neu'])
		nltk_scores['positive'].append(output['pos'])

	if 'compound' in feat_df.columns:
		feat_df.drop(['compound', 'negative', 'neutral', 'positive'], axis = 1, inplace = True)
		feat_df = pd.concat([feat_df, pd.DataFrame(nltk_scores)], axis = 1)
	else:
		feat_df = pd.concat([feat_df, pd.DataFrame(nltk_scores)], axis = 1)

	sentiment_scores = [TextBlob(' '.join(tweet)).sentiment for tweet in feat_df['message']]

	pol = list()
	subj = list()
	for scores in sentiment_scores:
		pol.append(scores.polarity)
		subj.append(scores.subjectivity)

	feat_df['polarity'] = pol
	feat_df['subjectivity'] = subj

	return feat_df

# eda = featureCreation(interactive)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates side bare header image
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	# Reorder the list to change the page order
	options = ["Information", "EDA", "Insights", "Prediction"] # These are the four main pages
	selection = st.sidebar.selectbox("Choose Page", options)

	### Building out the "Information" page
	if selection == "Information":
		info = open(r"resources\markdown\info.md").read()
		width = 700

		### Building "Information" sub pages
		info_options = ["General Information", "Problem Landscape", "Contributors"]
		info_selection = st.selectbox("",info_options)
			
		if info_selection == "General Information":
			st.image(r"resources\imgs\base_app\info-banner1.jpg", use_column_width = True)
			st.title("Tweet Classifer")
			st.subheader("Climate change belief classification")
			# You can read a markdown file from supporting resources folder
			st.markdown(info[0:2290])
			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'):
				st.write(raw[['sentiment', 'message']])

		if info_selection == "Problem Landscape":
			st.image(r"resources\imgs\base_app\info-banner1.jpg", use_column_width = True)
			ps = open(r"resources\markdown\problem_statement.md").read()
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
			st.image(r"resources\imgs\base_app\contributor3.jpg", width=128)
# Rirhandzu
			st.markdown(
f"""#### <a href="https://github.com/Rirhandzu95">Rirhandzu Mahlaule</a>
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\contributor4.jpg", width=128)
# Kgaogelo
			st.markdown(
f"""#### <a href="https://github.com/mrmamadi">Kgaogelo Mamadi</a>
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\theblobs.png", width=128)
# Stanley
			st.markdown(
f"""#### <a href="https://github.com/Martwuene">Stanley Machuene Kobo</a>
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\contributor6.jpeg", width=128)
# Zanele
			st.markdown(
f"""#### <a href="https://github.com/Zaneleg">Zanele Gwamanda</a>
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\contributor7.jpeg", width=128)
# Bulelani
			st.markdown(
f"""#### <a href="https://github.com/BNkosi">Bulelani Nkosi</a>
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\contributor2.jpg", width=128)
			
			st.markdown(
f"""#### <a href="https://www.linkedin.com/in/ebrahim-noormahomed-b88404141/">Ebrahim Noormahomed</a> - Supervisor
			""",unsafe_allow_html=True)
			st.image(r"resources\imgs\base_app\contributor1.jpg", width=128)

######################################################################################################
##################################------------PREDICTION-PAGE-----------##############################
######################################################################################################

	### DEADLINE: 27/06/2020 - Saturday
	### Delete an issue after committing please

	### ISSUES use: git commit -m "Description. Fixes issue x" : Where "x" is the issue number
	### 6. Add VECTORIZERS.PKL to resources\vectorizers folder
	### 7. Create a  selectbox to choose from vectorizers
	### 8. write an "if and else" function in order to make a prediction with the user selections
	### 9. Add vectorizers.md to the resources\markdown folder briefly explaining what a vectorizer does
	###    and the difference beterrn the two
	### 10. Add all model.pkl files to the resources\vectorizers folder
	### 11. Update selectbox with new nodels
	### 12. Write model.md files to explain each model briefly and perhaps mention the models f1-score

	##########################################################################################
	############################------------MELVA-MRMAMADI------------########################
	##########################################################################################

	# Building out the "Prediction" page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		
		# Creating a selection box to choose different models
		models = ['Support Vector Classifier','Logistic Regression']
		classifiers = st.selectbox("Choose a classifier", models)
		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):

			if classifiers == 'Support Vector Classifier':
				# Transforming user input with vectorizer
				vect_text = [tweet_text]#.toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/linear_svc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				
			elif classifiers == 'Logistic Regression':
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1:
				result = 'Anti'
			elif prediction == 0:
				result = 'Neutral'
			elif prediction == 1:
				result = 'Pro'
			else:
				result = 'News'
			
			st.success("Text Categorized as: {}".format(result))

	##########################################################################################
	############################----------MELVA-MRMAMADI-END----------########################
	##########################################################################################
	### Zanele and Bulelani review and finalize
	### Delete instruction comments when done
######################################################################################################
##################################----------PREDICTION-PAGE-END---------##############################
######################################################################################################

#====================================================================================================#

######################################################################################################
##################################----------------EDA-PAGE--------------##############################
######################################################################################################


	# Building out the "EDA" page
	if selection == "EDA":
		# Create a new Page
		eda_sections = ["Word Frequencies", "Text Analysis", "Sentiment Analysis"]
		eda_section = st.selectbox("", eda_sections) # if eda_section == "xxx":
		st.write('add stuff here')
		
		# Data preperation (Do not build complex functions, consider only using functions on the page you need them)

		# Building the Word Frequencies page
		if eda_section == "Word Frequencies":
			st.write("fill eda")
		# Building the Text Analysis Page
		if eda_section == "Text Analysis":
			st.write("fill eda")
		# Building the Sentiment Analysis Page
		if eda_section == "Sentiment Analysis":
			st.write("fill eda")
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

	##########################################################################################
	############################-----------BULELANI-ZANELE------------########################
	##########################################################################################

	# Building the insights page
	if selection == "Insights":
		# Import data
		ins_data = interactive.copy()
		# Feature Engineering
		@st.cache
		def feat_engine(df):
			feat = df.copy()
			feat['tweets'] = feat['tweets'].map(prep.tweetTokenizer)
			feat['tweets'] = feat['tweets'].map(prep.removeStopWords)
			feat['tweets'] = feat['tweets'].map(prep.lemmatizeTweet)
			return feat
		ins_data = feat_engine(ins_data)
		@st.cache
		def build_corpus(df):
			vocab = eda.getVocab(df = ins_data['tweets'])
			word_frequency_dict = eda.wordFrequencyDict(df,'target',vocab)
			class_words = eda.getClassWords(word_frequency_dict)
			ordered_words = eda.getOrder(class_words,df)
			most_common = eda.topNWords(ordered_words, n=5000)
			return vocab, word_frequency_dict, class_words, ordered_words, most_common
			
		vocab, word_frequency_dict, class_words, ordered_words, most_common = build_corpus(ins_data)

		insights_pages = ["Instructions", "Overview", "Neutral", "Pro", "Anti"]
		ins_page = st.selectbox("",insights_pages)

		# Building out Instructions Page
		if ins_page == "Instructions":
			st.write(f""" 1. `Filter` your wordcloud's vocabulary by `word frequency`
""")
		
		# Building out the Overview page
		if ins_page == "Overview":
			st.write("""# BLANK""")

		if ins_page == "Neutral":
			st.write("""# BLANK""")
			#vocab = eda.getVocab(df = ins_data[ins_data'tweets'])
	##########################################################################################
	############################---------BULELANI-ZANELE-END----------########################
	##########################################################################################
	## What was done to prep_df
	# 	prep_df = df.copy()
	# prep_df = prep.typeConvert(prep_df)
	# prep_df['urls'] = prep_df['message'].map(prep.findURLs)
	# prep_df = prep.strip_url(prep_df)
	# prep_df['handles'] = prep_df['message'].map(prep.findHandles)
	# prep_df['hash_tags'] = prep_df['message'].map(prep.findHashTags)
	# prep_df['tweets'] = prep_df['message'].map(prep.removePunctuation)
	# return prep_df
	### Zanele and Bulelani review and finalize
	### Delete instruction comments when done
######################################################################################################
##################################----------INSIGHTS-PAGE-END-----------##############################
######################################################################################################
	
	st.sidebar.image(r"resources\imgs\base_app\theblobs.png", width=100)
	st.sidebar.image(r"resources\imgs\EDSA_logo.png", width=225)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
