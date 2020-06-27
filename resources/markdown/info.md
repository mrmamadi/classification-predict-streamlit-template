###  How it works

This process requires the user to input text
(ideally a tweet relating to climate change), and will
classify it according to whether or not they believe in
climate change.Below you will find information about the data source
and a brief data description. You can have a look at word clouds and
other general EDA on the EDA page, and make your predictions on the
prediction page that you can navigate to in the sidebar.

###  Data description as per source

The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

### Variable definitions

__sentiment:__ Sentiment of tweet
- 2(News): the tweet links to factual news about climate change
- 1(Pro): the tweet supports the belief of man-made climate change
- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change
- -1(Anti): the tweet does not believe in man-made climate change

__message:__ Tweet body

__tweetid:__  Unique tweet id

## Objective
- __Competition:__ [Climate Change Belief Analysis](https://www.kaggle.com/c/climate-change-belief-analysis)
- __Determine:__ How the public perceives climate change and wheter or not they believe it is a threat
- __Classify:__ Individuals by belief in climate change based on novel tweet data
- __Evaluation:__ Weighted F1-Score

## Evaluation Metric

__weighted f1-score__

The traditional f1-score is the harmonic mean between precision and recall:
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/3607c634303f2fd8b69ca4f9d97a491c45083cc5"
     />

The F1 Scores are calculated for each label (`2`, `1`, `0`, `-1`) and then their average is weighted by support - being the number of true instances for each label. In other words, the f1-score for each label is weighted based on it's proportion of TN and TP in the  sample.

Just like the f1-score, the weighted f1 score will be a number between 0 and 1 where perfect precision and recall occurs at 1.

##########

### Context
Many companies are built around lessening their environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.

### Problem statement
With this context, EDSA is challenging you during the Classification Sprint with the task of creating a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.

Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.
 
To meet the problem statement we must:
- **Train** a *Classification model* to predict the sentiment of Tweets related to climate change.
- **Build** an app using *Streamlit*.
- **Host** the app on an *AWS EC2* instance.
- **Present** solution via *Video conference* presentation.
