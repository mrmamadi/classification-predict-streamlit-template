from . import base_app
from . import preprocessing as p

def test_preprocessing():
    """
    make sure preprocessing fuctions work correctly
    """
    bag = [
        ('change', 12634),
        ('climate', 12609),
        ('rt', 9720),
        ('urlweb', 9656),
        ('global', 3773)
        ]
    
    very_common_words = ['change', 'climate', 'rt', 'urlweb', 'global']
    
    assert p.findURLs("you can view this at https://github.com/mrmamadi/classification-predict-streamlit-template") == "https://github.com/mrmamadi/classification-predict-streamlit-template", 'incorrect'
    # assert p.strip_url - Create a test
    assert p.findHandles("hi @SenBernieSanders, you will beat @realDonaldTrump") == ['@SenBernieSanders','@realDonaldTrump']
    assert p.findHashTags("Oil is killing the world renewables and EVS are the way the go! #EVs #GlobalWarming #Fossilfuels") == ['#EVs', '#GlobalWarming', '#Fossilfuels']
    assert p.removePunctuation("Hey! Check out this story: urlweb. He doesn't seem impressed. :)") == "Hey Check out this story urlweb He doesn't seem impressed"
    assert p.tweetTokenizer("Read @swrightwestoz latest on climate change insurance amp lending featuring APRA speech and @CentrePolicyDev work urlweb") == ['read', 'latest', 'on', 'climate', 'change', 'insurance', 'amp', 'lending', 'featuring', 'apra', 'speech', 'and', 'work', 'urlweb']
    assert p.removeStopWords(['read', 'latest', 'on', 'climate', 'change', 'insurance', 'amp', 'lending', 'featuring', 'apra', 'speech', 'and', 'work', 'urlweb']) == ['read', 'latest', 'on', 'climate', 'change', 'insurance', 'amp', 'lending', 'featuring', 'apra', 'speech', 'and', 'work', 'urlweb']
    assert p.lemmatizeTweet(['read', 'latest', 'on', 'climate', 'change', 'insurance', 'amp', 'lending', 'featuring', 'apra', 'speech', 'and', 'work', 'urlweb']) == ['read', 'latest', 'climate', 'change', 'insurance', 'lending', 'featuring', 'apra', 'speech', 'work', 'urlweb']
    assert p.topNWords(bag, n = 1) == ['change']
    assert p.removeInfrequentWords(bag,2) == ['change', 'climate']
    assert p.removeCommonWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb']) == ['equation']
    assert p.lengthOfTweet(['This', 'is', 'a' , 'tweet']) == 4
    assert p.getPolarityScores(['polyscimajor', 'chief', 'carbon', 'dioxide', 'main', 'cause', 'wait']) == {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}