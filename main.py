# Custom functions
from collectTwitterData import scrapeTweets, processTweetFiles
from basicAnalytics import processBaseAnalysis
from knnClassifier import knnAnalysisTestLoop, prepareDataFrame, generateKnnModel

from datetime import datetime
from sys import exit

exit()

# Collect data from Twitter
scrapeTweets(keyword = 'domestic violence',
             startDate = datetime(2022, 1, 1),
             endDate = datetime(2022, 1, 30))

# Process the Tweets
processTweetFiles()

# Process the basic analytics
processBaseAnalysis()

# Loop through various KNN configurations to determine the best
knnAnalysisTestLoop(sampleRange = [1000, 2500, 5000, 10000, 25000, 50000],
                    nNeighbourRange = list(range(1, 16)))


    
clf = generateKnnModel()
df = prepareDataFrame()

# Make predictions
predicted = clf.predict(test_tfidf)
output = X_test
output['Actual Category'] = y_test
output['Prediction'] = predicted

# Print the model accuracy
accuracy = round(np.mean(predicted == y_test) * 100, 2)






# TODO
# Because victim/survivor is the most important, report on true/false positives and true/false negatives
# Need to minimise that figure




# So we have the tweets being scraped
# TODO
#   TopN phrases for DFV to find other keywords (per year? trends over time?), repeat above for them
#       Merge files for other keywords into DFV file, remove duplicates based off ID



    # Investigating the user mentions that created the aforementioned surges
    # usernameList = ['realDonaldTrump', 'MichaelAvenatti', 'Change']

    # Lots of tweets April-2020


    # Check for non-english tweets
    # Mentions per month
    # Hashtags per month
    # Sentiment analysis per unigram, bigram, and trigram
    # Sentiment analysis per hashtag
    # Sentiment analysis per mention
    # Sentiment analysis histograms
    # Sentiment analysis scatter plot (sentiment Y, number X) (sentiment Y, subjectivity X)

    # Sentiment analysis per topic
    # Topic count per month