from collectTwitterData import scrapeTweets, processTweetFiles
from basicAnalytics import processBaseAnalysis
from knnClassifier import knnAnalysisTestLoop, prepareDataFrame, generateKnnModel
from datetime import datetime

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
