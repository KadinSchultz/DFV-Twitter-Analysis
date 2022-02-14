# Author's Note
#   I recommend not running all of these functions in sequence, and they can take a very long time.
#   Instead, run them all individually, or as required.

from collectTwitterData import scrapeTweets, processTweetFiles
from basicAnalytics import processBaseAnalysis
from knnClassifier import knnAnalysisTestLoop, accuracyMeasurementLoop, runKnnModelForJanuary

from datetime import datetime
from pathlib import Path
from sys import exit

# Create the folders required for this script
def createFolders():
    Path('Data Collection\\Data to process').mkdir(parents = True, exist_ok = True)
    Path('Data Collection\\Method testing').mkdir(parents = True, exist_ok = True)
    
    for f in ['Tweets', 'Hashtags', 'Hashtag Lists', 'Mentions', 'Mention Lists']:
        Path('Data Collection\\Processed\\Processed {}'.format(f)).mkdir(parents = True, exist_ok = True)
        
    Path('Data Collection\\Processed\\Raw data').mkdir(parents = True, exist_ok = True)
    
    Path('Data Collection\\Summary data\\Data model tables\\Knn testing').mkdir(parents = True, exist_ok = True)
    Path('Data Collection\\Summary data\\n-grams').mkdir(parents = True, exist_ok = True)

# To prevent all functions from being run accidentally
exit()

# Create the folders required for this script
createFolders()

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
                    nNeighbourRange = list(range(3, 11, 2)))

# Loop through various KNN training data alteration methods, to determine
# the best configurationthrough True/False-Positive/Negative analysis
accuracyMeasurementLoop(sampleSize = 10000,
                        n_neighbours = 7)

# Execute the KNN algorithm for January's tweets
runKnnModelForJanuary()