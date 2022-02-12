# DFV-Twitter-Analysis
This repository contains the code used by Kadin Schultz during his thesis at UNSW.

### main.py
This file executes the key functions from the three others.

### collectTwitterData.py
This file contains all of the functions required to download Tweets from Twitter.

```scrapeTweets()``` will save all Tweets within the startDate and endDate, containing the specified keyword. The Tweets will be saved within monthly .CSV files. ```processTweetFiles()``` will process/clean all of the monthly Tweet .CSV files and compile them into a singular file.
```
scrapeTweets(keyword = 'domestic violence',
             startDate = datetime(2022, 1, 1),
             endDate = datetime(2022, 1, 30))
```

### basicAnalytics.py
```processBaseAnalysis()``` will process the .CSV file created by ```processTweetFiles()```, and generate an assortment of .CSV files for analysis in other tools such as Power BI or Tableau.

### knnClassifier.py
```knnAnalysisTestLoop()``` will train and test K-Nearest Neighbours Classifiers with various sample sizes and K-sizes, so that analysts can decide which parameters they want to use in their model.

```
knnAnalysisTestLoop(sampleRange = [1000, 2500, 5000, 10000, 25000, 50000], nNeighbourRange = [3, 5, 7, 9])
```
