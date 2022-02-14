# DFV Twitter Analysis
This repository contains the code used by Kadin Schultz during his thesis at UNSW.

### Required Folder Structure
```
.
├── basicAnalytics.py
├── collectTwitterData.py
├── knnClassifier.py
├── main.py
└── Data Collection
    ├── Data to process
    ├── Method testing
    ├── Processed
    |   ├── Processed Hashtag Lists
    |   ├── Processed Hashtags
    |   ├── Processed Mention Lists
    |   ├── Processed Mentions
    |   ├── Processed Tweets
    |   └── Raw data
    └── Summary data
        ├── Data model tables
        |   └── Knn testing
        └── n-grams
```

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
knnAnalysisTestLoop(sampleRange = [1000, 2500, 5000, 10000, 25000, 50000],
                    nNeighbourRange = [3, 5, 7, 9])
```

```accuracyMeasurementLoop()``` is a more indepth train and test function than knnAnalysisTestLoop(), reporting on the True/False-Positive/Negative totals, albeit taking much longer to process.

```
accuracyMeasurementLoop(sampleSize = 10000,
                            n_neighbours = 7)
```

```runKnnModelForJanuary()``` executes the K-Nearest Neighbours Classifiers on Tweets from January-22.

```
runKnnModelForJanuary(sampleSize = 5000,
                      n_neighbours = 9,
                      method = 0)
```
