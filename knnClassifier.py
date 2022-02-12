# Correlation heatmap
# Network strength diagram
# https://www.freecodecamp.org/news/basic-data-analysis-on-twitter-with-python-251c2a85062e/
# https://towardsdatascience.com/analysis-on-tweets-using-python-and-twint-c7e6ebce8805
# https://towardsdatascience.com/a-guide-to-mining-and-analysing-tweets-with-r-2f56818fdd16

import numpy as np
import pandas as pd

from datetime import datetime
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pd.options.mode.chained_assignment = None

# Global variables
fileTweets = 'Data Collection\\Tweets.csv'
fileTweetTopics = 'Data Collection\\Tweet Topics.csv'
fileHashtags = 'Data Collection\\Hashtags.csv'
fileHashtagList = 'Data Collection\\Hashtag Lists.csv'
fileMentions = 'Data Collection\\Mentions.csv'
fileMentionList = 'Data Collection\\Mention Lists.csv'
importTweets = 0
baseAnalysis = 0


# Prepare DataFrame for the KNN algorithm
def prepareDataFrame(addMentions = 0, addHashtags = 0, exportHashtags = 0):
    # Import Tweet data
    df = pd.read_csv(fileTweets)
    df = df[df['Month'] != '2022-01-01']
    df = df[['Content', 'Sentiment', 'Subjectivity', 'Mentioned Usernames', 'Hashtags']]

    # Clean the data
    df = df[df['Content'].notna()]
    df['Hashtags'] = df['Hashtags'].fillna('')
    df['Mentioned Usernames'] = df['Mentioned Usernames'].fillna('')

    # Round sentiment and subjectivity
    df['Sentiment'][df['Sentiment'] > 0] = 1            # Positive
    df['Sentiment'][df['Sentiment'] < 0] = -1           # Negative
    df['Subjectivity'] = df['Subjectivity'].round(2)    # Two decimal places

    df.rename(columns = {'Content': 'Contents', 'Mentioned Usernames': 'Mentions'}, inplace = True)
    return df.reset_index(drop = True)


# Prepare DataFrame for the KNN algorithm
def prepareMachineLearningData(addMentions = 0, addHashtags = 0, exportHashtags = 0):
    # # Import Tweet data
    # df = pd.read_csv(fileTweets)
    # df = df[['Content', 'Sentiment', 'Subjectivity', 'Mentioned Usernames', 'Hashtags']]

    # # Clean the data
    # df = df[df['Content'].notna()]
    # df['Hashtags'] = df['Hashtags'].fillna('')
    # df['Mentioned Usernames'] = df['Mentioned Usernames'].fillna('')

    # # Round sentiment and subjectivity
    # df['Sentiment'][df['Sentiment'] > 0] = 1            # Positive
    # df['Sentiment'][df['Sentiment'] < 0] = -1           # Negative
    # df['Subjectivity'] = df['Subjectivity'].round(2)    # Two decimal places

    df = prepareDataFrame()

    # Import categorised hashtags
    dfHashtags = pd.read_excel('hashtag classes.xlsx')
    dfHashtags = dfHashtags[dfHashtags.iloc[:, 2:].sum(axis = 1) > 0]
    dfHashtags = dfHashtags[['Phrase', 'Count', 'Survivor', 'Domestic Violence', 'Politics', 'Entertainment', 'Health']]

    # Apply survivor category before others
    survivorHashtags = dfHashtags[dfHashtags['Survivor'] == 1]['Phrase'].to_list()
    df['Survivor?'] = df['Hashtags'].apply(lambda phrase: sum([1 for word in phrase.split(' ') if word in survivorHashtags]))

    # Apply count weightings to hashtag category columns
    columnHeaders = dfHashtags.columns[3:]
    hashtags = {}
    for header in columnHeaders:
        dfHashtags[header] = dfHashtags['Count'] * dfHashtags[header]
        hashtags[header] = dict(zip(dfHashtags['Phrase'], dfHashtags[header]))
        # Drop NaN values
        hashtags[header] = {key:val for key, val in hashtags[header].items() if val == val}


    for header in columnHeaders:
        df[header + ' Weight'] = df['Hashtags'].apply(lambda phrase: sum([hashtags[header][word] for word in phrase.split(' ') if word in hashtags[header]]))

    df = df[df.iloc[:, 6:-1].sum(axis = 1) > 0]
    df['Largest Category'] = df.iloc[:, 6:].idxmax(axis = 1)

    def selectCategory(row):
        if row['Survivor?'] == 1:
            return 'Victim/Survivor'
        else:
            return row['Largest Category'][:-7]

    df['Category'] = df.apply(lambda row: selectCategory(row), axis = 1)
    df.drop(columns = ['Survivor?', 'Largest Category'], inplace = True)
    for header in columnHeaders:
        df.drop(columns = [header + ' Weight'], inplace = True)

    df.rename(columns = {'Content': 'Contents', 'Mentioned Usernames': 'Mentions'}, inplace = True)
    return df.reset_index(drop = True)


# Perform KNN testing
def knnAnalysisTesting(df_training, n_neighbors = 7):
    # Split data into training and testing sets
    y = df_training.pop('Category')
    X = df_training
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size = 0.2, stratify = y)    #, random_state = 10)
    X_train = X.iloc[X_train]
    X_test = X.iloc[X_test]

    # Prepare the training components
    columns = ['Contents', 'Mentions', 'Hashtags']
    trainingComponents = {}
    for c in columns:
        trainingComponents[c] = {}

        # Builds a dictionary of features and transforms documents to feature vectors and convert our text documents to a matrix of token counts (CountVectorizer)
        trainingComponents[c]['CountVectorizer'] = CountVectorizer()
        trainingComponents[c]['TokenCounts'] = trainingComponents[c]['CountVectorizer'].fit_transform(X_train[c])

        # transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
        trainingComponents[c]['TfidfTransformer'] = TfidfTransformer()
        trainingComponents[c]['NormalisedTfidt'] = trainingComponents[c]['TfidfTransformer'].fit_transform(trainingComponents[c]['TokenCounts'])


    # We fit our Multinomial Naive Bayes classifier on train data to train it.
    knn = KNeighborsClassifier(n_neighbors = 7)

    # training our classifier ; train_data.target will be having numbers assigned for each category in train data
    train_tfidf = hstack([trainingComponents[c]['NormalisedTfidt'] for c in columns], format = 'csr')
    clf = knn.fit(train_tfidf, y_train)

    # Prepare testing data
    testingComponents = {}
    for c in columns:
        testingComponents[c] = {}

        testingComponents[c]['FeatureVector'] = trainingComponents[c]['CountVectorizer'].transform(X_test[c])
        testingComponents[c]['TransformedTfidt'] = trainingComponents[c]['TfidfTransformer'].transform(testingComponents[c]['FeatureVector'])

    test_tfidf = hstack([testingComponents[c]['TransformedTfidt'] for c in columns], format = 'csr')

    # Make predictions
    predicted = clf.predict(test_tfidf)
    output = X_test
    output['Actual Category'] = y_test
    output['Prediction'] = predicted

    # Print the model accuracy
    accuracy = round(np.mean(predicted == y_test) * 100, 2)

    return output, accuracy


# Loop through various KNN tests
def knnAnalysisTestLoop(sampleRange = [1000, 2500, 5000, 10000, 25000, 50000], nNeighbourRange = [3, 5, 7, 9]):
    def checkPrediction(row):
        if row['Actual Category'] == row['Prediction']:
            return 'Correct'
        else:
            return 'Incorrect'
        
    df_originalData = prepareMachineLearningData()
    output= {}
    c = 0
    for samples in sampleRange:
        for n_neighbours in nNeighbourRange:
            for i in range(0, 5):
                output[c] = {
                    'Samples': samples,
                    'Neighbours': n_neighbours
                    }
                
                check = 0
                while check == 0:
                    try:
                        df = df_originalData.copy()
                        if i == 0:
                            # Original
                            df = df.sample(n = samples).reset_index()
                            
                            output[c]['Method'] = 'Original'
                            
                        elif i == 1:
                            # Weightings altered
                            df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
                            df = df.sample(n = samples, weights = df.freq).reset_index()
                            df.drop(columns = ['freq'], inplace = True)
                            
                            output[c]['Method'] = 'Weightings altered'
            
                        elif i == 2:
                            # Categories reduced
                            df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
                            df = df.sample(n = samples).reset_index()
                            
                            output[c]['Method'] = 'Categories reduced'
                        
                        elif i == 3:
                            # Weightings altered then categories reduced
                            df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
                            df = df.sample(n = samples, weights = df.freq).reset_index()
                            df.drop(columns = ['freq'], inplace = True)
                            df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
                            
                            output[c]['Method'] = 'Weightings altered then categories reduced'
                        
                        else:
                            # Categories reduced then weightings altered
                            df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
                            df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
                            df = df.sample(n = samples, weights = df.freq).reset_index()
                            df.drop(columns = ['freq'], inplace = True)
                            
                            output[c]['Method'] = 'Categories reduced then weightings altered'
                    
                        check = 1
                    except:
                        pass
                    
                output[c]['Category Dist.'] = df['Category'].value_counts(normalize = True)
                
                # Clean the prediction DataFrame
                df_predictions, accuracy = knnAnalysisTesting(df, n_neighbours)
                df_predictions['Accurate Prediction'] = df_predictions.apply(lambda row: checkPrediction(row), axis = 1)
                df_predictions = df_predictions.groupby(['Actual Category','Accurate Prediction']).size().reset_index()
                df_predictions = df_predictions.pivot(index = 'Actual Category', columns = 'Accurate Prediction', values = 0).reset_index()
                df_predictions['Correct'] = df_predictions['Correct'].fillna(0)
                df_predictions['Incorrect'] = df_predictions['Incorrect'].fillna(0)
                df_predictions['Accuracy'] = df_predictions['Correct'] / (df_predictions['Correct'] + df_predictions['Incorrect'])
                df_predictions.index.name = 'Index'        
        
                output[c]['Pred. Accuracy'] = df_predictions
                c += 1
                print('{} / {} complete. Samples: {}, Neighbours: {}, Method: {}.'.format(c, (len(sampleRange) * len(nNeighbourRange) * 5), samples, n_neighbours, i))

    # Prepare DataFrames to output to CSV
    df_keys = pd.DataFrame(columns = ['Samples', 'Neighbours', 'Method'])
    df_categoryDist = pd.DataFrame(columns = ['Category', 'Distribution', 'Key'])
    df_precAccuracy = pd.DataFrame(columns = ['Category', 'Correct', 'Incorrect', 'Accuracy', 'Key'])
    for key, value in output.items():
        i = len(df_keys)
        df_keys.loc[i] = [value['Samples'], value['Neighbours'], value['Method']]
        
        df_temp_categoryDist = pd.DataFrame(value['Category Dist.']).reset_index()
        df_temp_categoryDist['Key'] = i
        df_temp_categoryDist.rename(columns = {'index': 'Category', 'Category': 'Distribution'}, inplace = True)
        df_categoryDist = df_categoryDist.append(df_temp_categoryDist, ignore_index = True)
                    
        df_temp_precAccuracy = value['Pred. Accuracy'].reset_index(drop = True).rename(columns = {'Accurate Preduction': 'Index', 'Actual Category': 'Category'})
        df_temp_precAccuracy['Key'] = i
        df_precAccuracy = df_precAccuracy.append(df_temp_precAccuracy, ignore_index = True)
    
    # Output results to CSV
    filePath = 'Data Collection\\Summary data\\Data model tables\\Knn testing\\'
    df_keys.to_csv(filePath + 'keys.csv')
    df_categoryDist.to_csv(filePath + 'category distribution.csv', index = False)
    df_precAccuracy.to_csv(filePath + 'prediction accuracy.csv', index = False)


# Generate the KNN machine learning model
def generateKnnModel():
    # Prepare training data
    sampleSize = 10000
    nNeighbours = 7
    
    df = prepareMachineLearningData().reset_index(drop = True)
    df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
    df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
    df = df.sample(n = sampleSize, weights = df.freq).reset_index()
    df.drop(columns = ['freq'], inplace = True)
    
    # Split data into training and testing sets
    y = df.pop('Category')
    X = df
    
    # Prepare the training components
    columns = ['Contents', 'Mentions', 'Hashtags']
    trainingComponents = {}
    for c in columns:
        trainingComponents[c] = {}
    
        # Builds a dictionary of features and transforms documents to feature vectors and convert our text documents to a matrix of token counts (CountVectorizer)
        trainingComponents[c]['CountVectorizer'] = CountVectorizer()
        trainingComponents[c]['TokenCounts'] = trainingComponents[c]['CountVectorizer'].fit_transform(X[c])
    
        # transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
        trainingComponents[c]['TfidfTransformer'] = TfidfTransformer()
        trainingComponents[c]['NormalisedTfidt'] = trainingComponents[c]['TfidfTransformer'].fit_transform(trainingComponents[c]['TokenCounts'])
    
    
    # We fit our Multinomial Naive Bayes classifier on train data to train it.
    knn = KNeighborsClassifier(nNeighbours = 7)
    
    # training our classifier ; train_data.target will be having numbers assigned for each category in train data
    train_tfidf = hstack([trainingComponents[c]['NormalisedTfidt'] for c in columns], format = 'csr')
    clf = knn.fit(train_tfidf, y)

    return clf

if __name__ == "__main__":
    pass
    # knnAnalysisTestLoop()