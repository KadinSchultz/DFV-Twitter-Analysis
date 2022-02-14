import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pd.options.mode.chained_assignment = None

# Global variables
fileTweets = 'Data Collection\\Tweets.csv'


# Prepare DataFrame for the KNN algorithm
def prepareDataFrame(trainingData = True):
    # Import Tweet data
    df = pd.read_csv(fileTweets)
    if trainingData:
        df = df[df['Month'] != '2022-01-01']
        df = df[['Content', 'Sentiment', 'Subjectivity', 'Mentioned Usernames', 'Hashtags']]
    else:
        df = df[df['Month'] == '2022-01-01']
        df = df[['ID', 'Username', 'Content', 'Sentiment', 'Subjectivity', 'Mentioned Usernames', 'Hashtags']]

    # Clean the data
    df = df[df['Content'].notna()]
    df['Hashtags'] = df['Hashtags'].fillna('')
    df['Mentioned Usernames'] = df['Mentioned Usernames'].fillna('')

    # Round sentiment and subjectivity
    df['Sentiment'] = df['Sentiment'].round(2)
    df['Subjectivity'] = df['Subjectivity'].round(2)

    df.rename(columns = {'Content': 'Contents', 'Mentioned Usernames': 'Mentions'}, inplace = True)
    return df.reset_index(drop = True)


# Prepare DataFrame for the KNN algorithm
def prepareMachineLearningData(exportData = False):
    # # Import Tweet data
    df = prepareDataFrame()

    # Import categorised hashtags
    dfHashtags = pd.read_excel('hashtag classes.xlsx')
    dfHashtags = dfHashtags[dfHashtags.iloc[:, 2:].sum(axis = 1) > 0]
    dfHashtags = dfHashtags[['Phrase', 'Count', 'Survivor', 'Domestic Violence', 'Politics', 'Health']] #, 'Entertainment', 'Sports', 'Relationships', 'Kids']]

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
    df = df.reset_index(drop = True)
    
    if exportData:
        df.to_csv('Data Collection\\Classified Tweets.csv', index = False)
    
    return df


# Modify and sample the training data
def applyTrainingMethod(df, sampleSize, method):
    if method == 0:
        # Original
        df = df.sample(n = sampleSize).reset_index()
                    
    elif method == 1:
        # Weightings altered
        df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
        df['pend'] = df.groupby('Category')['Category'].transform('count')
        df = df.sample(n = sampleSize, weights = df.freq).reset_index()
        df.drop(columns = ['freq'], inplace = True)
        
    elif method == 2:
        # Categories reduced
        df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
        df = df.sample(n = sampleSize).reset_index()
        
    elif method == 3:
        # Weightings altered then categories reduced
        df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
        df = df.sample(n = sampleSize, weights = df.freq).reset_index()
        df.drop(columns = ['freq'], inplace = True)
        df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
    
    else:
        # Categories reduced then weightings altered
        df.loc[df['Category'] != 'Victim/Survivor', 'Category'] = 'Other'
        df['freq'] = len(df) - df.groupby('Category')['Category'].transform('count')
        df = df.sample(n = sampleSize, weights = df.freq).reset_index()
        df.drop(columns = ['freq'], inplace = True)
        
    return df


# Prepare the modelling components for the KNN algorithm
def prepareModellingComponents(X, y, n_neighbors):
    columns = ['Contents', 'Mentions', 'Hashtags']
    trainingComponents = {}
    for c in columns:
        trainingComponents[c] = {}
    
        # Builds a dictionary of features and transforms documents to feature vectors and convert our text documents to a matrix of token counts (CountVectorizer)
        trainingComponents[c]['CountVectorizer'] = CountVectorizer()
        trainingComponents[c]['TokenCounts'] = trainingComponents[c]['CountVectorizer'].fit_transform(X[c])
    
        # Transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
        trainingComponents[c]['TfidfTransformer'] = TfidfTransformer()
        trainingComponents[c]['NormalisedTfidt'] = trainingComponents[c]['TfidfTransformer'].fit_transform(trainingComponents[c]['TokenCounts'])

    # We fit our Multinomial Naive Bayes classifier on train data to train it.
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    
    # training our classifier ; train_data.target will be having numbers assigned for each category in train data
    train_tfidf = hstack([trainingComponents[c]['NormalisedTfidt'] for c in columns], format = 'csr')
    train_tfidf = hstack((train_tfidf,
                          np.array(X['Sentiment'])[:, None],
                          np.array(X['Subjectivity'])[:, None]))
    
    clf = knn.fit(train_tfidf, y)

    return trainingComponents, clf


# Prepare the testing components for the KNN algorithm
def prepareTestingComponents(X, trainingComponents):
    columns = ['Contents', 'Mentions', 'Hashtags']
    testingComponents = {}
    for c in columns:
        testingComponents[c] = {}

        testingComponents[c]['FeatureVector'] = trainingComponents[c]['CountVectorizer'].transform(X[c])
        testingComponents[c]['TransformedTfidt'] = trainingComponents[c]['TfidfTransformer'].transform(testingComponents[c]['FeatureVector'])

    test_tfidf = hstack([testingComponents[c]['TransformedTfidt'] for c in columns], format = 'csr')
    test_tfidf = hstack((test_tfidf,
                         np.array(X['Sentiment'])[:, None],
                         np.array(X['Subjectivity'])[:, None]))
    
    return test_tfidf


# Perform KNN testing
def knnAnalysisTesting(df_training, n_neighbors = 7):
    # Split data into training and testing sets
    y = df_training.pop('Category')
    X = df_training
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size = 0.2, stratify = y)
    X_train = X.iloc[X_train]
    X_test = X.iloc[X_test]
    
    # Prepare the training components and KNN model
    trainingComponents, clf = prepareModellingComponents(X_train, y_train, n_neighbors)

    # Prepare testing data
    test_tfidf = prepareTestingComponents(X_test, trainingComponents)

    # Make predictions
    predicted = clf.predict(test_tfidf)
    output = X_test
    output['Actual Category'] = y_test
    output['Prediction'] = predicted

    # Print the model accuracy
    accuracy = round(np.mean(predicted == y_test), 2)

    return output, accuracy


# Loop through various KNN tests
def knnAnalysisTestLoop(sampleRange = [1000, 2500, 5000, 10000, 25000, 50000], nNeighbourRange = [3, 5, 7, 9, 11]):
    def checkPrediction(row):
        if row['Actual Category'] == row['Prediction']:
            return 'Correct'
        else:
            return 'Incorrect'
    
    df_originalData = prepareMachineLearningData()
    output = {}
    c = 0
    for sampleSize in sampleRange:
        for n_neighbours in nNeighbourRange:
            for method in range(0, 5):
                output[c] = {
                    'Samples': sampleSize,
                    'Neighbours': n_neighbours
                    }
                
                check = 0
                while check == 0:
                    try:
                        df = df_originalData.copy()
                        df = applyTrainingMethod(df, sampleSize, method)
                        
                        if method == 0: output[c]['Method'] = 'Original'
                        elif method == 1: output[c]['Method'] = 'Weightings altered'
                        elif method == 2: output[c]['Method'] = 'Categories reduced'
                        elif method == 3: output[c]['Method'] = 'Weightings altered then categories reduced'
                        else: output[c]['Method'] = 'Categories reduced then weightings altered'
                    
                        output[c]['Category Dist.'] = df['Category'].value_counts(normalize = True)
                        df_predictions, accuracy = knnAnalysisTesting(df, n_neighbours)
                        check = 1
                    except:
                        pass
                    
                # Clean the prediction DataFrame
                df_predictions['Accurate Prediction'] = df_predictions.apply(lambda row: checkPrediction(row), axis = 1)
                df_predictions = df_predictions.groupby(['Actual Category','Accurate Prediction']).size().reset_index()
                df_predictions = df_predictions.pivot(index = 'Actual Category', columns = 'Accurate Prediction', values = 0).reset_index()
                df_predictions['Correct'] = df_predictions['Correct'].fillna(0)
                df_predictions['Incorrect'] = df_predictions['Incorrect'].fillna(0)
                df_predictions['Accuracy'] = df_predictions['Correct'] / (df_predictions['Correct'] + df_predictions['Incorrect'])
                df_predictions.index.name = 'Index'        
        
                output[c]['Accuracy'] = accuracy
                output[c]['Pred. Accuracy'] = df_predictions
                c += 1
                print('{} / {} complete. Samples: {}, Neighbours: {}, Method: {}.'.format(c, (len(sampleRange) * len(nNeighbourRange) * 5), sampleSize, n_neighbours, method))

    # Prepare DataFrames to output to CSV
    df_keys = pd.DataFrame(columns = ['Samples', 'Neighbours', 'Method', 'Accuracy'])
    df_categoryDist = pd.DataFrame(columns = ['Category', 'Distribution', 'Key'])
    df_precAccuracy = pd.DataFrame(columns = ['Category', 'Correct', 'Incorrect', 'Accuracy', 'Key'])
    for key, value in output.items():
        i = len(df_keys)
        df_keys.loc[i] = [value['Samples'], value['Neighbours'], value['Method'], value['Accuracy']]
        
        df_temp_categoryDist = pd.DataFrame(value['Category Dist.']).reset_index()
        df_temp_categoryDist['Key'] = i
        df_temp_categoryDist.rename(columns = {'index': 'Category', 'Category': 'Distribution'}, inplace = True)
        df_categoryDist = pd.concat([df_categoryDist, df_temp_categoryDist])
                    
        df_temp_precAccuracy = value['Pred. Accuracy'].reset_index(drop = True).rename(columns = {'Accurate Preduction': 'Index', 'Actual Category': 'Category'})
        df_temp_precAccuracy['Key'] = i
        df_precAccuracy = pd.concat([df_precAccuracy, df_temp_precAccuracy])
    
    # Output results to CSV
    filePath = 'Data Collection\\Summary data\\Data model tables\\Knn testing\\'
    df_keys.to_csv(filePath + 'keys.csv')
    df_categoryDist.to_csv(filePath + 'category distribution.csv', index = False)
    df_precAccuracy.to_csv(filePath + 'prediction accuracy.csv', index = False)


# Perform KNN testing, while reporting on True/False-Positive/Negative counts
def accuracyMeasurements(sampleSize = 10000, n_neighbors = 7, method = 0):
    # Return true/false-positive/negative indicators
    def checkPrediction(row, checkFor):
        if row['Actual Category']  == row['Prediction']:
            prefix = 'True'
        else:
            prefix =  'False'
            
        if row['Prediction'] == checkFor:
            suffix = 'Positive'
        else:
            suffix = 'Negative'
            
        return prefix + '-' + suffix
    
    
    df = prepareMachineLearningData()
    df = applyTrainingMethod(df, sampleSize, method)
    
    # Split data into training and testing sets
    y = df.pop('Category')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size = 0.2, stratify = y)
    X_train = X.iloc[X_train]
    X_test = X.iloc[X_test]
    
    # Prepare the training components and KNN model
    trainingComponents, clf = prepareModellingComponents(X_train, y_train, n_neighbors)

    # Prepare testing data
    test_tfidf = prepareTestingComponents(X_test, trainingComponents)

    # Make predictions
    predicted = clf.predict(test_tfidf)
    
    # Add predictions and accuracy checks to output
    output = X_test
    output['Actual Category'] = y_test
    output['Prediction'] = predicted
    output['V/S Check'] = output.apply(lambda row: checkPrediction(row, 'Victim/Survivor'), axis = 1)
    output['Other Check'] = output.apply(lambda row: checkPrediction(row, 'Other'), axis = 1)

    # Generate the model accuracy
    generalAccuracy = round(np.mean(predicted == y_test), 2)
    
    return output, generalAccuracy


# Loop through various configurations of accuracyMeasurements()
def accuracyMeasurementLoop(sampleSize = 10000, n_neighbours = 7):
    output = pd.DataFrame(columns = ['Category', 'Accuracy', 'Count'])
    for m in range(0, 5):
        check = 0
        while check == 0:
            try:
                df, acc = accuracyMeasurements(sampleSize, n_neighbours, m)
                df_summary = df.groupby(['Actual Category', 'V/S Check'])['index'].count().reset_index().rename(columns = {'Actual Category': 'Category', 'V/S Check': 'Accuracy', 'index': 'Count'})
                df_summary['Method'] = m
                df_summary['General Accuracy'] = acc
                output = pd.concat([output, df_summary])
                check = 1
            except:
                pass
        
        print('Method: {}, General Accuracy: {}%'.format(m, acc * 100))
        
    output['Samples'] = sampleSize
    output['Neighbour'] = n_neighbours
    # output.to_csv('Data Collection\\Method Testing (s = {}, k = {}).csv'.format(sampleSize, n_neighbours), index = False)
    output.to_csv('Data Collection\\Method testing\\Method testing (s = {}, k = {}).csv'.format(sampleSize, n_neighbours), index = False)
    

# Generate the KNN machine learning model
def generateKnnModel(sampleSize = 10000, n_neighbours = 7, method = 4):
    # Prepare training data
    df = prepareMachineLearningData().reset_index(drop = True)
    df = applyTrainingMethod(df, sampleSize, method)
    
    # Split data into training and testing sets
    y = df.pop('Category')
    X = df
    
    # Prepare the training components and KNN model
    trainingComponents, clf = prepareModellingComponents(X, y, n_neighbours)

    # Prepare validation data
    df_validation = prepareDataFrame(trainingData = False)
    test_tfidf = prepareTestingComponents(df_validation, trainingComponents)

    # Make predictions
    predicted = clf.predict(test_tfidf)
    df_validation['Prediction'] = predicted

    return df_validation


# Parent function for modelling the Tweets from January
def runKnnModelForJanuary(sampleSize = 5000, n_neighbours = 9, method = 0):
    df = generateKnnModel(sampleSize, n_neighbours, method)
    df_rawJanuary = pd.read_csv('Data collection\\Jan-22 Tweets.csv')
    df_out = pd.merge(df, df_rawJanuary[['ID', 'Display Name', 'Content', 'URL']], on = 'ID')
    df_out = df_out[['ID', 'Username', 'Display Name', 'Content', 'Sentiment', 'Subjectivity', 'Mentions', 'Hashtags', 'Prediction', 'URL']]
    df_out.to_csv('Data Collection\\Jan-22 Predictions.csv', index = False)   
    return df_out
    

if __name__ == "__main__":
    knnAnalysisTestLoop()
    accuracyMeasurementLoop()
    df = runKnnModelForJanuary(25000, 9, 3)