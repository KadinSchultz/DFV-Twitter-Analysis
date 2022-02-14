import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None

# Global variables
fileTweets = 'Data Collection\\Tweets.csv'
fileTweetTopics = 'Data Collection\\Tweet Topics.csv'
fileHashtags = 'Data Collection\\Hashtags.csv'
fileHashtagList = 'Data Collection\\Hashtag Lists.csv'
fileMentions = 'Data Collection\\Mentions.csv'
fileMentionList = 'Data Collection\\Mention Lists.csv'


# Collect and format data from CSVs for use in analysis functions
def collectData():
    # Import Tweet data
    dfTweets = pd.read_csv(fileTweets)
    dfTweets['Content'] = dfTweets['Content'].fillna('').dropna()
    dfTweets['Hashtags'] = dfTweets['Hashtags'].fillna('').dropna()
    dfTweets['Mentioned Usernames'] = dfTweets['Mentioned Usernames'].fillna('').dropna()
    dfTweets['Mentioned Display Names'] = dfTweets['Mentioned Display Names'].fillna('').dropna()

    # Import Tweet topic data
    dfTweetTopics = pd.read_csv(fileTweetTopics)
    dfTweets['Content'] = dfTweets['Content'].fillna('').dropna()
    dfTweets['Hashtags'] = dfTweets['Hashtags'].fillna('').dropna()
    dfTweets['Mentioned Usernames'] = dfTweets['Mentioned Usernames'].fillna('').dropna()
    dfTweets['Mentioned Display Names'] = dfTweets['Mentioned Display Names'].fillna('').dropna()

    # Import Hashtag data
    dfHashtags = pd.read_csv(fileHashtags)
    dfHashtags['Hashtags'] = dfHashtags['Hashtags'].fillna('').dropna()

    # Import Hashtag lists data
    dfHashtagLists = pd.read_csv(fileHashtagList)
    dfHashtagLists['Hashtags'] = dfHashtagLists['Hashtags'].fillna('').dropna()

    # Import Mention data
    dfMentions = pd.read_csv(fileMentions)
    dfMentions['Username'] = dfMentions['Username'].fillna('').dropna()
    dfMentions['Display Name'] = dfMentions['Display Name'].fillna('').dropna()

    # Import Mention lists data
    dfMentionLists = pd.read_csv(fileMentionList)
    dfMentionLists['Username'] = dfMentionLists['Username'].fillna('').dropna()
    dfMentionLists['Display Name'] = dfMentionLists['Display Name'].fillna('').dropna()

    return dfTweets, dfTweetTopics, dfHashtags, dfHashtagLists, dfMentions, dfMentionLists


# Get the most used [top] words, with a min and max length of ngram_range = (min, max)
def getTopWords(wordList, ngram_range = (1, 1), top = 20, firstword = '', exclusions = []):
    c = CountVectorizer(ngram_range = ngram_range)
    X = c.fit_transform(wordList)
    words = pd.DataFrame(X.sum(axis = 0), columns = c.get_feature_names_out()).T.sort_values(0, ascending = False).reset_index()
    words = words[~words['index'].isin(exclusions)]
    results = words[words['index'].apply(lambda x: firstword in x)].head(top)
    return results


# Count the instances of words with a min and max length of ngram_range = (min, max)
def getWordCounts(wordList, ngram_range = (1, 1)):
    c = CountVectorizer(ngram_range = ngram_range)
    X = c.fit_transform(wordList)
    words = pd.DataFrame(X.sum(axis = 0), columns = c.get_feature_names_out()).T.sort_values(0, ascending = False).reset_index()
    words = words[words[0] > 1]
    return words


# Master function for analytics
def processBaseAnalysis():
    # dfTweets keeps Tweets in the [Content] column, and lists of mentions and hashtags in [Mentioned Usernames], [Mentioned Display Names], [Hashtags] (space-seperated)
    # dfHashtags keeps each Tweet's hashtags in the same cell
    # dfHashtagLists splits each Tweet's hashtags into a seperate row
    # dfMentions keeps each Tweet's mention in the same cell
    # dfMentionLists splits each Tweet's mentions into a seperate row

    dfTweets, dfTweetTopics, dfHashtags, dfHashtagLists, dfMentions, dfMentionLists = collectData()
    pbiPath = 'Data Collection\\Summary data\\'

    # Define the min and max length of each of the 'grams
    nGram = {
        'unigram': (1, 1),
        'bigram': (2, 2),
        'trigram': (3, 3)
        }

    # Generate monthly totals for unigrams, bigrams, and trigrams and output to CSVs
    cols = ['Phrase', 'Count', 'Month']
    dateList = dfTweets['Month'].unique()
    dateList = [dL for dL in dateList if not dL.startswith('2022')]
    for n in nGram:
        topTweets = pd.DataFrame(columns = cols)
        topHashtags = pd.DataFrame(columns = cols)
        topMentions = pd.DataFrame(columns = cols)
        for d in dateList:
            topTweets_sub = getWordCounts(dfTweets['Content'][dfTweets['Month'] == d], ngram_range = nGram[n]).rename(columns = {'index': 'Phrase', 0: 'Count'})
            topTweets_sub['Month'] = d
            topTweets = topTweets.append(topTweets_sub)

            topHashtags_sub = getWordCounts(dfHashtags['Hashtags'][dfHashtags['Month'] == d], ngram_range = nGram[n]).rename(columns = {'index': 'Phrase', 0: 'Count'})
            topHashtags_sub['Month'] = d
            topHashtags = topHashtags.append(topHashtags_sub)

            topMentions_sub = getWordCounts(dfMentions['Username'][dfMentions['Month'] == d], ngram_range = nGram[n]).rename(columns = {'index': 'Phrase', 0: 'Count'})
            topMentions_sub['Month'] = d
            topMentions = topMentions.append(topMentions_sub)

        topTweets.to_csv('{}\\n-grams\\tweet {}s.csv'.format(pbiPath, n), index = False)
        topHashtags.to_csv('{}\\n-grams\\hashtag {}s.csv'.format(pbiPath, n), index = False)
        topMentions.to_csv('{}\\n-grams\\mention {}s.csv'.format(pbiPath, n), index = False)

    # List files for unigram, bigram, and trigram CSVs
    inputFilePath = pbiPath + 'n-grams\\'
    fileList = []
    for path, subdirs, files in os.walk(inputFilePath):
        for name in files:
            fileList.append(os.path.join(path, name))

    # Collate the unigram, bigram, and trigram CSVs into one CSV
    dfCollated = pd.DataFrame(columns = cols)
    for inputFile in fileList:
        fileName = inputFile[inputFile.rfind('\\') + 1:-4]
        category = fileName.split(' ')[0].capitalize()
        nGram = fileName.split(' ')[1].capitalize()

        df = pd.read_csv(inputFile)
        df['Category'] = category
        df['n-gram Type'] = nGram

        dfCollated = dfCollated.append(df)

    dfCollated.to_csv('{}\\Data model tables\\collated ngrams.csv'.format(pbiPath), index = False)

    # Output CSVs for mentions and hashtags for PBI reporting
    dfTweetTopics.drop(columns = ['Month', 'Mentioned Usernames', 'Mentioned Display Names', 'Hashtags']).to_csv('{}\\Data model tables\\tweets.csv'.format(pbiPath), index = False)
    dfHashtagLists[['ID', 'Hashtags']].to_csv('{}\\Data model tables\\hashtags.csv'.format(pbiPath), index = False)
    dfMentionLists[['ID', 'Username', 'Display Name']].to_csv('{}\\Data model tables\\mentions.csv'.format(pbiPath), index = False)


if __name__ == "__main__":
    processBaseAnalysis()