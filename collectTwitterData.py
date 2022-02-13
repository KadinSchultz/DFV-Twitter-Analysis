import os
import pandas as pd
import re
import snscrape.modules.twitter as sntwitter

from ast import literal_eval
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from gensim import corpora, models
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from time import time

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


# Import information from Twitter based on a keyword and date range
def scrapeTweets(keyword = 'domestic violence', startDate = datetime(2021, 12, 1), endDate = datetime(2022, 1, 1)):
    startTime = time()
    currentDate = startDate

    # Process the imports in monthly chunks
    while currentDate <= endDate:
        loopStart = time()
        tweetList = []

        sinceDateStr = currentDate.strftime('%Y-%m-%d')
        untilDateStr = (currentDate + relativedelta(months = 1)).strftime('%Y-%m-%d')
        scrapeString = '{} since:{} until:{} lang:en'.format(keyword, sinceDateStr, untilDateStr)
        print('Starting w/ search string: [{}] ...'.format(scrapeString))
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(scrapeString).get_items()):
         	tweetList.append([tweet.id,tweet.date, tweet.user.username, tweet.user.displayname,
                            tweet.content, tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.quoteCount,
                            tweet.mentionedUsers, tweet.hashtags])

        fileName = 'Data Collection\\Data to process\\[{}] tweets {}-{}.csv'.format(keyword, currentDate.strftime('%Y%m%d'), (currentDate + relativedelta(months = 1, days = -1)).strftime('%Y%m%d'))
        dfTweets = pd.DataFrame(tweetList, columns = ['ID', 'Date', 'Username', 'Display Name', 'Content', 'Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Mentioned Users', 'Hashtags'])
        dfTweets.to_csv(fileName)

        currentDate += relativedelta(months = 1)
        print('{} --- {} Tweets downloaded in {}.'.format(datetime.now().strftime("%H:%M:%S"), currentDate.strftime('%B-%y'), str(timedelta(seconds = round(time() - loopStart)))))

    print('Imports completed in {}.'.format(str(timedelta(seconds = round(time() - startTime)))))


# Remove unwanted content from a piece of text
def removeContent(text):
    # Remove &amp; symbols
    text = re.sub(r'&amp;', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+\.com\S+', '', text)

    # Remove mentions
    text = re.sub(r'\@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'\#\w+', '', text)
    return text


# Clean the input text
def processText(text, stem = False): #clean text
    text = removeContent(text)

    # Remove non-alphabet characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())
    tokenized_text = word_tokenize(text)

    # Remove words from NLTK's stopword library
    clean_text = [word for word in tokenized_text if word not in stopwords.words('english')]

    # if stem:
    #     clean_text = [stemmer.stem(word) for word in clean_text]

    return ' '.join(clean_text)

# Generate sentiment analysis figures based on Tweet contents
def sentimentAnalysis(df):
    df['Sentiment'] = df['Content'].apply(lambda x: TextBlob(x).sentiment[0])
    df['Polarity'] = df['Sentiment'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df['Subjectivity'] = df['Content'].apply(lambda x: TextBlob(x).sentiment[1])
    return df


# Generate the text data for topics and apply them to the DataFrame
def topicModelling():
    df = pd.read_csv('Data Collection\\Tweets.csv')
    df['Content'] = df['Content'].fillna('')
    tweets = df['Content'].dropna().tolist()

    # Pre-process tweets to a 'bag of words' (BOW)
    r = [s.split() for s in tweets]
    dictionary = corpora.Dictionary(r)
    corpus = [dictionary.doc2bow(rev) for rev in r]

    # Initialize the model, and print the topics
    model = models.ldamodel.LdaModel(corpus, num_topics = 10, id2word = dictionary, passes = 15)
    topics = model.print_topics(num_words = 5)

    # Add topic keys to the DataFrame
    labelKeys = []
    for x in model[corpus]:
        labelKeys.append(sorted(x, key = lambda x: x[1], reverse = True)[0][0])
    df['Topic'] = pd.Series(labelKeys)

    # Add topic labels to the DataFrame
    dfTopics = pd.DataFrame([processText(t[1]) for t in topics], columns = ['Topic Label'])
    dfTopics.index.names = ['Topic']
    df = df.join(dfTopics, on = 'Topic', lsuffix = '', rsuffix = '')

    # Output the results
    df.to_csv('Tweet Topics.csv', index = False)


# Process the raw data from Twitter, and split into Tweets, Hashtags, and Mentions
def processRawData(inputFile):
    # Start processing raw data
    df = pd.read_csv(inputFile)
    df.drop(['Unnamed: 0'], axis = 1, inplace = True)

    df['Content'] = df['Content'].apply(lambda word: processText(word))
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d').dt.date
    df['Month'] = df['Date'].apply(lambda d: datetime(d.year, d.month, 1)).dt.date

    df = sentimentAnalysis(df)

    # Process mentions
    cols = ['ID', 'Date', 'Month', 'Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Sentiment', 'Polarity', 'Subjectivity', 'Mentioned Users']
    df_mentions = df[df['Mentioned Users'].notnull()][cols]

    df_mentions['Mentioned Username'] = df_mentions['Mentioned Users'].apply(lambda mention: re.findall("username='.*?'", mention))
    df_mentions['Mentioned Username'] = df_mentions['Mentioned Username'].apply(lambda usernames: [uName[10:-1] for uName in usernames])

    df_mentions['Mentioned Display Name'] = df_mentions['Mentioned Users'].apply(lambda mention: re.findall(r'displayname=.*?,', mention))
    df_mentions['Mentioned Display Name'] = df_mentions['Mentioned Display Name'].apply(lambda displayname: [dName[13:-2] for dName in displayname])

    df_mentions.drop(['Mentioned Users'], axis = 1, inplace = True)
    df_mentions.rename(columns = {'Mentioned Username': 'Username', 'Mentioned Display Name': 'Display Name'}, inplace = True)

    df_mentionsExploded = df_mentions.explode(['Username', 'Display Name'])
    df_mentions['Username'] = df_mentions['Username'].astype(str).apply(lambda word: word.replace(',', '').replace('[', '').replace(']', '').replace("'", ''))
    df_mentions['Display Name'] = df_mentions['Display Name'].astype(str).apply(lambda word: word.replace(',', '').replace('[', '').replace(']', '').replace("'", ''))

    # Process hashtags
    cols = ['ID', 'Date', 'Month', 'Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Sentiment', 'Polarity', 'Subjectivity', 'Hashtags']
    df_hashtags = df[df['Hashtags'].notnull()][cols]
    df_hashtags['Hashtags'] = df_hashtags['Hashtags'].apply(lambda hashtag: literal_eval(hashtag))

    df_hashtagsExploded = df_hashtags.explode('Hashtags')
    df_hashtagsExploded['Hashtags'] = df_hashtagsExploded['Hashtags'].apply(lambda word: '#{}'.format(word.lower()))
    df_hashtags['Hashtags'] = df_hashtags['Hashtags'].astype(str).apply(lambda word: word.replace(',', '').replace('[', '').replace(']', '').replace("'", '').lower())

    # Finish processing raw data
    cols = ['ID', 'Date', 'Month', 'Username', 'Display Name', 'Content', 'Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Sentiment', 'Polarity', 'Subjectivity', 'Mentioned Users', 'Hashtags']
    df = df[cols]

    df.drop(columns = ['Mentioned Users', 'Hashtags'], inplace = True)
    df = df.merge(df_mentions[['ID', 'Username', 'Display Name']].rename(columns = {'Username': 'Mentioned Usernames', 'Display Name': 'Mentioned Display Names'}), how = 'left', on = 'ID', suffixes = ('', ''))
    df = df.merge(df_hashtags[['ID', 'Hashtags']], how = 'left', on = 'ID', suffixes = ('', ''))

    # Generate processed CSVs
    df.to_csv('Data Collection\\Processed\\Processed Tweets\\{}'.format(inputFile[inputFile.rfind('\\') + 1:]), index = False)

    df_mentions.to_csv('Data Collection\\Processed\\Processed Mentions\\{}'.format(inputFile[inputFile.rfind('\\') + 1:]), index = False)
    df_mentionsExploded.to_csv('Data Collection\\Processed\\Processed Mention Lists\\{}'.format(inputFile[inputFile.rfind('\\') + 1:]), index = False)

    df_hashtags.to_csv('Data Collection\\Processed\\Processed Hashtags\\{}'.format(inputFile[inputFile.rfind('\\') + 1:]), index = False)
    df_hashtagsExploded.to_csv('Data Collection\\Processed\\Processed Hashtag Lists\\{}'.format(inputFile[inputFile.rfind('\\') + 1:]), index = False)

    # Archived the processed data
    fileDestination = 'Data Collection\\Processed\\Raw Data\\{}'.format(inputFile[inputFile.rfind('\\') + 1:])
    os.replace(inputFile, fileDestination)

    print('{} --- {} done.'.format(datetime.now().strftime("%H:%M:%S"), inputFile))


# Loop through all non-archived files and process their data
def processTweetFiles():
    # List unprocessed files
    inputFilePath = 'Data Collection\\Data to process\\'
    fileList = []
    for path, subdirs, files in os.walk(inputFilePath):
        for name in files:
            fileList.append(os.path.join(path, name))

    print('Processing {} files. Start time: {}.'.format(len(fileList), datetime.now().strftime("%H:%M:%S")))

    # Process files
    for inputFile in fileList:
        processRawData(inputFile)

    # Combine processed files into single CSV
    inputFilePathList = ['Data Collection\\Processed\\Processed Tweets',
                         'Data Collection\\Processed\\Processed Hashtags', 'Data Collection\\Processed\\Processed Hashtag Lists',
                         'Data Collection\\Processed\\Processed Mentions', 'Data Collection\\Processed\\Processed Mention Lists']
    for inputFilePath in inputFilePathList:
        fileList = []
        for path, subdirs, files in os.walk(inputFilePath + '\\'):
            for name in files:
                if name[-4:] == '.csv':
                    fileList.append(os.path.join(path, name))

        combinedCSV = pd.concat([pd.read_csv(f) for f in fileList])
        outputFile = 'Data Collection\\Data to process\\{}.csv'.format(inputFilePath[36:])
        combinedCSV.to_csv('{}.csv'.format(outputFile), index = False)
        print('{} completed.'.format(outputFile))


if __name__ == "__main__":
    scrapeTweets()
    processTweetFiles()
