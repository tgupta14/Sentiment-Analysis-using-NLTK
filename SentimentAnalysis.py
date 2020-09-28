
# Before running this code, please ensure that Python and NLTK is installed

''' NLTK is a very large toolkit, and several of its tools actually require 
a second download step to gather the necessary collection of data 
(often coded lexicons) to function correctly. 

Please execute Installation.py before running this code if you are working
on a new computer on which these libraries have never been used.

If Installation.py script has not been executed on the machine and the script is 
missing for whatsoever reason, please remove the comments ('#' symbol) and 
run the whole code the first time you open this program on a new machine. 
Please comment them again after it runs the first time to save time on later executions.'''

# import nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt')


# First, we import the relevant modules from the NLTK library, and other libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np



def doSentimentAnalysis(filename, extensionName, index):
    # We initialize VADER so we can use it within our Python script
    sid = SentimentIntensityAnalyzer()

    file= filename+extensionName
    if extensionName=='.csv':
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    headers=list(df.columns.values)
    
    # df is our dataframe with headers and values
    # Convert dataframe into an array by pulling only its values
    dataset= df.values
    row, column= np.shape(dataset)
    
    # Add 4 extra columns to store the sentiment scores
    extra_columns= np.zeros((row,4))
    
    # This is the structure of our final output, with all the original data plus 
    # the 4 sentiment scores. We will add the headers later.
    final_output= np.append(dataset, extra_columns, axis=1)
    
    # Now read each text message one at a time, calculate its sentiment score and 
    # store that sentiment score with the original message.
    
    for i in range(row):
        message_text= str(dataset[i,index-1])
        # The variable 'message_text' now contains the incoming text we will analyze.
        # Calling the polarity_scores method on sid and passing in the message_text 
        # outputs a dictionary with negative, neutral, positive, and compound scores for the input text
        scores = sid.polarity_scores(message_text)
        final_output[i,index-1]= message_text
        
        counter=0
        for key in sorted(scores, reverse=True):
            #print('{0}: {1}, '.format(key, scores[key]), end='')
            final_output[i,column+counter]= scores[key]
            counter=counter+1
            
    return final_output, headers



''' Read the csv/xls/xlsx worksheet. For csv, the command is different than the
rest of the two so code has been written to read the correct file. 

It is preferred to not use .csv and if possible, please save the file as .xlsx 
or .xls

Ensure that the excel worksheet is at the same local folder as this Python script 
and take care for the file extension type .xls or .xlsx or .csv'''


'''Enter the name of the file, its extension type, and the column number of the 
message to be analyzed.'''

# Example filename= 'Retention Sunny Chatlog 10.2018 to 7.18.2019'
filename = input("Enter the excel filename without the extension: ")

# Example extensionName='.xls'
extensionName= input("Enter the extension of the file (.csv/.xls/.xlsx): ")

# Example messageColumnNumber=5
messageColumnNumber= int(input("Enter the integer column number of the string message to be analyzed: "))


finalOutput, headers= doSentimentAnalysis(filename, extensionName, messageColumnNumber)

# We will now add column names to the data we have created in order to store it as
# an excel sheet. So we add 4 column names to the original headers.
headers.extend(("Positive", "Neutral", 'Negative', 'Compound'))

saveName= filename+' - Sentiment Analysis.xls'
# Convert the array to dataframe and store it as an xls worksheet
finalDataframe = pd.DataFrame(finalOutput, columns = headers)
finalDataframe.to_excel(saveName)

