import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

#personName = input('Enter your full name: ')
#linkedInData = input('Do you have LinkedIn data to parse through (y/n)?')
personName = "Piyush Soni"
linkedInData = "y"

def getLinkedInData():
	df = pd.read_csv('/media/piyush/New Volume/Projects/FB_chatbot/data/linkedin/messages.csv')
	dateTimeConverter = lambda x: datetime.strptime(x.replace("T", ' '),'%Y-%m-%d %H:%M:%S')
	responseDictionary = dict()
	peopleContacted = df['FROM'].unique().tolist()
    
	for person in peopleContacted:
		print(person)

		receivedMessages = df[df['FROM'] == person]
		sentMessages = df[df['TO'] == person]

		if (len(sentMessages) == 0 or len(receivedMessages) == 0):
			continue#    There was no actual conversation
		combined = pd.concat([sentMessages, receivedMessages])
		combined['DATE'] = combined['DATE'].apply(dateTimeConverter)
		combined = combined.sort_values(['DATE'])
		otherPersonsMessage, myMessage = "",""
		firstMessage = True
		for index, row in combined.iterrows():
			if (row['FROM'] != personName):
				if myMessage and otherPersonsMessage:
					otherPersonsMessage = cleanMessage(otherPersonsMessage)
					myMessage = cleanMessage(myMessage)
					responseDictionary[otherPersonsMessage.rstrip()] = myMessage.rstrip()
					otherPersonsMessage, myMessage = "",""
			else:
				if (firstMessage):
					firstMessage = False
					# Don't include if I am the person initiating the convo
					continue
				myMessage = myMessage + str(row['CONTENT']) + " "

			print(otherPersonsMessage)

			otherPersonsMessage = otherPersonsMessage + str(row['CONTENT']) + " "
	return responseDictionary


def cleanMessage(message):
	# Remove new lines within message
	cleanedMessage = message.replace('\n',' ').lower()
	# Deal with some weird tokens
	cleanedMessage = cleanedMessage.replace("\xc2\xa0", "")
	# Remove punctuation
	cleanedMessage = re.sub('([.,!?])','', cleanedMessage)
	# Remove multiple spaces in message
	cleanedMessage = re.sub(' +',' ', cleanedMessage)
	return cleanedMessage

combinedDictionary = {}

if (linkedInData == 'y'):
	print('Getting LinkedIn Data')
	combinedDictionary.update(getLinkedInData())

print('Total len of dictionary', len(combinedDictionary))

print('Saving conversation data dictionary')
np.save('conversationDictionary.npy', combinedDictionary)

conversationFile = open('conversationData.txt', 'w')

for key,value in combinedDictionary.items():
    if (not key.strip() or not value.strip()):
        continue    # If there are empty strings
        
    conversationFile.write(key.strip() + value.strip())

