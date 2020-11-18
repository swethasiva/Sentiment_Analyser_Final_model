import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

contractions = {
"ain't": "am not",
"won't" : "will not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

lemmatizer = WordNetLemmatizer()

def main():
	english_stopwords = set(stopwords.words('english'))
	english_stopwords.remove('not')
	max_len = 100
	loaded_model = load_model('best_model_1.h5')
	with open('tokenizer.json') as f:
	    data = json.load(f)
	    tokenizer = tokenizer_from_json(data)

	review = input('Enter review to predict sentiment:')
	for word in review.split(' '):
		if word.lower() in contractions:
			review = review.replace(word, contractions[word.lower()])
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print('Cleaned: ', review)
	words = review.split(' ')
	filtered = [w for w in words if w.lower() not in english_stopwords]
	print(filtered)
	if(len(review.split(' '))<=15):
		flag = [True for word in filtered if word.lower() in ['not', 'no', 'never']]
		if(len(flag)==0):
			flag=[False]
	
	#filtered = [lemmatizer.lemmatize(w, 'a' if tag[0].lower() == 'j' else tag[0].lower()) for w, tag in pos_tag(filtered) if tag[0].lower() in ['j', 'r', 'n', 'v']]
	#print(filtered)
	filtered = ' '.join(filtered)
	filtered = [filtered.lower()]
	print('Filtered: ', filtered)
	tokenize_words = tokenizer.texts_to_sequences(filtered)
	tokenize_words = pad_sequences(tokenize_words, maxlen=max_len, padding='post', truncating='post')
	print(tokenize_words)
	result = loaded_model.predict(tokenize_words)
	print(result)
	#print(flag)
	#print(len(review))
	if (len(review.split(' '))<=15):
		if result >= 0.5:
		    if (flag[0] == True):
		    	print('Negative')
		    	print("Sentiment Probability \n Positive: " + str(1 - result[0][0]) + ", Negative: " + str(result[0][0]))
		    else:
		    	print('positive')
		    	print("Sentiment Probability \n Positive: " + str(result[0][0]) + ", Negative: " + str(1 - result[0][0]))
		else:
			if(flag[0] == True):
				print('positive')
				print("Sentiment Probability \n Positive: " + str(1 - result[0][0]) + ", Negative: " + str(result[0][0]))
			else:
				print('negative')
				print("Sentiment Probability \n Positive: " + str(result[0][0]) + ", Negative: " + str(1 - result[0][0]))
	else:
		if(result>=0.5):
			print('positive review')
		else:
			print('negative review')

	return


if __name__ == '__main__':
	main()
