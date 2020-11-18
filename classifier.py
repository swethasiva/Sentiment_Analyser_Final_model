import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag 

test_size = 0.2
vocab_size = 40000
trunc_type = 'post'
pad_type = 'post'
embedd_dim = 64
lstm_out = 140
epochs = 5
batch_size = 128

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

global tokenizer
english_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

from keras.callbacks import Callback

lemmatizer = WordNetLemmatizer()

def load_dataset(filename):
	data = pd.read_csv(filename)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	for review in reviews:
		for word in review.split():
			if word.lower() in contractions:
				review = review.replace(word, contractions[word.lower()])
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	print(reviews[0]+"\n\n")
	reviews = reviews.apply(lambda review : [w for w in review.split() if not w.lower() in english_stopwords])
	reviews = reviews.apply(lambda review: [w.lower() for w in review]) 
	'''for review in reviews:
		for word in review:
			if word.lower() in contractions:
				review = review.replace(word, contraction[word.lower()])'''
	print(reviews[0])
	'''for word in reviews.split():
    if word.lower() in contractions:
        text = t.replace(word, contractions[word.lower()])'''
	#reviews = reviews.apply(lambda review: [lemmatizer.lemmatize(w, 'a' if tag[0].lower() == 'j' else tag[0].lower()) for w, tag in pos_tag(review) if tag[0].lower() in ['j', 'v', 'n', 'r']])
	'''for review in reviews:
		stopword_removed_reviews.append(' '.join([word.lower() for word in review.split('\t') if word not in english_stopwords]))'''
	label_encoder = preprocessing.LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	return reviews, labels

def split_dataset(reviews, labels):
	reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size = 0.2)
	return reviews_train, reviews_test, labels_train, labels_test

def get_max_length(reviews):
	review_length = []
	for review in reviews:
		review_length.append(len(review))
	return int(np.ceil(np.mean(review_length)))

def tokenize_pad_trunc(train_reviews, test_reviews, max_len):
	tokenizer = Tokenizer(num_words = vocab_size, lower= False)
	tokenizer.fit_on_texts(train_reviews)
	train_reviews = tokenizer.texts_to_sequences(train_reviews)
	test_reviews = tokenizer.texts_to_sequences(test_reviews)
	train_reviews = pad_sequences(train_reviews, maxlen=max_len, padding= pad_type,truncating= trunc_type)
	test_reviews = pad_sequences(test_reviews, maxlen=max_len, padding= pad_type ,truncating= trunc_type)
	total_words = int(len(tokenizer.word_index) + 1)
	return tokenizer, train_reviews, test_reviews, total_words

def save_tokenizer(tokenizer):
	tokenizer_json = tokenizer.to_json()
	with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))
'''
class stopAtLossValue(Callback):
	def on_batch_end(self, batch, logs={}):
		loss_threshold = 0.2
		if logs.get('loss') <= loss_threshold:
			print('Loss reached threshold, stopping training')
			self.model.stop_training = True '''

def sentiment_classification_model(total_words, max_len):
	model = tf.keras.Sequential([
	  tf.keras.layers.Embedding(total_words, embedd_dim, input_length= max_len),
	  tf.keras.layers.LSTM(lstm_out),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(1, activation='sigmoid')])
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	print(model.summary())
	return model
'''
def checkpoint_model():
	checkpoint = ModelCheckpoint(
	  'models/LSTM.h5',
	  monitor='accuracy',
	  save_best_only=True,
	  verbose=1
	)
	return checkpoint '''

def plot_training(history):
	import matplotlib.pyplot as plt
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))
	plt.plot(epochs, acc, 'r', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig('best_accu.png')
	plt.figure()
	plt.plot(epochs, loss, 'r', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('best_loss.png')
	plt.show()

def main():
	data = load_dataset('IMDB Dataset.csv')
	reviews, labels = clean_data(data)
	reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
	max_len = get_max_length(reviews)
	#print(max_len)
	#print(type(reviews_train))
	tokenizer, reviews_train, reviews_test, total_words = tokenize_pad_trunc(reviews_train, reviews_test, max_len)
	#print(tokenizer.word_index)
	save_tokenizer(tokenizer)
	model = sentiment_classification_model(total_words, max_len)
	#checkpoint = checkpoint_model()
	#callbacks = stopAtLossValue()
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
	mc = ModelCheckpoint('best_model_1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	history = model.fit(reviews_train, labels_train, validation_data = (reviews_test, labels_test), batch_size = batch_size, epochs = epochs, callbacks=[es, mc])
	#plot_training(history)
	#model.save('20_earlystopping_dropout20_40000Vocab_sigmoid_lstm130.h5')

if __name__ == "__main__":
	main()
	print('calling')

