import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import json
import io
from nltk.stem import WordNetLemmatizer
import pickle

#english_stopwords = set(stopwords.words('english'))
#english_stopwords.remove('not')
english_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
lemmatizer = WordNetLemmatizer()

def load_dataset(filename):
	data = pd.read_csv(filename)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	stopword_removed_reviews = []
	lemmatized_reviews = []
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	for review in reviews:
		stopword_removed_reviews.append(' '.join([word.lower() for word in review.split(' ') if not word in english_stopwords]))
	label_encoder = preprocessing.LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	'''for review in stopword_removed_reviews:
		lemmatized_reviews.append(' '.join([lemmatizer.lemmatize(word) for word in review.split('\t')]))'''
	return stopword_removed_reviews, labels

def split_dataset(reviews, labels):
	reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size = 0.0002)
	return reviews_train, reviews_test, labels_train, labels_test

def Ngram_Vectorizer(reviews_train, reviews_test):
	tfidf = TfidfVectorizer(analyzer = 'word', ngram_range=(1,4))
	tfidf.fit(reviews_train)
	#print(tfidf.vocabulary_)
	feature_names = tfidf.get_feature_names()
	#print(feature_names)
	pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
	reviews_train = tfidf.transform(reviews_train)
	reviews_test = tfidf.transform(reviews_test)
	return reviews_train, reviews_test

def LogisticRegressor(reviews_train, labels_train, reviews_test, labels_test):
	for c in [1]:
	    lr = LogisticRegression(C=c)
	    lr.fit(reviews_train, labels_train)
	    print ("Accuracy for C=%s: %s" 
	           % (c, accuracy_score(labels_test, lr.predict(reviews_test))))
	filename = 'tfidf_bigram_model.sav'
	pickle.dump(lr, open(filename, 'wb'))
    

def main():
	data = load_dataset('IMDB Dataset.csv')
	reviews, labels = clean_data(data)
	reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
	reviews_train.append('it was not recommended')
	reviews_train.append('definitely not recommended')
	reviews_train.append('it was not bad')
	reviews_train.append('not at all bad')
	reviews_train.append('wasnt bad')
	reviews_train.append('it was not that bad')
	reviews_train.append('honestly not bad')
	reviews_train.append('wasnt bad')
	reviews_train.append('it was not bad')
	reviews_train.append('not bad')
	reviews_train.append('wasnt bad')
	reviews_train.append('it was not bad')
	reviews_train.append('not bad')
	reviews_train.append('wasnt bad')
	reviews_train.append('it was not bad')
	reviews_train.append('not bad')
	reviews_train.append('wasnt bad')
	reviews_train.append('it was not bad')
	reviews_train.append('not a bad movie')
	reviews_train.append('wasnt bad')
	reviews_train.append('trust me it was not bad')
	reviews_train.append('I can vouch that it was not bad')
	reviews_train.append('not bad a movie')
	reviews_train.append('not bad a performance')
	reviews_train.append('not a bad performance')
	reviews_train.append('not bad in terms of acting')
	reviews_train.append('it was not bad')
	reviews_train.append('not a bad movie')
	reviews_train.append('wasnt bad')

	print((labels_train))
	labels_train = np.append(labels_train, int(0))
	labels_train = np.append(labels_train, int(0))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))
	labels_train = np.append(labels_train, int(1))


	#print(len(labels_train))
	#print(len(reviews_train))
	reviews_train, reviews_test= Ngram_Vectorizer(reviews_train, reviews_test)
	LogisticRegressor(reviews_train, labels_train, reviews_test, labels_test)

if __name__ == "__main__":
	main()
	print('calling')
