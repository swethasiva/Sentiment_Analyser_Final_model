import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
import io
from nltk.stem import WordNetLemmatizer
import pickle

#english_stopwords = set(stopwords.words('english'))
#english_stopwords.remove('not')
english_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

lemmatizer = WordNetLemmatizer()

def main():
	english_stopwords = set(stopwords.words('english'))
	english_stopwords.remove('not')
	loaded_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
	loaded_model = pickle.load(open('tfidf_bigram_model.sav', 'rb'))
	review = input('Enter review to predict sentiment:')
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print(review.split(' '))
	filtered = []
	filtered.append(' '.join([word.lower() for word in review.split(' ') if not word in english_stopwords]))
	print('Cleaned: ', filtered)
	filtered = loaded_vectorizer.transform(filtered)
	result = loaded_model.predict(filtered)

	if result[0] == 1:
		print('Positive Review')
	elif result[0] == 0:
		print('Negative Review')

if __name__ == "__main__":
	main()
