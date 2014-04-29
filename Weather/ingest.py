#!/usr/bin/env python
#
# NYU Machine Learning, Spring 2014
# Authors: Asher Krim and Bryan Ball
#
#

import os.path
import math
import operator

import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
import string

#np.set_printoptions(threshold=np.nan)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.lancaster import LancasterStemmer
from scipy import sparse


_corpus = None
_stateCorpus = None
_positive_words = []
_negative_words = []


#
# This function reads in the data file
# and aggregates it in some to-be-determined fashion.
def readData(filename):

	global _negative_words
	global _positive_words

	#open sentiment files and store in global lists
	f = 'negative.txt'
	with open(f, 'r') as d:
		_negative_words = d.read()

	f = 'positive.txt'
	with open(f, 'r') as d:
		_positive_words = d.read()	

	

	tweets = []
	locations = []
	prediction = []

	np_array = pandas.io.parsers.read_csv(filename).as_matrix()
	#np_array = np.delete(np_array,0,1)
	
	data = np_array[:,0:4]
	labels = np_array[:,4:]
	# print np_array[0]
	# print data[0]
	# print labels[0]
	# print np_array.shape
	# print data.shape
	# print labels.shape
	return data, labels

#
# build the corpus given an array with words. A
# cutoff can optionally be specified which specifies
# the minimum number of times a given word needs
# to appear in the data in order to be included
# in the corpus.
#
# Note that location data is included in corpus.
#
def buildCorpus(data, cutoff=30, build=True):

	corpus = {}
	tweets = []
	locations = []
	locations1 = []
	locations2 = []
	ids = []

	#print 'building corpus with threshold = ', cutoff
	if not build:
		#print 'not building, reading from file'
		with open('corpus.txt', 'r') as f:
			corpus = f.read().split('\n')

	
	
	for line in data:
		#print line[0]
		#print line[1]
		#print line[2]

		loc1 = cleanNans(line[2])
		locations1.append(loc1)

		loc2 = cleanNans(line[3])
		locations2.append(loc2)		
		ids.append(line[0])

		l = cleanNans(line[1]).split()
		x = list()
		for word in l:
			word = word.lower()
			to_token=filter(is_not_punct,word)
			#print word , ' ', to_token
			if not to_token in x:
				x.append(to_token)
		tweets.append(x)
		
		if build:	
			for word in x:	
				if word in corpus:
					corpus[word] += 1
				else:
					corpus[word] = 1	
	
	if build:	
	# remove words without enough occurrences
		updated_corpus = corpus.copy()
		for word in corpus:
			if corpus[word] < cutoff:
				del updated_corpus[word]
		corpus = updated_corpus
		corpus = corpus.keys()

		with open('corpus.txt', 'w') as f:
			for line in corpus:
				print>>f,line   
	#print 'done with corpus building'
	#location1 = state, location2 = city/state/location
	return corpus, tweets, zip(locations1, locations2), np.array(ids)		

def noCorpus(data):

	tweets = []
	locations = []
	locations1 = []
	locations2 = []
	ids = []

	
	for line in data:
		#print line[0]
		#print line[1]
		#print line[2]

		loc1 = cleanNans(line[2])
		locations1.append(loc1)

		loc2 = cleanNans(line[3])
		locations2.append(loc2)		
		ids.append(line[0])

		l = cleanNans(line[1]).split()
		x = list()
		for word in l:
			word = word.lower()
			to_token=filter(is_not_punct,word)
			#print word , ' ', to_token
			if not to_token in x:
				x.append(to_token)
		tweets.append(x)
		
	#location1 = state, location2 = city/state/location
	return tweets, zip(locations1, locations2), np.array(ids)	

def cleanNans(d):
	if type(d) is float and math.isnan(d):
		return ''
	return d	

# Function to use for filtering
def is_not_punct(char):
	# keep dashes, all other punctuation gets removed
	if char == '-': return True
	elif char in string.punctuation: return False
	else: return True

def buildStateCorpus(states):
	seenStates = list()
	for state in states:
		if not state[0] in seenStates:
			seenStates.append(state[0])
	return seenStates	

def buildStateFeatures(stateCorpus, locations):
	statesFeatures = np.zeros( (len(locations),len(stateCorpus)) )
	row = 0
	for s in locations:
		if s[0] in stateCorpus:
			statesFeatures[row][stateCorpus.index(s[0])] = 1
		row += 1
	
	return sparse.csr_matrix( statesFeatures )		

#
#
#
def buildFeatureVectors(corpus, orig_data):
	global _negative_words
	global _positive_words

	#build feature vectors
	length = len(corpus)
	height = len(orig_data)
	#print length, ' ', height
	#print 'building feature vect'
	
	row = 0
	data = np.zeros( (height,length) )
	#print 'data: ', data, 'shape: ', data.shape 
	sentiment = np.zeros( (height,2) )
	
	positive_counter = 0
	negative_counter = 0
	for line in orig_data:
		positive_counter = 0
		negative_counter = 0
		col = 0
		for word in line:
			if word in corpus:
				#print 'adding to data'
				data[row,corpus.index(word)] += 1
			if word in _negative_words:
				negative_counter += 1
			if word in _positive_words:
				positive_counter += 1			
			col += 1		
		sentiment[row,0] = float(positive_counter)/len(line)
		sentiment[row,1] = float(negative_counter)/len(line)
		row += 1
		#print row

	#print sentiment
	return data, sentiment			

def writeDocMatrixToFile(dataMatrix,labels,jobType='train'):

	filename = 'documentMatrix_'+jobType+'.txt'
	#with open(filename, 'w') as f:
	#	for line in datamatrix:
	#		print>>f,line   
	np.savetxt(filename, dataMatrix, newline=" ")
	filename = 'documentLabels_'+jobType+'.txt'
	np.savetxt(filename, labels, newline=" ")


#
# utility function for getting the feature matrix.
# if dataCategory='train' a corpus is built from words in filename.
# otherwise if dataCategory='test' a corpus is not built.
# returns a datamatrix as a 2-d numpy array and a 2-d matrix
# of labels.
#
# Returned: id array, data array, labels array
#
def getFeatureMatrix( filename,dataCategory='train'):
	global _corpus
	data, labels = readData(filename)
	labels = np.array(labels)
	
	if dataCategory == 'train':
		print 'training'
    	corpus, orig_data, locations, ids = buildCorpus(data,build=False)
    	_corpus = corpus
    	stateCorpus = buildStateCorpus(locations)
    	_stateCorpus = stateCorpus
    
	if dataCategory == 'test':
		print 'testing'
		corpus = _corpus
		stateCorpus = _stateCorpus

	stateFeatures = buildStateFeatures(stateCorpus, locations)
	#print 'got states'
	#print stateFeatures.shape
	#print len(orig_data)
	#print stateFeatures
	data, sentiment = buildFeatureVectors(corpus, orig_data)
	#print data
	#print sentiment
	#print sentiment
	#print stateFeatures.shape
	#print sentiment.shape
	#print data.shape
	#print ids.shape
	#print labels.shape
	result = np.concatenate((data,stateFeatures,sentiment),axis=1)
	#print 'result: ',result.shape
	return ids, result, labels

def getFeatureMatrix2( trainFilename, testFilename):
	global _corpus
	global _negative_words
	global _positive_words

	#open sentiment files and store in global lists
	f = 'negative.txt'
	with open(f, 'r') as d:
		_negative_words = d.read()

	f = 'positive.txt'
	with open(f, 'r') as d:
		_positive_words = d.read()	

	train = pandas.io.parsers.read_csv(trainFilename,header=None).as_matrix()
	#np_array = np.delete(np_array,0,1)
	
	train_data = train[:,0:4]
	train_tweets = train[:,1]
	train_labels = train[:,4:]
	print "done reading train"
	#np.save('train_labels', train_labels)

	test = pandas.io.parsers.read_csv(testFilename,header=None).as_matrix()
	#np_array = np.delete(np_array,0,1)
	
	test_data = test[:,0:4]
	test_tweets = test[:,1]
	test_labels = test[:,4:]
	print "done reading test"
	#np.save('test_labels', test_labels)

	#combine strings in orig_data
	# tweets = []
	# for tweet in orig_data:
	# 	t = ' '.join(tweet)
	# 	tweets.append(t)

	# _negative_words = stemPosNegWords(_negative_words)
	# #print _negative_words
	# print "done stemming neg"
	# _positive_words = stemPosNegWords(_positive_words)
	# print "done stemming pos"
	steemedTweets = stemIt(train_tweets)
	print "done stemming tweets"
	steemedTestTweets = stemIt(test_tweets)
	print "done stemming tweets"

	#sentiments = buildSentiments(steemedTestTweets)
	#np.save('test_sentiments', sentiments)

	# sentiments = buildSentiments(steemedTweets)
	# np.save('train_sentiments', sentiments)
	# print sentiments

	# print train_tweets[0]
	
	# print steemedTweets[0]

	#state_corpus = buildStateCorpus(train[:,2])

	#states = buildStateFeatures(state_corpus,test[:,2])
	#np.save('test_statesFeatures', states)

	#
	#
	#
	#vectorizer stuff:
	# vectorizer = CountVectorizer()
	# #X = vectorizer.fit_transform(train_tweets) # or for stemmed: 
	# X = vectorizer.fit_transform(steemedTweets)
	# np.save('train_stemmedDocMatrix', X)
	# vectorizer_test = CountVectorizer(vocabulary=vectorizer.vocabulary_)
	# X_test = vectorizer_test.fit_transform(steemedTestTweets)
	# np.save('test_stemmedDocMatrix', X_test)
	# #print X_test
	#print corpus
	#print tweets
	#print X[0,:]

	#
	#
	#
	#vectorizer with TFIDF stuff:
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(train_tweets) # or for stemmed: 
	#X = vectorizer.fit_transform(steemedTweets)
	np.save('train_regTFIDFMatrix', X)
	vectorizer_test = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
	X_test = vectorizer_test.fit_transform(test_tweets)
	np.save('test_regTFIDFMatrix', X_test)
	#print X_test
	#print corpus
	#print tweets
	print X[0,:]

def stemPosNegWords(data):
	tweets = []

	st = LancasterStemmer()
	#print data
	for line in data.split():
		x = list()
		#print line
		for word in line.split():
			#print word
			word = word.lower()
			to_token=filter(is_not_punct,word)
			steemed = st.stem(to_token)
			#print to_token, ' -> ', steemed 
			
			x.append(steemed)
		tweets.append(' '.join(x))
	
	return tweets	

def stemIt(data):

	tweets = []

	st = LancasterStemmer()

	for line in data:
		x = list()
		#print line
		for word in line.split():
			#print word
			word = word.lower()
			to_token=filter(is_not_punct,word)
			steemed = st.stem(to_token)
			#print to_token, ' -> ', steemed 
			#return
			x.append(steemed)
		tweets.append(' '.join(x))
	
	return tweets	

def buildSentiments(orig_data):
	global _negative_words
	global _positive_words

	#build feature vectors
	
	height = len(orig_data)
	#print length, ' ', height
	print 'building feature vect'

	
	row = 0
	sentiment = np.zeros( (height,2) )
	
	positive_counter = 0
	negative_counter = 0
	for line in orig_data:
		#print line
		positive_counter = 0
		negative_counter = 0
		col = 0
		for word in line.split():
			#print word
			if word in _negative_words:
				negative_counter += 1
			if word in _positive_words:
				positive_counter += 1			
			col += 1		
		sentiment[row,0] = float(positive_counter)/len(line)
		sentiment[row,1] = float(negative_counter)/len(line)
		row += 1
		#print row

	#print sentiment
	return sentiment

if __name__ == "__main__":
    import sys
    jobType = 'train'
    
    getFeatureMatrix2(str(sys.argv[1]),str(sys.argv[2]))
    #writeDocMatrixToFile(dataMatrix,labels,'train')
    #ids, dataMatrix, labels = getFeatureMatrix(str(sys.argv[2]),'test')
    #writeDocMatrixToFile(dataMatrix,labels,'test')
    