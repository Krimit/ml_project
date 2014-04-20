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

np.set_printoptions(threshold=np.nan)

id=0
tweet=1
state=2
location=3
s1=4
s2=5
s3=6
s4=7
s5=8
w1=9
w2=10
w3=11
w4=12
k1=13
k2=14
k3=15
k4=16
k5=17
k6=18
k7=19
k8=20
k9=21
k10=22
k11=23
k12=24
k13=25
k14=26
k15=27

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
	np_array = np.delete(np_array,0,1)
	
	data = np_array[:,0:3]
	labels = np_array[:,3:]
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
def buildCorpus(data, cutoff=30):



	print 'building corpus with threshold = ', cutoff

	corpus = {}
	tweets = []
	locations = []
	locations1 = []
	locations2 = []
	
	for line in data:
		#print line[0]
		#print line[1]
		#print line[2]


		#NOTE: currently location data NOT used in building corpus
		loc1 = cleanNans(line[1])
		locations1.append(loc1)

		loc2 = cleanNans(line[2])
		locations2.append(loc2)		

		l = cleanNans(line[0]).split()
		x = list()
		for word in l:
			word = word.lower()
			to_token=filter(is_not_punct,word)
			#print word , ' ', to_token
			if not to_token in x:
				x.append(to_token)
			
		for word in x:	
			if word in corpus:
				corpus[word] += 1
			else:
				corpus[word] = 1	
		tweets.append(x)


	# remove words without enough occurrences
	updated_corpus = corpus.copy()
	for word in corpus:
		if corpus[word] < cutoff:
			del updated_corpus[word]
	corpus = updated_corpus
	corpus = corpus.keys()

	return corpus, tweets, zip(locations1, locations2)		

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

#
#
#
def buildFeatureVectors(corpus, orig_data):
	global _negative_words
	global _positive_words

	#build feature vectors
	length = len(corpus)
	height = len(orig_data)

	
	row = 0
	data = np.zeros( (height,length) )
	sentiment = np.zeros( (height,2) )
	positive_counter = 0
	negative_counter = 0
	for line in orig_data:
		positive_counter = 0
		negative_counter = 0
		col = 0
		for word in line:
			if word in corpus:
				data[row][corpus.index(word)] = 1
			if word in _negative_words:
				negative_counter += 1
			if word in _positive_words:
				positive_counter += 1		
			col += 1	
		sentiment[row][0] = positive_counter/len(line)
		sentiment[row][1] = negative_counter/len(line)
		row += 1
		print row

	#print sentiment
	return data, sentiment			


if __name__ == "__main__":
    import sys
    data, labels = readData(str(sys.argv[1]))
    corpus, orig_data, locations = buildCorpus(data)
    
    data, sentiment = buildFeatureVectors(corpus, orig_data)

