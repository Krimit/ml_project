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

#
# This function reads in the data file
# and aggregates it in some to-be-determined fashion.
def readData(filename):

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
	return data

#
# build the corpus given an array with words. A
# cutoff can optionally be specified which specifies
# the minimum number of times a given word needs
# to appear in the data in order to be included
# in the corpus.
#
# Note that location data is included in corpus.
#
def buildCorpus(data, cutoff=5):
	print 'building corpus with threshold = ', cutoff

	corpus = {}
	orig_data = []

	
	for line in data:
		for d in line:
			if type(d) is float and math.isnan(d):
				d = ''
				
			l = d.split()
			x = set()
			for word in l:
				x.add(word)
			for word in x:	
				if word in corpus:
					corpus[word] += 1
				else:
					corpus[word] = 1	
			orig_data.append(x)


	# remove words without enough occurrences
	updated_corpus = corpus.copy()
	for word in corpus:
		if corpus[word] < cutoff:
			del updated_corpus[word]
	corpus = updated_corpus

	return corpus, orig_data		

def buildFeatureVectors(corpus, orig_data):
	#build feature vectors
	length = len(corpus)
	height = len(orig_data)
	
	row = 0
	data = np.zeros( (height,length) )
	for line in orig_data:
		
		col = 0
		for word in corpus:
			if word in line:
				data[row][col] = 1
			else:
				data[row][col] = 0
			col += 1	
		row += 1
		print row

	return data			


if __name__ == "__main__":
    import sys
    data = readData(str(sys.argv[1]))
    corpus, orig_data = buildCorpus(data)
    data = buildFeatureVectors(corpus, orig_data)
    print data
