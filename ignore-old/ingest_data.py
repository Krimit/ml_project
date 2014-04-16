#!/usr/bin/env python
#
# NYU Machine Learning, Spring 2014
# Authors: Asher Krim and Bryan Ball
#
#

from numpy import *
import os.path

import operator

import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas

PATH = 'data/'
ACTOR1Code = 5
ACTOR1CountryCode = 7
ACTOR2Code = 15
ACTOR2CountryCode = 17

EVENTCode = 26
EVENTBaseCode = 27
EVENTRootCode = 28

# Code for donwloading all files automatically:
# blog post:
# http://johnbeieler.org/blog/2013/06/01/making-gdelt-downloads-easy/
# github:
# https://github.com/johnb30/gdelt_download

#
# This function reads in the given gdelt format file
# and aggregates it in some to be determined fashion.
def read_gdelt_file(filename):

	unique_actors = dict()
	internal_to_actor = dict()
	unique_events_base = dict()
	counts = 0;


	aggreg = dict()

	
	#
	# Ingest the column headers.
	# I'm not sure if we need this. This reads in the header file.
	# Since we know the column numbers we are interested in, it might make
	# sense to hard code them, as done below.
	# For future - possibly a better idea would be to read the other header file on GDELT which
	# has the headers with column id (index). Although if we are only using a handful, probably won't be needed.
	#  - ak
	#
	headers = []
	header_file = PATH + 'CSV.header.historical.txt'
	with open(header_file, 'r') as data:
		for basic_line in data:
			line = basic_line.split()
			for heading in line:
				headers.append(heading)
				
	
	#
	# Ingest the event codes.
	#
	event_codes = dict()
	event_codes_file = PATH + 'CAMEO.eventcodes.txt'
	with open(event_codes_file, 'r') as data:
		for basic_line in data:
			basic_line = basic_line.replace('\n', '')
			line = basic_line.split('\t')
			if not event_codes.get(line[0]): 
				event_codes[line[0]] = line[1]
	

	
	#
	# Open a single daily file and count the number of events by base type.
	# Only looking at events ehich involve France.
	#
	filename = PATH + filename
	with open(filename, 'r') as data:
		for basic_line in data:
			try:

				split_line = basic_line.split('\t')
				if split_line[ACTOR1CountryCode] == 'FRA':
					counts += 1
					if not unique_actors.get(split_line[ACTOR1Code]): 
						unique_actors[split_line[ACTOR1Code]] = 1
						
					else:
						unique_actors[split_line[ACTOR1Code]] += 1

					
							
	 			elif split_line[ACTOR2CountryCode] == 'FRA':
	 				counts += 1
					if not unique_actors.get(split_line[ACTOR2Code]):
						unique_actors[split_line[ACTOR2Code]] = 1
						
					else:
						unique_actors[split_line[ACTOR2Code]] += 1

				if split_line[ACTOR1CountryCode] == 'FRA' or split_line[ACTOR2CountryCode] == 'FRA':
					# event aggreg
					if not unique_events_base.get(split_line[EVENTBaseCode]): 
						unique_events_base[split_line[EVENTBaseCode]] = 1
						
					else:
						unique_events_base[split_line[EVENTBaseCode]] += 1

				if split_line[ACTOR1CountryCode] == 'FRA' and split_line[ACTOR2CountryCode] == 'FRA':			
					if not internal_to_actor.get(split_line[EVENTBaseCode]):
						internal_to_actor[split_line[EVENTBaseCode]] = 1
						
					else:
						internal_to_actor[split_line[EVENTBaseCode]] += 1
	
			except IndexError:
				pass

	print len(unique_actors.keys()) , ' ' , unique_actors
	
	print unique_events_base
	print 'most frequent base event: ', event_codes.get(max(unique_events_base.iteritems(), key=operator.itemgetter(1))[0])
	print 'occurred this many times: ', max(unique_events_base.values())

	print 'of all events, this was most frequent internal to country: ',  event_codes.get(max(internal_to_actor.iteritems(), key=operator.itemgetter(1))[0]) 
	print 'occurred this many times: ', max(internal_to_actor.values())
			


if __name__ == "__main__":
    import sys
    read_gdelt_file(str(sys.argv[1]))
