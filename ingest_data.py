#!/usr/bin/env python
#
# NYU Machine Learning, Spring 2014
# Authors: Asher Krim and Bryan Ball
#
#

from numpy import *
import os.path

import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas

PATH = 'data/'

# Code for donwloading all files automatically:
# blog post:
# http://johnbeieler.org/blog/2013/06/01/making-gdelt-downloads-easy/
# github:
# https://github.com/johnb30/gdelt_download

#
# This function reads in the given gdelt format file
# and aggregates it in some to be determined fashion.
def read_gdelt_file(filename):

	aggreg = {}
	headers = []
	header_file = PATH + 'CSV.header.historical.txt'
	with open(header_file, 'r') as data:
		for basic_line in data:
			line = basic_line.split()
			for heading in line:
				aggreg[heading] = []
				headers.append(heading)
				
	#print aggreg

		
	
	filename = PATH + filename
	with open(filename, 'r') as data:
		for basic_line in data:
			try:

				line = basic_line.split('\t')
				print line
				for index, item in enumerate(line):
					#print item, index
					#print headers[index]
					aggreg[headers[index]].extend(item)

				#print aggreg	


				
				return
			except:
				pass

			


if __name__ == "__main__":
    import sys
    read_gdelt_file(str(sys.argv[1]))
