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

	ys = [] #the true label
	data = []
	print filename
	filename = PATH + filename
	with open(filename, 'r') as data:
		for basic_line in data:
			try:

				line = basic_line.split('\t')

				print line
				return
			except:
				pass
					
			


if __name__ == "__main__":
    import sys
    read_gdelt_file(str(sys.argv[1]))
