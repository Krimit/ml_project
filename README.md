# used these websites for most of code for this part
# http://www.andrewbenjaminhall.com/wp-content/uploads/2011/12/Process_Text.txt
# http://www.andrewbenjaminhall.com/wp-content/uploads/2011/12/Gen_TDM.txt

# create term doc matrix
import nltk
from nltk.corpus import stopwords
import re
import pickle
import os
import string
from __future__ import division

# Function to use for filtering
def is_not_punct(char):
	# keep dashes, all other punctuation gets removed
	if char == '-': return True
	elif char in string.punctuation: return False
	else: return True


# import reviews and make them a list
t=open('C:\\Users\\Bryan\\Documents\\school\\statistics for social data\\project\\text.txt','r')
textList=list(t)
t.close()

fd_list=[] # list of each review's freq dist
stpwords=stopwords.words('english') # stopwords are uninteresting common words (e.g. the)

for t in textList:
	text=t.lower()	
	to_token=filter(is_not_punct,text) # remove punctuation
	tokens=nltk.word_tokenize(to_token) # turns string into list of words
	doc=[word for word in tokens if word not in stpwords]
	fd_list.append(nltk.FreqDist(doc)) #creating list of frequency dists


num_docs=len(fd_list)
LOWER_FREQ_BOUND=.01 # words with appearances/doc below this are excluded

words=[key for dict in fd_list for key in dict.keys()] # all words
total_fd=nltk.FreqDist(words)
words_to_remove=[]
for key in total_fd.keys():
	freq=total_fd[key]/num_docs
	if freq<LOWER_FREQ_BOUND:
		words_to_remove.append(key)

goodwords=set(words).difference(set(words_to_remove)) # take set of all words and remove infrequent words


# make term doc mtx
matrix=[]
for dict in fd_list:
    count = []
    for word in goodwords:
        if word in dict.keys(): count.append(dict[word])
        else: count.append(0)
    matrix.append(count)


# write mtx to file
matrix_file=open('C:\\Users\\Bryan\\Documents\\school\\statistics for social data\\project\\tdm.txt','w')
matrix_file.write('Document'+','+','.join(goodwords)+'\n')
count=1
for doc in matrix:
	matrix_file.write(str(count) + ',' +  ','.join([str(i) for i in doc]) + '\n')
	count+=1


matrix_file.close()
