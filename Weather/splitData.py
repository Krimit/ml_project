import random

#
# Split the original training data file into
# our own test and train files.
# The files should already be included in folder
# so there should be no need to run this again.
#

with open("train.csv", "r") as f:
    data = f.read().split('\n')

data.pop(0)
#print data[0]
data = filter(None, data)
#data = [line for line in data.split('\n') if data.strip() != '']
random.shuffle(data)

n = 15580
train_data = data[n:]
test_data = data[:n]

with open('ourTrain.csv', 'w') as f:
	for line in train_data:
		print>>f,line    

with open('ourTest.csv', 'w') as f:
	for line in test_data:
		print>>f,line    
