from numpy import *
import os.path

#
#
def read_gdelt(filename):

	ys = [] #the true label
	data = []
	print filename
	#filename = os.path.dirname(__file__) + '/../data/' +  filename
	with open(filename, 'r') as d:
		for line in d:
			print line
			return
			l = line.split()
			c = l.pop(0) #the class
			if c == '0':
				c = -1
			else:
				c = 1	
			ys.append(c)
			# store words in a set
			x = set()
			for word in l:
				x.add(word)	
			data.append(x)


if __name__ == "__main__":
    import sys
    read_gdelt(str(sys.argv[1]))
