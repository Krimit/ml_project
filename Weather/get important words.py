import numpy as np
import pandas as pd
import heapq

when_coefs=np.array(pd.io.parsers.read_csv('when ridge coefficients.csv',header=None))
sent_coefs=np.array(pd.io.parsers.read_csv('sentiment ridge coefficients.csv',header=None))
kind_coefs=np.array(pd.io.parsers.read_csv('uni kind ridge coefficients.csv',header=None))

#These functions below return the indices of the largest and smallest coefficients respectively
#from the "when" problem in row k.  Each row represents a different field for a category.
#If you go here: http://www.kaggle.com/c/crowdflower-weather-twitter/data, you will see how 
#they are organized.  So when k=0, we are looking at the first "when" field, aka w1, meaning
#current weather.  When k=1, that is w2 or future weather, and so on.
high_inds=[t[0] for t in heapq.nlargest(20, enumerate(when_coefs[k]), lambda t: t[1])]
low_inds=[t[0] for t in heapq.nsmallest(20, enumerate(when_coefs[k]), lambda t: t[1])]
