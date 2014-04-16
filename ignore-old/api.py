#kitties!
#from urllib2 import urlopen

#website = urlopen("http://placekitten.com/")
#kittens = website.read()


#print kittens[559:1000]	

############################

#api stuff
#import requests
#import pprint

#query_params = { 'apikey': 'f1f2058236884fc4bfb1baabe60e6922',
#				 'per_page': 3,
# 		   		 'phrase': 'election night',
# 		   		 'state':'MD'
# 		 }

# endpoint = 'http://capitolwords.org/api/text.json'
# response = requests.get(endpoint, params= query_params)

# data = response.json()
# pprint.pprint(data)


########################
#stubhub: need 7 digit event id. Example id: 4270639

import pprint

import urllib2

query_params = { 'eventid': 4270639
}

req = urllib2.Request('https://api.stubhub.com/search/inventory/v1?eventid=4413647')
req.add_header('Authorization', 'Bearer k2Yor8Vwx3MqhjHOlCf7700r33Ea')
req.add_header('Accept', 'application/json')
req.add_header('Accept-Encoding', 'application/json')

res = urllib2.urlopen(req).read()


pprint.pprint(res.listing)