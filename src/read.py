import sys
import json
import gzip
import re
#hagupit_number_Cluster.txt
fp = open("hagupit_number_Cluster.txt",'rb')
dic = json.loads(fp.readline())
count = 0
wf=open('write_clusters.txt','w')
for k,v in dic.iteritems():
	#print(k)
	v=set(v)
	p=re.compile(r'http.+?\s', re.DOTALL)
	tweets=[]
	for text in v:
		tweets.append(re.sub(p, '', text))
	tweets=set(tweets)
	if len(tweets) <= 5 :
		continue
	for sentence in tweets:
		wf.write(k+ "\t"+sentence+"\n")
	#count+=1
	#if count==5:
	#	break
fp.close()
wf.close()
print(len(dic.keys()))



