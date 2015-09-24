#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 02-Aug-2015

@author: siddban
'''

import WGGraph
import simplejson as json
from nltk.tag.stanford import POSTagger
import os, gzip
import re
#from pynlpl.lm.lm import ARPALanguageModel
import kenlm
#os.environ['JAVA_HOME'] ='C:/jdk1.7.0_07/bin'  ##Laptop\
#os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk1.7.0_17/bin' ##Lab desktop
# mypath = 'Output_clusters/'
# listing = os.listdir(mypath)
# #print listing
#     #print("current file is: " + infile)
# for files in listing:

PROJECT_DIR=os.path.dirname(__file__)+"/../../"
print "Project dir", PROJECT_DIR
RESOURCES_DIR=PROJECT_DIR+"/"+"resources/"
### THE actual work happens here#####
def sentenceCapitalize(sent):
    sentences = sent.split(". ")
    sentences2 = [sentence[0].capitalize() + sentence[1:] for sentence in sentences]
    string2 = '. '.join(sentences2)
    return string2




english_postagger = POSTagger(RESOURCES_DIR+'jars/english-left3words-distsim.tagger',RESOURCES_DIR+'jars/stanford-postagger.jar', encoding='utf-8')
#langModel=ARPALanguageModel("resources/lm_giga_20k_nvp_3gram.arpa")

stopwords=WGGraph.load_stopwords(RESOURCES_DIR+"resources/stopwords.en.dat")  
lm = kenlm.LanguageModel(RESOURCES_DIR+'resources/lm-3g.klm')

#numClusters=[25,30,35,40,45,50]
numClusters=[1]
allEvents=os.listdir(RESOURCES_DIR+"old_Clusters/"+"Cluster_Data")
absdir="abstracts"

if not os.path.exists(absdir):
    os.makedirs(absdir)
for eventfile in allEvents:
    #if eventfile !="kuwait_number_Cluster.txt":
    #    continue
    
    if os.path.isdir(RESOURCES_DIR+"old_Clusters/"+"Cluster_Data/"+eventfile):
        continue
    #if eventfile not in ("kuwait"):
    #    continue
    print "Reading folder", eventfile
    writeAbstracts=open(absdir+"/"+eventfile+".txt","w")
    #clusFiles = os.listfiles("Cluster_Data/"+eventfile)
    #files = filter(os.path.isfile, os.listdir("Cluster_Data/") )
    for i in numClusters:
        print "Cluster:", i
        #fp=gzip.open("Cluster_Data/"+eventfile+"/"+eventfile+"_"+str(i)+".txt.gz")
        fp=gzip.open(RESOURCES_DIR+"old_Clusters/"+"Cluster_Data/"+eventfile)#+"_Cluster.txt.gz")
        print "Reading file....", fp
        dic = json.loads(fp.readline())
        gengroupList=[]
        clustNum=0
        origSentences=[]
        for k,v in dic.iteritems():
            #print(k)
            #print v
            clustNum+=1
            #if clustNum == 5:
            #    break
            tweets=[]
            for elem in v:
            #v=set(v)
                timestamp, text = elem
                #print 'text ==>',text
                p=re.compile(r'http.+?\s', re.DOTALL)
                #print 'B:', text
                text=text.replace(' ./,',' ./PUNCT')
                #print 'A:', text
                #for text in v:
                origSentences.append(text)
                text=re.sub(p, '', text)
                text=text.strip().split('./PUNCT')
                
                for r in text:
                    if len(r.strip())!=0:
                        tweets.append(r.strip()+' ./PUNCT')
            tweets=set(tweets)
            
            #print len(tweets), tweets
            genSentences=WGGraph.retrieveNewSentences(tweets, english_postagger, stopwords)
            gengroupList.append(genSentences)
            print "Done with ", clustNum
        print gengroupList
        print 'Num of clusters', len(gengroupList)    
        finalSentencesRetained=WGGraph.solveILP(gengroupList,lm, stopwords, origSentences, intraGenSimThreshold=0.2, l_max=10)
 
        txtSummary=""
        for sent in finalSentencesRetained:
            sent=sentenceCapitalize(sent)
                
            txtSummary=txtSummary+"\n"+sent
    #print 'Number of tweets', len(tweets) , tweets 
        txtSummary=txtSummary.strip()
        writeAbstracts.write("==========="+eventfile+"~"+str(i)+":========\n"+txtSummary+"\n\n")
    writeAbstracts.close()   



