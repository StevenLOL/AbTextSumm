#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 02-Aug-2015

@author: siddban
'''

import WGGraph
import json
from nltk.tag.stanford import POSTagger
import os
import re
from pynlpl.lm.lm import ARPALanguageModel
import kenlm
#os.environ['JAVA_HOME'] ='C:/jdk1.7.0_07/bin'  ##Laptop\
#os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk1.7.0_17/bin' ##Lab desktop
# mypath = 'Output_clusters/'
# listing = os.listdir(mypath)
# #print listing
#     #print("current file is: " + infile)
# for files in listing:


### THE actual work happens here#####

english_postagger = POSTagger('jars/english-left3words-distsim.tagger','jars/stanford-postagger.jar', encoding='utf-8')
#langModel=ARPALanguageModel("resources/lm_giga_20k_nvp_3gram.arpa")

stopwords=WGGraph.load_stopwords("resources/stopwords.en.dat")  
lm = kenlm.LanguageModel('resources/lm-3g.klm')

allEvents=os.listdir("clusters")
for eventfile in allEvents:
    #if eventfile !="kuwait_number_Cluster.txt":
    #    continue
    print "Reading file", eventfile
    fp = open("clusters/"+eventfile,'rb')
    dic = json.loads(fp.readline())
    gengroupList=[]
    for k,v in dic.iteritems():
    #print(k)
        v=set(v)
        p=re.compile(r'http.+?\s', re.DOTALL)
        tweets=[]
        for text in v:
            tweets.append(re.sub(p, '', text))
        tweets=set(tweets)
        if len(tweets) <= 10:
            continue
        genSentences=WGGraph.retrieveNewSentences(tweets, english_postagger, stopwords)
        #languagemodel=[]
        gengroupList.append(genSentences)

    print 'Num of clusters', len(gengroupList)    
    finalSentencesRetained=WGGraph.solveILP(gengroupList,lm, stopwords, intraGenSimThreshold=0.5, l_max=10)
 
    txtSummary=""
    for sent in finalSentencesRetained:
        txtSummary=txtSummary+"\n"+sent
    #print 'Number of tweets', len(tweets) , tweets 
    txtSummary=txtSummary.strip()
    print "==========="+eventfile+"========\n",txtSummary,"\n\n"


