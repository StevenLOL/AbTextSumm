# -*- coding: utf-8 -*-
'''
Created on Aug 21, 2015

@author: siddban
'''
import WGGraph as wg
from nltk.tag.stanford import POSTagger
import os
import re, sys
import kenlm, codecs
from sentenceRanker import createDict

reload(sys)  
sys.setdefaultencoding('utf8')


PROJECT_DIR=os.path.dirname(__file__)+"/../../"
print "Project dir", PROJECT_DIR
RESOURCES_DIR=PROJECT_DIR+"resources/"

mainDatafolder=RESOURCES_DIR+"Summarization/"
### THE actual work happens here#####
#english_postagger = POSTagger(RESOURCES_DIR+'jars/english-left3words-distsim.tagger',RESOURCES_DIR+'jars/stanford-postagger.jar', encoding='utf-8')
#langModel=ARPALanguageModel("resources/lm_giga_20k_nvp_3gram.arpa")
stopwords=wg.load_stopwords(RESOURCES_DIR+"resources/stopwords.en.dat")  
lm = kenlm.LanguageModel(RESOURCES_DIR+'resources/lm-3g.klm')

folder_mode = {'Extract':'AIDR_Extract', 'Original':'AIDR_Original'}
rankingModes={"C":"Centroid","TR":"textrank", "CW":"contentWeighing"}


def sentenceCapitalize(sent):
    sentences = sent.split(". ")
    sentences2 = [sentence[0].capitalize() + sentence[1:] for sentence in sentences]
    string2 = '. '.join(sentences2)
    return string2

def tweetCleaner(tweets):
    p=re.compile(r'http?:\/\/.*[\s\r\n]*', re.DOTALL) #Regex to remove http from tweets
    p2=re.compile(r'(^|\s)#.+?\s', re.DOTALL) #Regex
    p3=re.compile(r'(^|\s)@.+?(\s|$)', re.DOTALL) 
    #p4=re.compile(r'\u003F', re.DOTALL)
    #p5=re.compile(r'?', re.DOTALL) 
   
    final_tweets=[]
    for text in tweets:
        #text=text.decode("utf-8","ignore")

        text=text.strip()
        text=text.replace(' ./,',' ./PUNCT')
        text=re.sub(p,' ', text)
        text=re.sub(p2, ' ', text)
        text=re.sub(p3, ' ' , text)
        #text=re.sub(p4, '', text)
        #text=re.sub(p5, '', text)
        text=text.strip().split('./PUNCT')
        #print text
        for r in text:
            if len(r.strip())!=0:
                #r=r.replace('/,','/PUNCT')
                r=re.sub( '\s+', ' ', r ).strip()
                final_tweets.append(r.strip()+' ./PUNCT')
    
    final_tweets=set(final_tweets) 
    #final_tweets=[final_tweets]
    return final_tweets

def getClasses(folder):
    return [f for f in os.listdir(folder) if os.path.isdir(folder+"/"+f)]
    

'''
This method simply takes in a list of POS tagged sentences and produces a summary
Inputs: Sentences (tweets pos tagged), ranking strategy
'''
def summaryGenerator(class_name, tweets, folder_mode, ranker):
    tweets=tweetCleaner(tweets) #Some cleaning
    #print "Set of Tweets=>", len(tweets)
    #tweetlist=[tweet for tweet in tweets]
    #print "List of tweets", tweetlist
    genSentences=wg.retrieveNewSentences(tweets, stopwords)
    wordScores=createDict(mainDatafolder+"/"+folder_mode['Extract']+"/"+class_name+"/"+class_name+"_weight.txt")

    #emptysentences=[sent for sent in genSentences if len(sent.strip())==0]
    #print "EMPTY::::", len(emptysentences)
    '''
    This is where the ILP works to select the best sentences and form the summary
    '''
    finalSentencesRetained=wg.solveILP(genSentences,wordScores,
                                            lm, 
                                            stopwords, 
                                            ranker,
                                            intraGenSimThreshold=0.25, 
                                            l_max=200
                                            )
    
    return finalSentencesRetained

def txtFromSents(finalsummarySents):
    txtSummary=""
    if finalsummarySents is None:
        return txtSummary
    for sent in finalsummarySents:
        sent=sentenceCapitalize(sent)+"."
        sent=sent.replace(":.",".")
        txtSummary=txtSummary+"\n"+sent
    #print 'Number of tweets', len(tweets) , tweets 
        txtSummary=txtSummary.strip()
    return txtSummary
#print "List of Classes:", list_of_classes

RESULTS_DIR=RESOURCES_DIR+"abstractivesummaries"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

import gc
gc.enable()

def generateSummaries(mode = folder_mode['Extract'] #Original: Replace 'Extract' with 'Original'
    ,ranker = rankingModes['C']):
    '''
    Settings: What mode? Extracts or Entire Text
    If extract, directly take tweets
    if entire text, take the 5th column from the files
    '''
    print 'Using mode --> ', mode , ' and Ranker -->', ranker
    list_of_classes=getClasses(mainDatafolder+folder_mode['Extract'])
    
    for classname in list_of_classes:
        gc.collect()
        print 'Running class, ', classname
        if classname not in 'infrastructure_damage':
            continue
        #
        #
        finalsummarySents=[]
        
        if mode in 'AIDR_Extract':
            writefile=open(RESULTS_DIR+"/"+classname+"."+mode+"."+ranker+".txt","w")#,"utf-8", errors="ignore")
            f= open(mainDatafolder+"/"+mode+"/"+classname+"/"+classname+".txt","r")
            tweets=f.readlines()
            f.close()
            finalsummarySents=summaryGenerator(classname, tweets, folder_mode, ranker)
            summary=txtFromSents(finalsummarySents)
            writefile.write(summary)
            writefile.close()
        else:
            writefile=open(RESULTS_DIR+"/"+classname+"."+mode+"."+ranker+".txt","w")#,"utf-8", errors="ignore")
            f= open(mainDatafolder+"/"+mode+"/"+classname+".txt","r")
            lines=f.readlines()
            f.close()
            #print "length", len(lines)
            tweets=[]
            for line in lines:
                line=line.strip()
                if len(line) <= 10:
                    continue
                tweet=line.split('\t')[3]
                #print len(tweet.split(" ")), tweet
                tweets.append(tweet)
            #print "length", len(tweets)
            finalsummarySents=summaryGenerator(classname, tweets, folder_mode, ranker)
            summary=txtFromSents(finalsummarySents)
            writefile.write(summary)
            writefile.close()
        
'''
folder_mode = {'Extract':'AIDR_Extract', 'Original':'AIDR_Original'}
rankingModes={"C":"Centroid","TR":"textrank", "CW":"contentWeighing"}
'''


if __name__ == "__main__":
    #generateSummaries(mode = folder_mode['Extract'], ranker = rankingModes['C'])
    #generateSummaries(mode = folder_mode['Extract'], ranker = rankingModes['CW'])
    #generateSummaries(mode = folder_mode['Extract'], ranker = rankingModes['TR'])
    generateSummaries(mode = folder_mode['Original'], ranker = rankingModes['C'])
    #generateSummaries(mode = folder_mode['Original'], ranker = rankingModes['CW'])
    #generateSummaries(mode = folder_mode['Original'], ranker = rankingModes['TR'])
    #if 

#with open(mainDatafolder+"/"+folder_mode['Extract']+"/"+class_name+"/"+class_name+".txt"),"r") as f:
#    tweets=f.readlines()
#    summaryGenerator(tweets, folder_mode, rankingModes)

