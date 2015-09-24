'''
Created on Sep 24, 2015

@author: sub253
'''
import csv
def createDict(wordScoresFile):
    d={}
    readfile = open(wordScoresFile, "r")
    reader=csv.reader(readfile,delimiter="\t")
    for word, score in reader:
        d[word.lower()] = float(score)
    readfile.close()
    return d


def scorer(sentences, wordScoreDict):
    scoreList=[]
    for sentence in sentences:
        words=sentence.split(" ")
        score=0.0
        for word in words:
            if word.lower() in wordScoreDict:
                score+=wordScoreDict[word.lower()]
        
        score=score/len(words)
        scoreList.append(score)
    
    return scoreList

def factContent(sentences, wordScoreDict):
    factList=[]
    for sentence in sentences:
        fact=''
        words=sentence.split(" ")
        for word in words:
            if word in wordScoreDict:
                fact=fact + ' '+ word.lower()
        
        factList.append(fact.strip())
    
    return factList