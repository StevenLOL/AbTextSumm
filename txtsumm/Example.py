# -*- coding: utf-8 -*-
'''
Created on Aug 21, 2015

@author: siddban
'''
import absummarizer.WGGraph as wg
import os
import re
import nltk
from absummarizer.summarizer import segmentize

PROJECT_DIR=os.path.dirname(__file__)+"/../"
print "Project dir", PROJECT_DIR
RESOURCES_DIR=PROJECT_DIR+"resources/"
stopwords=wg.load_stopwords(RESOURCES_DIR+"stopwords.en.dat")  

rankingModes={"C":"Centroid","TR":"textrank", "CW":"contentWeighing"}

def sentenceCapitalize(sent):
    sentences = sent.split(". ")
    sentences2 = [sentence[0].capitalize() + sentence[1:] for sentence in sentences]
    string2 = '. '.join(sentences2)
    return string2

def tweetCleaner(sentences):
    p=re.compile(r'http?:\/\/.*[\s\r\n]*', re.DOTALL) #Regex to remove http from sentences
    p2=re.compile(r'(^|\s)#.+?\s', re.DOTALL) #Regex
    p3=re.compile(r'(^|\s)@.+?(\s|$)', re.DOTALL) 
    print "Initial sentences=>", len(sentences)
    final_sentences=[]
    for text in sentences:
        text=text.strip()
        text=text.replace(' ./,',' ./PUNCT')
        text=re.sub(p,' ', text)
        text=re.sub(p2, ' ', text)
        text=re.sub(p3, ' ' , text)
        text=text.strip().split('./PUNCT')
        for r in text:
            if len(r.strip())!=0:
                r=re.sub( '\s+', ' ', r ).strip()
                final_sentences.append(r.strip()+' ./PUNCT')
    
    final_sentences=set(final_sentences) 
    #print "Final sentences=>", len(final_sentences)
    return final_sentences

def getClasses(folder):
    return [f for f in os.listdir(folder) if os.path.isdir(folder+"/"+f)]
    
'''
This method simply takes in a list of POS tagged sentences and produces a summary
Inputs: Sentences (sentences pos tagged), ranking strategy
'''

def txtFromSents(finalsummarySents):
    txtSummary=""
    if finalsummarySents is None:
        return txtSummary
    for sent in finalsummarySents:
        if sent.strip().endswith("."):
            sent=sentenceCapitalize(sent)
        else:
            sent=sentenceCapitalize(sent)+"."
        sent=sent.replace(":.",".")
        txtSummary=txtSummary+"\n"+sent
        txtSummary=txtSummary.strip()
    return txtSummary

def find_bigrams(input_list):
    bigram_list=[]
    for i in range(len(input_list)-1):
        bigram_list.append((input_list[i], input_list[i+1]))
    return bigram_list

def getDates(path):
    return [foldername for foldername in os.listdir(path) if os.path.isdir(path+"/"+foldername)]

def bigramTweetGenerator(sentences):
    bigramsentences=[]
    for tweet in sentences:
        bitweet=''
        tweet=tweet.strip().lower()
        words=tweet.split(" ")
        bigramlist=find_bigrams(words)
        for bigrams in bigramlist:
            bigramword=''
            word1, word2=bigrams
            m1 = re.match("^(.+)/(.+)$", word1)
            m2 = re.match("^(.+)/(.+)$", word2)
            if m1 and m2:
                bigramword=m1.group(1)+"||"+m2.group(1)+"/"+m1.group(2)+"||"+m2.group(2)
                bitweet=bitweet+" "+bigramword
        if len(bitweet.strip())>0:
            bigramsentences.append(bitweet.strip())
    return bigramsentences


def generateSummaries(sentences, length=100, mode = "Extractive", ranker = rankingModes['TR']):
    
        
    '''
    This is where the ILP works to select the best sentences and form the summary
    '''
    if mode == "Abstractive":
        import kenlm
        lm = kenlm.LanguageModel(RESOURCES_DIR+'/lm-3g.klm')
        '''
        Here sentences should have POS tagged format
        '''
        taggedsentences=[]
        for sent in sentences: 
            sent=sent.decode('utf-8','ignore')
            tagged_sent=''
            tagged_tokens=nltk.pos_tag(nltk.word_tokenize(sent))
            for token in tagged_tokens:
                word, pos=token
                tagged_sent=tagged_sent+' '+word+"/"+pos
            taggedsentences.append(tagged_sent.strip())
            
        sentences=bigramTweetGenerator(taggedsentences)
        genSentences, svolist=wg.retrieveNewSentences(sentences, stopwords)
    
        if len(genSentences) <= 1:
            return [k for k, v in genSentences]
        finalSentencesRetained=wg.solveILPFactBased(genSentences,
                                            lm,                                             
                                            stopwords, 
                                            ranker,
                                            intraGenSimThreshold=0.5, 
                                            l_max=length,
                                            mode="Abstractive"
                                            )
    
        
        summary=txtFromSents(finalSentencesRetained)
        print "=======Summary:===== \n", summary           
    
    if mode == "Extractive":
        lm=[] #No need of language model in Extractive
        #if len(sentences) <= 2:
        #    summary=txtFromSents(sentences)
        #    print "Summary: ", summary 
        #    return 
        
        print sentences
        finalSentencesRetained=wg.solveILPFactBased(sentences,
                                            lm,                                            
                                            stopwords, 
                                            ranker,
                                            intraGenSimThreshold=0.7, 
                                            l_max=length,
                                            mode="Extractive"
                                            )
        
        print 'Final sentences,', finalSentencesRetained
        summary=txtFromSents(finalSentencesRetained)
        print "=======Summary:===== \n", summary          
    
    
'''
rankingModes={"C":"Centroid","TR":"textrank"}
mode=["Extractive","Abstractive"]
'''
if __name__ == "__main__":
    passage="As a scientific endeavour, machine learning grew out of the quest for artificial intelligence.\
    Already in the early days of AI as an academic discipline, some researchers were interested in having machines learn from data.\
    They attempted to approach the problem with various symbolic methods, as well as what were then termed \"neural networks\"; these were \
    mostly perceptrons and other models that were later found to be reinventions of the generalized linear models of statistics.\
    Probabilistic reasoning was also employed, especially in automated medical diagnosis.\
    However, an increasing emphasis on the logical, knowledge-based approach caused a rift between AI and machine learning.\
    Probabilistic systems were plagued by theoretical and practical problems of data acquisition and representation.\
    By 1980, expert systems had come to dominate AI, and statistics was out of favor.[11] Work on symbolic/knowledge-based learning did\
    continue within AI, leading to inductive logic programming, but the more statistical line of research was now outside the field of AI proper,\
    in pattern recognition and information retrieval.[10]:708–710; 755 Neural networks research had been abandoned by AI and computer science\
    around the same time. This line, too, was continued outside the AI/CS field, as \"connectionism\",\
    by researchers from other disciplines including Hopfield, Rumelhart and Hinton.\
    Their main success came in the mid-1980s with the reinvention of backpropagation. Machine learning, reorganized as a separate field,\
    started to flourish in the 1990s. The field changed its goal from achieving artificial intelligence\
    to tackling solvable problems of a practical nature. It shifted focus away from the symbolic approaches it had inherited\
    from AI, and toward methods and models borrowed from statistics and probability theory."
    
    list_Sentences=segmentize(passage)
    generateSummaries(list_Sentences, mode="Extractive")
    
    
    
    
    
  