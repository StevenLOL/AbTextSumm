#coding: utf-8
"""
Centroid-based Summarization 
[Jurafsky & Martin, 2nd ed, ch23, sec23.4.1]
Author: Marcelo Criscuolo (criscuolo[dot]marcelo[at]gmail[dot]com)
Date: 2013-07-19
"""
from __future__ import print_function

import glob
import re
import sys
import math
from collections import Counter


def segmentize(text):
    sentences = []
    begin = 0
    for match in re.finditer(r'([^\s.]*)\s*(\.+)\s*([^\s.]?)', text):
        prefix = match.group(1)
        suffix = match.group(3)
        if is_eos(prefix, suffix, match.group(0)):
            end = match.start(3)
            sentence = text[begin:end].strip()
            begin = end
            sentence=re.sub('\s+', ' ', sentence).strip()
            sentences.append(sentence)
    return [s for s in sentences if len(s) > 1] # FIXME len(s) > 1: SUCH A SHAME!!!

def is_eos(prefix, suffix, region):
    # domains?
    if re.search(r'\S\.\S', region):
        return False
    # numbers?
    if len(suffix) > 0 and suffix[0].isdigit():
        return False
    # abbreviations?
    if len(prefix) in range(1, 5) and prefix[0].isupper():
        return False
    return True


def load_text(filename):
    with open(filename) as fin:
        text = fin.read()
    start = text.find('title:') + len('title:')
    finish = text.find('\n', start)
    title = text[start:finish].strip()
    text = text[finish:]
    return (title, text)

def words(text):
    return [w.lower() for w in re.findall(r'[^\s.:,;!()?/\'"]+', text)]

def compute_idf(sentences):
    idf = {}
    filecount = 0.0
    for sent in sentences:
        filecount += 1
        for w in set(words(sent)):
            idf[w] = idf.get(w, 0) + 1
    for w in idf.keys():
        idf[w] = math.log(filecount / idf[w])
    return idf

def compute_tf(text):
    counter = Counter(words(text))
    total_words = sum(c for w, c in counter.items())
    tf = dict((w, float(c)/total_words) for w, c in counter.items())
    return tf

def cosinesim(idf, text1, text2):
    tf1 = compute_tf(text1)
    words1 = [w for w in tf1.keys()]
    tf2 = compute_tf(text2)
    words2 = [w for w in tf2.keys()]
    dotprod = sum(tf1.get(w, 0) * tf2.get(w, 0) * (idf[w] ** 2) for w in
            set(words1 + words2))
    norm1 = math.sqrt(sum((tf1[w] * idf[w])**2 for w in words1))
    norm2 = math.sqrt(sum((tf2[w] * idf[w])**2 for w in words2))
    return dotprod / (norm1 * norm2)

def rank(idf, sentences):
    scount = len(sentences)
    sim_matrix = [[0] * scount for i in range(scount)]
    for i in range(scount-1):
        for j in range(i+1, scount):
            sim_matrix[i][j] = cosinesim(idf, sentences[i], sentences[j])
            sim_matrix[j][i] = sim_matrix[i][j]
    ranking = []
    for spos in range(scount):
        avg_sim = sum(sim_matrix[spos]) / scount
        ranking.append(avg_sim)
    #ranking.sort(key=lambda t: t[1], reverse=True)
    return ranking
    

def show_tf():
    filename = sys.argv[1]
    idf = compute_idf()
    title, text = load_text(filename)
    tf = compute_tf(text)
    for w, c in tf.items():
        print(c * idf.get(w, 0), w)

def main():
    filename = sys.argv[1]
    idf = compute_idf()
    title, text = load_text(filename)
    sentences = segmentize(text)
    ranking = rank(idf, sentences)
    relevant = [s for s, r in ranking]
    relevant.sort()
    # the order the sentences occur originally
    print()
    print('==', title, '==')
    print()
    for s in relevant:
        print(sentences[s])
    print()
    print()

def getScores(sentences):
    idf = compute_idf(sentences)
    ranking = rank(idf, sentences)
    return ranking
    
if __name__ == '__main__':
    main()