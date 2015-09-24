#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
import re, igraph
from pattern.text import parsetree 
import numpy as np
import networkx as nx
from random import shuffle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from pulp import *#LpAffineExpression, LpConstraint, LpConstraintVar, lpSum
import Stemmer
from sklearn.metrics.pairwise import cosine_similarity
from absummarizer import summarizer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))
#import matplotlib.pyplot as plt

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# [ Class word_graph
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class word_graph:
    """
    The word_graph class constructs a word graph from the set of sentences given
    as input. The set of sentences is a list of strings, sentences are tokenized
    and words are POS-tagged (e.g. ``"Saturn/NNP is/VBZ the/DT sixth/JJ 
    planet/NN from/IN the/DT Sun/NNP in/IN the/DT Solar/NNP System/NNP"``). 
    Four optional parameters can be specified:

    - nb_words is is the minimal number of words for the best compression 
      (default value is 8).
    - lang is the language parameter and is used for selecting the correct 
      stopwords list (default is "en" for english, stopword lists are localized 
      in /resources/ directory).
    - punct_tag is the punctuation mark tag used during graph construction 
      (default is PUNCT).
    """

    #-T-----------------------------------------------------------------------T-
    def __init__(self, sentence_list, lang="en", punct_tag="PUNCT"):

        self.sentence = list(sentence_list)
        """ A list of sentences provided by the user. """

        self.length = len(sentence_list)
        """ The number of sentences given for fusion. """
        
        #self.nb_words = nb_words
        """ The minimal number of words in the compression. """

        self.resources = os.path.dirname(__file__)+"/../"+"resources/"
        #self.resources = os.path.dirname(__file__) + '/resources/'
        """ The path of the resources folder. """
        
        
        
        self.stopword_path = self.resources+'stopwords.'+lang+'.dat'
        """ The path of the stopword list, e.g. stopwords.[lang].dat. """

        self.stopwords = self.load_stopwords(self.stopword_path)
        """ The set of stopwords loaded from stopwords.[lang].dat. """

        self.punct_tag = punct_tag
        """ The stopword tag used in the graph. """

        self.graph = nx.DiGraph()
        """ The directed graph used for fusion. """
    
        self.start = '-start-'
        """ The start token in the graph. """

        self.stop = '-end-'
        """ The end token in the graph. """

        self.sep = '/-/'
        """ The separator used between a word and its POS in the graph. """
        
        self.term_freq = {}
        """ The frequency of a given term. """
        
        self.verbs = set(['VB', 'VBD', 'VBP', 'VBZ', 'VH', 'VHD', 'VHP', 'VBZ', 
        'VV', 'VVD', 'VVP', 'VVZ'])
        """
        The list of verb POS tags required in the compression. At least *one* 
        verb must occur in the candidate compressions.
        """
        
        # Replacing default values for French
        if lang == "fr":
            self.verbs = set(['V', 'VPP', 'VINF'])

        # 1. Pre-process the sentences
        self.pre_process_sentences()

        # 2. Compute term statistics
        self.compute_statistics()

        # 3. Build the word graph
        self.build_graph()
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def pre_process_sentences(self):
        """
        Pre-process the list of sentences given as input. Split sentences using 
        whitespaces and convert each sentence to a list of (word, POS) tuples.
        """

        for i in range(self.length):
        
            # Normalise extra white spaces
            self.sentence[i] = re.sub(' +', ' ', self.sentence[i])
            
            # Tokenize the current sentence in word/POS
            sentence = self.sentence[i].split(' ')
            #print sentence
            # Creating an empty container for the cleaned up sentence
            container = [(self.start, self.start)]
            
            #print "======", sentence,'==========='
            # Looping over the words
            j=0
            p=0
            for w in sentence:
                w=w.strip()
                #w=w.lstrip("#")
                p=p+1
                # Splitting word, POS
                if w.startswith("/"):
                    continue
                if w.startswith("@"):
                    continue
                if w.startswith("http"):
                    continue
                if w.startswith(".") or w.startswith("\?"):
                    continue
                if (w.startswith(":")) and j == 0:
                    continue
                #if w.startswith(":") and j == len(sentence) - 1:
                if w[0] in ':-\?' and p == (len(sentence)-1):
                    continue
                                
                j+=1
                #if w.startswith("#"):
                #    w=w[1:]
                m = re.match("^(.+)/(.+)$", w)
                
                # Extract the word information
                token, POS = m.group(1), m.group(2)
                               
             
                if (POS.strip() == ",") and p == 1:
                    continue
                #print token, POS
                if "RT" in token:
                    continue
                token=token.lstrip("#")
                # Add the token/POS to the sentence container
                if (sentence.index(w)==0 and (POS[0] not in ('^')) and w!="I"):
                    container.append((token.lower(), POS))
                else:
                    container.append((token, POS))  
                
                 
            
            # Add the stop token at the end of the container
            container.append((self.stop, self.stop))

            # Recopy the container into the current sentence
            self.sentence[i] = container
    #-B-----------------------------------------------------------------------B-
    

    def build_graph(self):
        """
        Constructs a directed word graph from the list of input sentences. Each
        sentence is iteratively added to the directed graph according to the 
        following algorithm:

        - Word mapping/creation is done in four steps:

            1. non-stopwords for which no candidate exists in the graph or for 
               which an unambiguous mapping is possible or which occur more than
               once in the sentence

            2. non-stopwords for which there are either several possible
               candidates in the graph

            3. stopwords

            4. punctuation marks

        For the last three groups of words where mapping is ambiguous we check 
        the immediate context (the preceding and following words in the sentence 
        and the neighboring nodes in the graph) and select the candidate which 
        has larger overlap in the context, or the one with a greater frequency 
        (i.e. the one which has more words mapped onto it). Stopwords are mapped 
        only if there is some overlap in non-stopwords neighbors, otherwise a 
        new node is created. Punctuation marks are mapped only if the preceding 
        and following words in the sentence and the neighboring nodes are the
        same.

        - Edges are then computed and added between mapped words.
        
        Each node in the graph is represented as a tuple ('word/POS', id) and 
        possesses an info list containing (sentence_id, position_in_sentence)
        tuples.
        """     

        # Iteratively add each sentence in the graph ---------------------------
        for i in range(self.length):

            # Compute the sentence length
            sentence_len = len(self.sentence[i])

            # Create the mapping container
            mapping = [0] * sentence_len

            #-------------------------------------------------------------------
            # 1. non-stopwords for which no candidate exists in the graph or for 
            #    which an unambiguous mapping is possible or which occur more 
            #    than once in the sentence.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue
            
                # Create the node identifier
                node = token + self.sep + POS

                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)],
                                         label=token )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)

                # If there is only one matching node in the graph (id is 0)
                elif k == 1:

                    # Get the sentences id of this node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, 0)]['info']:
                        ids.append(sid)
                    
                    # Update the node in the graph if not same sentence
                    if not i in ids:
                        self.graph.node[(node, 0)]['info'].append((i, j))
                        mapping[j] = (node, 0)

                    # Else Create new node for redundant word
                    else:
                        self.graph.add_node( (node, 1), info=[(i, j)], 
                                             label=token )
                        mapping[j] = (node, 1)

            #-------------------------------------------------------------------
            # 2. non-stopwords for which there are either several possible
            #    candidates in the graph.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]
                
                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue

                # If word is not already mapped to a node
                if mapping[j] == 0:

                    # Create the node identifier
                    node = token + self.sep + POS
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token + self.sep + prev_POS
                    next_node = next_token + self.sep + next_POS
                    
                    # Find the number of ambiguous nodes in the graph
                    k = self.ambiguous_nodes(node)

                    # Search for the ambiguous node with the larger overlap in
                    # context or the greater frequency.
                    ambinode_overlap = []
                    ambinode_frequency = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)

                        # Add the frequency of the ambiguous node
                        ambinode_frequency.append(
                            len( self.graph.node[(node, l)]['info'] )
                        )
                
                    # Search for the best candidate while avoiding a loop
                    found = False
                    selected = 0
                    while not found:
                    
                        # Select the ambiguous node
                        selected = self.max_index(ambinode_overlap)
                        if ambinode_overlap[selected] == 0:
                            selected = self.max_index(ambinode_frequency)
                        
                        # Get the sentences id of this node
                        ids = []
                        for sid, p in self.graph.node[(node, selected)]['info']:
                            ids.append(sid)
                        
                        # Test if there is no loop
                        if i not in ids:
                            found = True
                            break
            
                        # Remove the candidate from the lists
                        else:
                            del ambinode_overlap[selected]
                            del ambinode_frequency[selected]
                            
                        # Avoid endless loops
                        if len(ambinode_overlap) == 0:
                            break
                    
                    # Update the node in the graph if not same sentence
                    if found:
                        self.graph.node[(node, selected)]['info'].append((i, j))
                        mapping[j] = (node, selected)

                    # Else create new node for redundant word
                    else:
                        self.graph.add_node( (node, k), info=[(i, j)], 
                                             label=token )
                        mapping[j] = (node, k)
            
            #-------------------------------------------------------------------
            # 3. map the stopwords to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* stopword, continues
                if not token in self.stopwords :
                    continue

                # Create the node identifier
                node = token + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token + self.sep + prev_POS
                    next_node = next_token + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes, the
                        # boolean indicates to consider only non stopwords
                        l_context = self.get_directed_context(node, l, 'left',\
                                    True)
                        r_context = self.get_directed_context(node, l, 'right',\
                                    True)
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 0:
                    # if i not in ids and \
                    # (ambinode_overlap[selected] > 1 and POS==self.punct_tag) or\
                    # (ambinode_overlap[selected] > 0 and POS!=self.punct_tag) :

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k) , info=[(i, j)],
                                             label=token )

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. lasty map the punctuation marks to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* punctuation mark, continues
                if not re.search('(?u)^\W$', token):
                    continue

                # Create the node identifier
                node = token + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token + self.sep + prev_POS
                    next_node = next_token + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 1:

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k), info=[(i, j)], 
                                             label=token )

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. Connects the mapped words with directed edges
            #-------------------------------------------------------------------
            for j in range(1, len(mapping)):
                #print 'Mapping j-1', mapping[j-1], self.graph.node[mapping[j-1]]['info']
                self.graph.add_edge(mapping[j-1], mapping[j], marker=1)

        # Assigns a weight to each node in the graph ---------------------------
        for node1, node2 in self.graph.edges_iter():
            edge_weight = self.get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)
    #-B-----------------------------------------------------------------------B-

 
    #-T-----------------------------------------------------------------------T-
    def ambiguous_nodes(self, node):
        """
        Takes a node in parameter and returns the number of possible candidate 
        (ambiguous) nodes in the graph.
        """
        k = 0
        #if node == "," + "/" + "," :
        #    return k
        while(self.graph.has_node((node, k))):
            k += 1
        return k
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_directed_context(self, node, k, dir='all', non_pos=False):
        """
        Returns the directed context of a given node, i.e. a list of word/POS of
        the left or right neighboring nodes in the graph. The function takes 
        four parameters :

        - node is the word/POS tuple
        - k is the node identifier used when multiple nodes refer to the same 
          word/POS (e.g. k=0 for (the/DET, 0), k=1 for (the/DET, 1), etc.)
        - dir is the parameter that controls the directed context calculation, 
          it can be set to left, right or all (default)
        - non_pos is a boolean allowing to remove stopwords from the context 
          (default is false)
        """

        # Define the context containers
        l_context = []
        r_context = []

        # For all the sentence/position tuples
        for sid, off in self.graph.node[(node, k)]['info']:
            
            prev = self.sentence[sid][off-1][0] + self.sep +\
                   self.sentence[sid][off-1][1]
                   
            next = self.sentence[sid][off+1][0] + self.sep +\
                   self.sentence[sid][off+1][1]
                   
            if non_pos:
                if self.sentence[sid][off-1][0] not in self.stopwords:
                    l_context.append(prev)
                if self.sentence[sid][off+1][0] not in self.stopwords:
                    r_context.append(next)
            else:
                l_context.append(prev)
                r_context.append(next)

        # Returns the left (previous) context
        if dir == 'left':
            return l_context
        # Returns the right (next) context
        elif dir == 'right':
            return r_context
        # Returns the whole context
        else:
            l_context.extend(r_context)
            return l_context
    #-B-----------------------------------------------------------------------B-

    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_edge_weight(self, node1, node2):
        """
        Compute the weight of an edge *e* between nodes *node1* and *node2*. It 
        is computed as e_ij = (A / B) / C with:
        
        - A = freq(i) + freq(j), 
        - B = Sum (s in S) 1 / diff(s, i, j)
        - C = freq(i) * freq(j)
        
        A node is a tuple of ('word/POS', unique_id).
        """

        # Get the list of (sentence_id, pos_in_sentence) for node1
        info1 = self.graph.node[node1]['info']
        
        # Get the list of (sentence_id, pos_in_sentence) for node2
        info2 = self.graph.node[node2]['info']
        
        # Get the frequency of node1 in the graph
        # freq1 = self.graph.degree(node1)
        freq1 = len(info1)
        
        # Get the frequency of node2 in cluster
        # freq2 = self.graph.degree(node2)
        freq2 = len(info2)

        # Initializing the diff function list container
        diff = []

        # For each sentence of the cluster (for s in S)
        for s in range(self.length):
        
            # Compute diff(s, i, j) which is calculated as
            # pos(s, i) - pos(s, j) if pos(s, i) < pos(s, j)
            # O otherwise
    
            # Get the positions of i and j in s, named pos(s, i) and pos(s, j)
            # As a word can appear at multiple positions in a sentence, a list
            # of positions is used
            pos_i_in_s = []
            pos_j_in_s = []
            
            # For each (sentence_id, pos_in_sentence) of node1
            for sentence_id, pos_in_sentence in info1:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_i_in_s.append(pos_in_sentence)
            
            # For each (sentence_id, pos_in_sentence) of node2
            for sentence_id, pos_in_sentence in info2:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_j_in_s.append(pos_in_sentence)
                    
            # Container for all the diff(s, i, j) for i and j
            all_diff_pos_i_j = []
            
            # Loop over all the i, j couples
            for x in range(len(pos_i_in_s)):
                for y in range(len(pos_j_in_s)):
                    diff_i_j = pos_i_in_s[x] - pos_j_in_s[y]
                    # Test if word i appears *BEFORE* word j in s
                    if diff_i_j < 0:
                        all_diff_pos_i_j.append(-1.0*diff_i_j)
                        
            # Add the mininum distance to diff (i.e. in case of multiple 
            # occurrencies of i or/and j in sentence s), 0 otherwise.
            if len(all_diff_pos_i_j) > 0:
                diff.append(1.0/min(all_diff_pos_i_j))
            else:
                diff.append(0.0)
                
        weight1 = freq1
        weight2 = freq2

        return ( (freq1 + freq2) / sum(diff) ) / (weight1 * weight2)
    #-B-----------------------------------------------------------------------B-
   
   
    #-T-----------------------------------------------------------------------T-
        
   
    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def max_index(self, l):
        """ Returns the index of the maximum value of a given list. """

        ll = len(l)
        if ll < 0:
            return None
        elif ll == 1:
            return 0
        max_val = l[0]
        max_ind = 0
        for z in range(1, ll):
            if l[z] > max_val:
                max_val = l[z]
                max_ind = z
        return max_ind
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def compute_statistics(self):
        """
        This function iterates over the cluster's sentences and computes the
        following statistics about each word:
        
        - term frequency (self.term_freq)
        """

        # Structure for containing the list of sentences in which a term occurs
        terms = {}

        # Loop over the sentences
        for i in range(self.length):
            #print self.sentence[i]
            # For each tuple (token, POS) of sentence i
            for token, POS in self.sentence[i]:
            
                # generate the word/POS token
                #node = token + self.sep + POS
                node = token + self.sep + POS
                # Add the token to the terms list
                if not terms.has_key(node):
                    terms[node] = [i]
                else:
                    terms[node].append(i)

        # Loop over the terms
        for w in terms:

            # Compute the term frequency
            self.term_freq[w] = len(terms[w])
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def load_stopwords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip())

        # Return the set of stopwords
        return stopwords
    #-B-----------------------------------------------------------------------B-
    

    #-T-----------------------------------------------------------------------T-
    def write_dot(self, dotfile):
        """ Outputs the word graph in dot format in the specified file. """
        nx.write_dot(self.graph, dotfile)
    #-B-----------------------------------------------------------------------B-
    
    def normalizedWords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        words = dict()

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            ww=line.split("==>")
            words[ww[0].strip()]=ww[1].strip()
        # Return the set of stopwords
        return words
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ] Ending word_graph class
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def wordpos_to_tuple(self, word, delim='/'):
        """
        This function converts a word/POS to a (word, POS) tuple. The character
        used for separating word and POS can be specified (default is /).
        """

        # Splitting word, POS using regex
        m = re.match("^(.+)"+delim+"(.+)$", word)

        # Extract the word information
        token, POS = m.group(1), m.group(2)

        # Return the tuple 
        return (token, POS)
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def tuple_to_wordpos(self, wordpos_tuple, delim='/'):
        """
        This function converts a (word, POS) tuple to word/POS. The character 
        used for separating word and POS can be specified (default is /).
        """

        # Return the word +delim+ POS
        return wordpos_tuple[0]+delim+wordpos_tuple[1]
    #-B-----------------------------------------------------------------------B-
    

def sentenceTuple(sentence):
    modSentence = []
    position=0
    for w,t in sentence:
        if t in ('NNP', 'NNPS','^'):
            modSentence.append((w.title(),t))  
        else:
            modSentence.append((w,t))  
        position +=1    
    #print modSentence
    return modSentence

def load_stopwords(path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip())

        # Return the set of stopwords
        return stopwords
    #-B-------------------
 
def getVertex(graph, string):
    for vertex in graph.vs:
        if string in vertex['name']: 
            return vertex
 

def find_all_paths_igraph(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    paths = []
    for node in set(graph.neighbors(start)) - set(path):
        paths.extend(find_all_paths_igraph(graph, node, end, path))
    return paths

def findPaths2(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths2(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths

def find_all_paths_nx(graph, start, end):
    path  = []
    paths = []
    queue = [(start, end, path)]
    while queue:
        start, end, path = queue.pop()
        #print 'PATH', path
        path = path + [start]
        if start == end and len(path)>= 12 and len(path)<= 15:
            paths.append(path)
            if len(paths)>50000:
                return paths
        for node in set(graph[start]).difference(path):
            queue.append((node, end, path))
    return paths


def find_all_paths_igraph_adj(graph, start, end):
    def find_all_paths_aux(adjlist, start, end, path):
        path = path + [start]
        if start == end and (len(path) > 10 and len(path) < 25):
                return [path]
        paths = []
        for node in adjlist[start] - set(path):
            paths.extend(find_all_paths_aux(adjlist, node, end, path))
            if len(paths) >=10000:
                return paths
        return paths

    adjlist = [set(graph.neighbors(node)) for node in xrange(graph.vcount())]
    return find_all_paths_aux(adjlist, start, end, [])

def find_all_paths_tamas(graph, start, end):
    def find_all_paths_aux_tamas(adjlist, start, end, path):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        #if len(path) < 22:# and len(path) < 25:
        for node in adjlist[start] - set(path):
            paths.extend(find_all_paths_aux_tamas(adjlist, node, end, path))
            #if len(paths) > 20000:
            #    return paths
            #if len(paths) > 10000:
            #    return paths
        return paths

    adjlist = [set(graph.neighbors(node)) for node in xrange(graph.vcount())]
    return find_all_paths_aux_tamas(adjlist, start, end, [])



 
def adjlist_find_paths(a, n, m, path=[]):
    "Find paths from node index n to m using adjacency list a."
    path = path + [n]
    if n == m:
        return [path]
    paths = []
    #successors=a.successors(n)
    #shuffle(successors)
    for child in a[n]:
        if child not in path:
            child_paths = adjlist_find_paths(a, child, m, path)
            for child_path in child_paths:
                if len(child_path)<12: #<=25:
                    continue
                if len(child_path)>25: #<=25:
                    continue
                #print "Appending path"
                paths.append(child_path)
                #print len(paths)
                if (len(paths)>=1000):
                    return paths
    return paths
 
def paths_from_to_old(graph, source, dest):
    "Find paths in graph from vertex source to vertex dest."
    a = graph.get_adjlist()
    n = source.index
    m = dest.index
    return adjlist_find_paths(a, n, m)


#import shortestPath as sp
def paths_from_to(graph, source, dest):
    "Find paths in graph from vertex source to vertex dest."
    a = graph.get_adjlist()
    n = source.index
    m = dest.index
    #return graph.get_shortest_paths(source, dest)
    #return sp.yen_igraph(graph, source, dest, 1000)
    #return graph.get_shortest_paths(n, to=m)
    #r#eturn yenksp.algorithms.ksp_yen(graph, source, dest, max_k=10000)
    #return sp.yen_igraph(graph, source, dest, 2000, weights=None)
    return adjlist_find_paths(a, n, m)      
 
def getWordFromVertexName(nameString):
    word=nameString.split('/')[0]
    if word not in ['-start-','-end-']:
        return word
    else:
        return ''
 
def generateTempRewrittenSentences(taggedSentences):
    final_tagged_Sentences=[]
    for ensent in taggedSentences:
        tagged_sentence=''
        for w, t in ensent:
            tagged_sentence=tagged_sentence+w+'/'+t+' '
        final_tagged_Sentences.append(tagged_sentence.strip())
    return final_tagged_Sentences
    

def retrieveNewSentences(sentences, stopwords, mode=None):
    taggedTweets=list(sentences)
    shuffle(taggedTweets)
    simMatrix=simCalcMatrix(taggedTweets)
    compresser = word_graph(taggedTweets, 
                                lang = 'en', 
                                punct_tag = "PUNCT" )
    print "Number of nodes", nx.number_of_nodes(compresser.graph)
#candidates = compresser.get_compression(100)
    #print compresser.graph.nodes(data='True')
    '''
    #NOT USING THIS NOW: THIS is for IGRAPH
    
    matrix=nx.to_scipy_sparse_matrix(compresser.graph)
    nodelist=compresser.graph.nodes()
    g = scipy_to_igraph(matrix,nodelist)
    
    startvertex = getVertex(g, '-start-/-/-start-')
    endvertex = getVertex(g, '-end-/-/-end-')
    vertexList = g.vs()
    startvertex=startvertex.index
    endvertex=endvertex.index
    #elements=g.get_all_shortest_paths(startvertex, endvertex)[0]
    
    print "WG done"
    #allpaths=elements
    allpaths = find_all_paths_igraph_adj(g, startvertex, endvertex)
    '''
    #nlist= nx.nodes(compresser.graph)
    
    #for n in nlist:
    #    print 'Info', compresser.graph.node[n]['info']
        
    
    g=nx.convert_node_labels_to_integers(compresser.graph)
    nodelist=g.nodes(data=True)
    for node in nodelist:
        n, r = node
        #print 'Labels--', r['label']
        if r['label']=='-start-':
            startnode=n
        if r['label']=='-end-':
            endnode=n
    
    
    label_list=[]
    
    '''
    Labellist contains info and label : info has info on which sentences resulted in that node
    '''
    
    for node in nodelist:
        #print node
        r, d = node
        if d['label'] not in ['-start-','-end-']:
            label_list.append((d['info'],d['label']))  
        else:
            label_list.append(('',''))  
    
    #if mode in 'EXTRACT':
    cutoff_threshold=18
    import timeit
    starttime=timeit.default_timer()
    
    allpaths=[]
    for path in nx.all_simple_paths(g, startnode, endnode, cutoff=cutoff_threshold):
        allpaths.append(path)
        if len(allpaths) >=100000:
            break
    print "Total time for getting all paths in seconds: ", (timeit.default_timer()-starttime), " s" 
   
    
    print "Total paths, ", len(allpaths)
    shuffle(allpaths)

    #allpaths=allpaths[0:10000]
    generatedSentences = []
    sentence_container = {}
    for path in allpaths:
        #print 'Path', path
        if len(path)<12:
            continue
        paired_parentheses = 0
        quotation_mark_number = 0
        sentence = ' '.join(label_list[element][1].split("||")[0] for element in path)
        
        avgSim = avgPairwiseSimilarity(simMatrix, getSentIndices(path, label_list))
        for word in sentence.split():
                #print word
            if word == '(':
                paired_parentheses -= 1
            elif word == ')':
                paired_parentheses += 1
            elif (word == '\"' or word == '\'\'' or word == '``' or word == '"'):
                quotation_mark_number += 1
                
        if (paired_parentheses == 0 and quotation_mark_number%2 == 0 and not sentence_container.has_key(sentence.strip())):   
            generatedSentences.append((sentence.strip(), avgSim))  
            sentence_container[sentence.strip()]=1

    shuffle(generatedSentences)
    #if mode in 'EXTRACT':
    generatedSentences= sorted(generatedSentences, key=lambda tup: tup[1], reverse=True)
    
    #else:
    generatedSentences=generatedSentences[0:300]
    print "Num variables -->", len(generatedSentences)
    
    svolist=[]
    
    generatedSentences=removeSimilarSentences(generatedSentences, sentences, stopwords)
    
    return generatedSentences, svolist
#    print generatedSentences


def avgPairwiseSimilarity(simMatrix, indices):
    num_elements=len(indices)
    sum_sim=0.0
    num=0.0
    if num_elements==1:
        return 0.00001
    for i in xrange(0, num_elements-1):
        for j in xrange(i+1, num_elements):
            sum_sim+=simMatrix[i,j] 
            num+=1
    
    #print num
    return (sum_sim/num)
            
    
def getSentIndices(path, labellist):
    sentenceSet=set()
    path_length=len(path)
    
    for i in xrange(1,path_length):
        keys1=labellist[path[i-1]][0]
        sentindices_1=[sentnum for sentnum, wordnum in keys1]
        keys2=labellist[path[i]][0]
        sentindices_2=[sentnum for sentnum, wordnum in keys2]
        sentence_indices=set(sentindices_1).intersection(set(sentindices_2))
        for ind in sentence_indices:
            sentenceSet.add(ind)
    return sentenceSet



def removeSimilarSentences(generatedSentences, originalSentences,  stopwords,threshold=0.80,):
    docs=[]
    for sent, sim in generatedSentences:
        docs.append(sent)
    docs.extend(originalSentences)
    
    bow_matrix = StemmedTfidfVectorizer(stop_words=stopwords).fit_transform(docs)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
    #simMatrix = (normalized[0:] * normalized[0:].T).A
    simindices=[]
    #print 'Num original, ', len(originalSentences)
    for i in xrange(len(generatedSentences)):
        simGeneratedScores = linear_kernel(normalized[i], normalized[len(generatedSentences):]).flatten()
        if(max(simGeneratedScores) >= threshold):
            simindices.append(i)
    
    #print simindices
    finalGen=[sentence for k,sentence in enumerate(generatedSentences) if k not in simindices]
    #print len(generatedSentences), len(finalGen)
    return finalGen


def getSVO(sentence):
    sbjstring=' '
    objstring=' '
    vpstring=' '
    s = parsetree(sentence, tokenize = True, relations=True, lemmata = False)
    relationList=s.sentences[0].relations
    if 'SBJ' in relationList:
        for chunk in relationList['SBJ'].values():
            #print chunk.words
            sbjstring =sbjstring+' '+ ' '.join(word.string for word in chunk.words)
    if 'OBJ' in relationList:
        for chunk in relationList['OBJ'].values():
            #print chunk.words
            objstring =objstring+' '+ ' '.join(word.string for word in chunk.words)
    if 'VP' in relationList:
        for chunk in relationList['VP'].values():
            #print chunk.words
            vpstring =vpstring+' '+ ' '.join(word.string for word in chunk.words)        
    #print sbjstring.strip()
    #print objstring.strip()
    #print vpstring.strip()
    return sbjstring.strip(), vpstring.strip(), objstring.strip()


def simCalcMatrix(docs):
    tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=None)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(docs)  #finds the tfidf score with normalization
    cosineSimilarities=cosine_similarity(tfidf_matrix_train, tfidf_matrix_train) 
    return cosineSimilarities

def generateSimMatrix(phraseList):
    #print 'Num elements', len(phraseList), phraseList
    all_elements=[]
    #for elementlist in phraseList:
    for element in phraseList:
        if len(element.strip())==0:
            all_elements.append(' ')
        else:
            all_elements.append(element.strip())
    tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=None)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(all_elements)  #finds the tfidf score with normalization
    cosineSimilarities=cosine_similarity(tfidf_matrix_train, tfidf_matrix_train) 
    return cosineSimilarities



from nltk.tokenize import WordPunctTokenizer

def getredundantComponents(sentences):
    window_size=4
    introList=[]
    midlist=[]
    endlist=[]
    
    for sent in sentences:
        words = WordPunctTokenizer().tokenize(sent)
        length_sent=len(words)
        
        f_point = (length_sent)//3
        m_point=(length_sent)//2
        index_span=window_size//2
        intro=' '.join(word for word in words[0:window_size])
        mid=' '.join(word for word in words[m_point-index_span:m_point+index_span])
        end=' '.join(word for word in words[-window_size:])
        introList.append(intro)
        midlist.append(mid)
        endlist.append(end)
    return introList, midlist, endlist

    
def solveILPFactBased(groupedList, languageModel, stopwords,ranker, intraGenSimThreshold=0.5,l_max=100, mode="Extractive" ):
    
    if len(groupedList) == 0:
        return
    # Create a new model
    print "Starting to solve ILP...."
    
    m = LpProblem("mip1", LpMaximize)
    sbjthreshold=0.3
    objthreshold=0.4
    
    docs=[]
    intraSentRelatedNessScores=[]
    if mode == "Extractive":
        for element in groupedList:
            docs.append(element)
    else:
        for element, sim in groupedList:
            docs.append(element)
            intraSentRelatedNessScores.append(sim)

    '''
    Full sentence cosine sim comparison
    '''
    bow_matrix = StemmedTfidfVectorizer(stop_words=stopwords).fit_transform(docs)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
    cosine_similarity_matrix = (normalized * normalized.T).A
   
    '''
    splitting a sentence into three parts
    '''
    ilist, mlist, elist=getredundantComponents(docs)
    bow_matrix_i = StemmedTfidfVectorizer(stop_words=None).fit_transform(ilist)
    normalized_i = TfidfTransformer().fit_transform(bow_matrix_i)
    cosine_similarity_matrix_i = (normalized_i * normalized_i.T).A

    bow_matrix_m = StemmedTfidfVectorizer(stop_words=None).fit_transform(mlist)
    normalized_m = TfidfTransformer().fit_transform(bow_matrix_m)
    cosine_similarity_matrix_m = (normalized_m * normalized_m.T).A
    
    #elist=factContent(elist, contentWordScores)
    bow_matrix_e = StemmedTfidfVectorizer(stop_words=None).fit_transform(elist)
    normalized_e = TfidfTransformer().fit_transform(bow_matrix_e)
    cosine_similarity_matrix_e = (normalized_e * normalized_e.T).A
    

    txtRankScores=[]
    if ranker == "Centroid":
        txtRankScores=summarizer.getScores(docs)
    if ranker == "textrank":
        sources, targets = cosine_similarity_matrix.nonzero()
        similarity_igraph = igraph.Graph(zip(sources, targets), directed=True)
        txtRankScores = igraph.Graph.pagerank(similarity_igraph)
 
    
    lingQualityScores=[]
    if mode=="Abstractive":
        for sent in docs:
            normalizer=len(sent.split(" "))
            lm_score=(1/normalizer)*languageModel.score(sent)
            lm_score=1/(1-lm_score) 
            lingQualityScores.append(lm_score)


    varlist=[]        
    for i in xrange(len(docs)):
            var=LpVariable("var_"+str(i),cat=LpBinary)
            varlist.append(var)
    
    if mode=="Abstractive":
        m += lpSum([txtRankScores[i]*lingQualityScores[i]*varlist[i] for i in xrange(len(txtRankScores))]), "Obj function"  
    else:
        m += lpSum([txtRankScores[i]*varlist[i] for i in xrange(len(txtRankScores))]), "Obj function" 
    
    visitedlist=[]
    for i in xrange(len(varlist)):
        i_indices=np.where(cosine_similarity_matrix_i[i,:]>= objthreshold)[0]
        m_indices=np.where(cosine_similarity_matrix_m[i,:]>= objthreshold)[0]
        e_indices=np.where(cosine_similarity_matrix_e[i,:]>= objthreshold)[0]
        s_indices=np.where(cosine_similarity_matrix[i,:]>= sbjthreshold)[0]    
        all_indices=np.concatenate((i_indices, m_indices))
        all_indices=np.concatenate((all_indices, e_indices))
        all_indices=np.unique(all_indices)
        all_indices=np.concatenate((all_indices, s_indices))
        all_indices=np.unique(all_indices).tolist()
        for j in all_indices:
            if i==j:
                continue
            
            if j==len(varlist):
                continue
            #print j
            if (i, j) not in visitedlist:
                visitedlist.append((i,j))
                m+=varlist[i] + varlist[j] <=1.0, "constraint_facts_svo_"+str(i)+"_"+varlist[i].name+"_"+varlist[j].name 

        completelist=[]
        completelist.extend(mlist)
        completelist.append(ilist[i])     
        lastelement=len(completelist)-1   
        bow_matrix_ilist = StemmedTfidfVectorizer(stop_words=None).fit_transform(completelist)
        normalized_ilist = TfidfTransformer().fit_transform(bow_matrix_ilist)
        cosine_similarity_matrix_ilist = (normalized_ilist * normalized_ilist.T).A
        r_indices=np.where(cosine_similarity_matrix_ilist[lastelement,:]>= objthreshold)[0]
        #print len(completelist)
        #print "R_indices", r_indices
        oth_indices=r_indices
        completelist=[]
        completelist.extend(elist)
        completelist.append(ilist[i])
        bow_matrix_ilist = StemmedTfidfVectorizer(stop_words=None).fit_transform(completelist)
        normalized_ilist = TfidfTransformer().fit_transform(bow_matrix_ilist)
        cosine_similarity_matrix_ilist = (normalized_ilist * normalized_ilist.T).A
        r_indices=np.where(cosine_similarity_matrix_ilist[lastelement,:]>= objthreshold)[0]
        #print "P_indices", p_indices
        oth_indices=np.concatenate((oth_indices, r_indices)) 
         
        completelist=[]
        completelist.extend(mlist)
        completelist.append(elist[i])
        bow_matrix_ilist = StemmedTfidfVectorizer(stop_words=None).fit_transform(completelist)
        normalized_ilist = TfidfTransformer().fit_transform(bow_matrix_ilist)
        cosine_similarity_matrix_ilist = (normalized_ilist * normalized_ilist.T).A
        r_indices=np.where(cosine_similarity_matrix_ilist[lastelement,:]>= objthreshold)[0]
        #print "Q_indices", q_indices
        oth_indices=np.concatenate((oth_indices, r_indices)) 
        
        oth_indices=np.unique(oth_indices).tolist()
        for j in oth_indices:
            if i==j:
                continue
            
            if j==len(varlist):
                continue
            #print j
            if (i, j) not in visitedlist:
                visitedlist.append((i,j))
                m+=varlist[i] + varlist[j] <=1.0, "constraint_facts_comps_"+str(i)+"_"+varlist[i].name+"_"+varlist[j].name 
        

    gen_lengths=[]
    for i in xrange(len(txtRankScores)):
        words=docs[i].split(" ")
        count=0
        for word in words:
            if word[0].isalpha() or word[0].isdigit():
                count+=1
        gen_lengths.append(count)
    #print "Gen Lengths" , gen_lengths
 
    m += lpSum([varlist[i]*gen_lengths[i] for i in xrange(len(txtRankScores))]) <= l_max, "length of summary"  

    m.solve()
    solutionList=[] 
    
    for v in m.variables():
        if v.varValue == 1.0:
            indexVar=v.name.split("_")[1]
            solutionList.append(docs[int(indexVar)])
    return solutionList


