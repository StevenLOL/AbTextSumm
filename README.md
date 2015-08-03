# TweetAbSumm
Abstractive Summarization on Twitter

For language model:
Needs kenlm: https://kheafield.com/code/kenlm/ [See how to install]
Use any available ARPA format language model and convert to kenlm format as binary. KENLM is really fast. 

Other several packages required: PuLP for optimization, sklearn, nltk, cpattern
Best option is to use Anaconda package. All the above mentioned packages can be installed using pip.

Create a folder in src called jars and put the files as mentioned in the python file. Namely, these files are:
english-left3words-distsim.tagger
english-left3words-distsim.tagger.props
stanford-postagger.jar

All these can be the latest POS tagger from Stanford. 

**If you use the summarization code, please cite this paper:**
_Banerjee, Siddhartha, Prasenjit Mitra, and Kazunari Sugiyama_. "Multi-Document Abstractive Summarization Using ILP based Multi-Sentence Compression." Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI 2015), Buenos Aires, Argentina. 2015.
