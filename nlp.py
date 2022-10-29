
# for nlp and regular expression
import modu as nlp
from nltk.tokenize import word_tokenize
import string
import re

# for word cloud
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud , STOPWORDS

import sys

filename = sys.argv[1]

# read file
text=""
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

# Preprocessing
text = nlp.convert_lower(text)
text = nlp.remove_symbols(text)
text = nlp.tokenize(text)
text = nlp.remove_url(text)
text = nlp.lemment(text)

# Tokenization
T1 = word_tokenize(text)
nlp.tok_vs_freq(T1)

# raw word cloud
nlp.make_cloud(T1)

# word cloud without stopwords
stopwords = set(STOPWORDS)
Temp = T1
T1 = nlp.remove_stop(T1,stopwords)
nlp.make_cloud(T1)

nlp.length_vs_freq(Temp)
nlp.length_vs_freq(T1)

# POS tagging
Tags=nlp.pos_tag_penn(T1)

#Tag Distribution
nlp.tag_dist(Tags)