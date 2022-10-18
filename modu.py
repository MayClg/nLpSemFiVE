# for nlp and regular expression
import nltk
import re

# for word cloud
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud , STOPWORDS

# make lowercase
def convert_lower(text):
    return text.lower()

# remove everything that is not english alphabets and single length chaacters except 'a'
def remove_symbols(text):
    text = re.sub('[\n]', ' ', text)
    text = re.sub('[-]+', '', text)
    text = re.sub('[^A-Za-z ]+', '', text)
    text = re.sub(r'(?i)\b[a-z]\b', ' ', text)
    return text

# Tokenize
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens]
    return ' '.join(words)

# remove urls
def remove_url(text):
    text = re.sub(r"http\S+","",text)
    return text

# lemmentization
def lemment(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word,pos='v') for word in tokens]
    return ' '.join(lemmas)

# making word cloud
def make_cloud(T):
    word_cloud = WordCloud(width=1920*2,height=1080*2,collocations = False, background_color = 'white',stopwords={},min_font_size=1).generate(' '.join(T))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()

# removing stop words
def remove_stop(T,stopwords):
    return [word for word in T if word not in stopwords]

#graph plotting
def length_vs_freq(T):
    gset=[]
    for i in T:
        gset.append(len(i))
    pd.Series(gset).value_counts().plot(kind='bar')
    plt.title("Length of word Vs Frequency")
    plt.xlabel('Length')
    plt.ylabel("Frequency")
    plt.show()

def tok_vs_freq(T):
    pd.Series(T).value_counts()[:20].plot(kind='bar')
    plt.title("Tokes Vs Frequency")
    plt.xlabel('Tokens')
    plt.ylabel("Frequency")
    plt.show()

# POS tagging
def pos_tag_penn(T):
    tokset = set()
    for i in T:
        tokset.add(i)
    Tags = nltk.pos_tag(tokset)
    for i in Tags:
        print(i)
    return Tags
