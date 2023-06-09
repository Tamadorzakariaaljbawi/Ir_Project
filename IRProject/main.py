from turtle import pd

#import np as np
import statsmodels.api as sm
import sm as sm
#import en_core_web_sm
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from autocorrect import Speller
import string
import re
import datefinder
import nltk
from nltk.corpus import wordnet
from jellyfish import soundex, metaphone, nysiis, match_rating_codex,\
    levenshtein_distance, damerau_levenshtein_distance, hamming_distance,\
    jaro_similarity
from itertools import groupby
import pandas as pd
import numpy as np
import soundex
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances



lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def stem_words(txt):
    stems = [stemmer.stem(word) for word in txt]
    return stems


def lemmatize_word(txt):
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in txt]
    return lemmas

####-----------------6,7-----#######
def tf_idf(search_keys, Docs):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_weights_matrix = tfidf_vectorizer.fit_transform(Docs)
    search_query_weights = tfidf_vectorizer.transform([search_keys])
    return search_query_weights, tfidf_weights_matrix


def cos_similarity(search_query_weights, tfidf_weights_matrix):
    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_distance[0]

    return similarity_list


def most_similar(similarity_list, min_Doc=1):
    most_similar = []

    while min_Doc > 0:
        tmp_index = np.argmax(similarity_list)
        most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0
        min_Doc -= 1

    return most_similar

def K_Means(tfidf_weights_matrix,num=5):
    kmeans = KMeans(n_clusters= num, init='k-means++', max_iter = 500, n_init = 1)
    kmeans.fit(tfidf_weights_matrix)
    print(kmeans.cluster_centers_)
    return kmeans

# ***************query clean************* #
file = open("common_words", "r")
fileData = file.read()
file.close()
stopwords = re.findall("\S+", fileData)

query = "hoq are yuo pleased me,?"

# auto correct
spell = Speller(lang='en')
Query = spell(query)

# split into words
tokens = word_tokenize(Query)

# convert to lower case
tokens = [w.lower() for w in tokens]

# remove punctuation from each word
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

# filter out stop words
stop_words = set(stopwords)
words = [w for w in stripped if not w in stop_words]

# stemming
stem_word = stem_words(words)

#print(stem_word)

# ****************text preprocessing start***************** #
#
# # read stop word file
#
file = open("common_words", "r")
fileData = file.read()
file.close()
stopwords = re.findall("\S+", fileData)

for x in range(1, 1460):
    # read from files
    f = open("corpus1/{}.text".format(x), "r")
    text = f.read()
    f.close()

##------------((( SENT TOKENIZE ))): to split up -------------##

    sent_tokenize(text)

##------------(((TOKENIZE by word))):split into words----------##

    print("-------------------")
    print("Orginal text :",text)
    tokens = word_tokenize(text)
    print("Doc :",x)
    print("TOKENIZE:",tokens)
    #print(tokens)

##------------((( convert to lower case)))---------------------##
##----(((  remove all punctuation except words and space)))----------------##

    tokens = [w.lower() for w in tokens]
    print("lower:",tokens)

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    print("remove punctuation:",stripped)
#
  # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    print("not alphabetic:",words)
#
# #   # filter out stop words
    stop_words = set(stopwords)
    words = [w for w in words if not w in stop_words]
    print("filter stop words:",words)
#
#   # stemming
    stem_word = stem_words(words)
    print("stemming:",stem_word)
#

##------- ((lemmetization with pos))------##

    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    listToStr = ' '.join([str(elem) for elem in words])
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))

    print("pos_tagged :",pos_tagged)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    print("wordnet_tagged :",wordnet_tagged)

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:

            lemmatized_sentence.append(word)
        else:

            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    print("lemmatized_sentence :",lemmatized_sentence)

###-------------imp 6,7---------------###
    text2="res q res n"
    res,res2 = tf_idf(text2, words)
    print(res,res2)

    res3=cos_similarity(res,res2)
    print(res3)

    res4=most_similar(res3)
    print(res4)
###---------------(Q): 3 -----------------------###
    matches = datefinder.find_dates(text)
    #for match in matches:
        #print(match)

    spacy_model = spacy.load('en_core_web_sm')
    listToStr2 = ' '.join([str(elem) for elem in words])
    entity_doc = spacy_model(listToStr2)
    print([(entity.text, entity .label_) for entity in entity_doc.ents])




# nlp = spacy.load("en_core_web_sm")
#
# # text = "Zoni I want to find a pencil, a eraser and a sharpener"
#
# text = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
#
# doc = nlp(text)
# print(doc.ents)

###---------------(Q): 5 -----------------------###
    sounds_encoding_methods = [soundex, metaphone, nysiis, match_rating_codex]
    report = pd.DataFrame([words]).T
    report.columns = ['word']
    for i in sounds_encoding_methods:
        print(i.__name__)
        report[i.__name__] = report['word'].apply(lambda x: i(x))
    #print(report)

    """Select the closer by algorithm
    for instance levenshtein_distance"""
report2 = pd.DataFrame([words]).T
report2.columns = ['word']

report.set_index('word', inplace=True)
report2 = report.copy()
for sounds_encoding in sounds_encoding_methods:
        report2[sounds_encoding.__name__] = np.nan
        matched_words = []
        for word in words:
            closest_list = []
            for word_2 in words:
                if word != word_2:
                    closest = {}
                    closest['word'] = word_2
                    closest['similarity'] =levenshtein_distance(report.loc[word, sounds_encoding.__name__]
                                         ,report.loc[word_2, sounds_encoding.__name__])
                    closest_list.append(closest)

            report2.loc[word, sounds_encoding.__name__] = pd.DataFrame(closest_list). \
                sort_values(by='similarity').head(1)['word'].values[0]

#print(report2)


def calculate_precision(res, gold_standard):
    true_pos = 0
    for item, x in res.items():
        for i, j in mappings.items():
            if i == '01':
                for k in j:

                    if k == str(item):
                        print(k)
                        true_pos += 1
    print(true_pos)
    return float(true_pos) / float(len(res.items()))



