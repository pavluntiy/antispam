# coding: utf-8
import pickle

import sys
import base64
import random
import chardet
import codecs
from re import sub
from collections import defaultdict

from nltk.stem import SnowballStemmer
from HTMLParser import HTMLParser

import numpy as np

import zlib

stemmer = SnowballStemmer("russian")

import pymorphy2

from nltk.tokenize import TweetTokenizer

from HTMLParser import HTMLParser

from collections import defaultdict, Counter

from sklearn.feature_extraction import DictVectorizer



model = pickle.load(open("model.pickle", "r"))


tokenizer = TweetTokenizer()

class SpamHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = ""
        self.title_len = 0
        self.a_len = 0
        self.is_title = False
        self.is_a = False
        
        self.a_count = 0
        
        self.is_em = False
        self.em_count = 0
        self.em_len = 0
        
        self.script_count = 0
        
        self.a_counter = Counter()
        
        self.h1_count = 0
        self.h2_count = 0
        self.h3_count = 0

        self.bold_count = 0
        self.img_count = 0
        
        self.table_count = 0
        self.p_count = 0
#         self.is_h1 = False
#         self.h1_len = 0
        

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = sub('[ \t\r\n]+', ' ', text)
            self.__text += text + ' '
            
        if self.is_title:
            self.title_len = len(text.split(' '))
            
        if self.is_a:
            tokens = text.split(' ')
            
            # a_tokens = tokenizer.tokenize(text)
            self.a_len += len(tokens)
            self.a_counter.update(tokens)

        if self.is_em:
            tokens = text.split(' ')
            
            # tokens = tokenizer.tokenize(text)
            self.em_len += len(tokens)
            
            

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.is_title = True
            
        if tag == 'table':
            self.table_count += 1
            
        if tag == 'script':
            self.script_count += 1
            
        if tag == 'h1':
            self.h1_count += 1
            
        if tag == 'h2':
            self.h2_count += 1
            
        if tag == 'h3':
            self.h3_count += 1

        if tag == 'img':
            self.img_count += 1

        if tag == 'b':
            self.bold_count += 1

        if tag == 'p':
            self.p_count += 1
            
        if tag == 'a':
            self.is_a = True
            self.a_count += 1
            
        if tag == 'em':
            self.is_em = True
            self.em_count += 1
        
        if tag == 'p':
            self.__text += '\n\n'
        elif tag == 'br':
            self.__text += '\n'
            
            
#     def handle_data(self, data):
        
            
    def handle_endtag(self, tag):
        if tag == 'title':
            self.is_title = False
            
        if tag == 'a':
            self.is_a = False
            
        if tag == 'em':
            self.is_em = False

    def handle_startendtag(self, tag, attrs):
        
            
        if tag == 'a':
            self.is_a = False
            
        if tag == 'br':
            self.__text  += '\n\n'
            
    def get_stat(self):
        res = dict(
            title_len = self.title_len,


            a_len = self.a_len,
            a_av_len = self.a_len * 1.0 / (self.a_count or 1.0),
            a_count = self.a_count,

            em_len = self.em_len,
            em_av_len = self.em_len * 1.0 / (self.em_count or 1.0),
            em_count = self.em_count,

            script_count = self.script_count,

            a_unique_words = len(self.a_counter),

            h1_count = self.h1_count,
            h2_count = self.h2_count,
            h3_count = self.h3_count,

            table_count = self.table_count,
            bold_count = self.bold_count,

            img_count = self.img_count,
            p_count = self.p_count
            
        )
        
        return res
        

    def text(self):
        return ''.join(self.__text).strip()

def read_file(DATA_FILE, max_count = None):
    res = []

    with open (DATA_FILE) as df:
        for doc_cnt, line in enumerate(df):
            line = line.strip()
            # делим по \t
            parts = line.split()
            # класс примера 0 || 1
            item_class = int(parts[1])
            url = parts[2]
            pageInb64 = parts[3]

            html = base64.b64decode(pageInb64).decode('utf-8') 

            # stat, tokens = process_doc(pageInb64, url)
            
            
            res.append((html, url, item_class))
            sys.stdout.write("\r{0}".format(doc_cnt))  

            if max_count is not None and doc_cnt >= max_count:
                break
            
            
        return res

def process_docs(docs, max_count  = None):
    res = []
    doc_cnt = 0
    for doc_cnt, doc in enumerate(docs):
        html = doc[0]
        url = doc[1]
        item_class = doc[2]
        stat, tokens = process_doc(html, url)

        res.append((item_class, stat, url, tokens, html))

        sys.stdout.write('\r' + "%s" % (doc_cnt))  

        if max_count is not None and doc_cnt >= max_count:
            break

    return res



def process_doc(html, url):


    parser = SpamHTMLParser()

    parser.feed(html)
    text = parser.text()

    if len(parser.text()) > 0:
        tokens = parser.text().split(' ')
    else:
        tokens = tokenizer.tokenize(parser.text())

    stat = parser.get_stat()

    stat["text_to_html_ratio"] = 1.0 * len(text)/len(html)
        
    compressed_html = zlib.compress(html.encode("utf-8"))
    stat["compressed_to_html"] = 1.0 * len(compressed_html)/len(html.encode("utf-8"))
    
    compressed_text = zlib.compress(text.encode("utf-8"))
    stat["compressed_to_text"] = 1.0 * len(compressed_text)/len(text.encode("utf-8"))

    stat["a_ratio"] = stat["a_len"]/len(text)

    stat["word_len"] = reduce(lambda acc, w: len(w) * 1.0 + acc, tokens, 0.0)/len(tokens)
    stat["word_cnt"] = len(tokens)

    stat["text_len"] = len(text)

    words = Counter(tokens)
    
    stat["a_unique_words_ratio"] = stat["a_unique_words"] * 1.0/len(words)
    stat["unique_words_ratio"] = len(words) * 1.0/len(tokens)

    
    return stat, tokens      

processed = 0

def is_spam(pageInb64, url):
    html = base64.b64decode(pageInb64).decode('utf-8') 
    raw_x, tokens =  process_doc(html, url)
    dict_vectorizer = DictVectorizer(sparse=False)
    # print raw_x
    x = dict_vectorizer.fit_transform(raw_x) 

    global processed
    sys.stdout.write("\rPredicting {0}".format(processed)) 
    processed += 1
    try:
        return model.predict(x)[0]
    except:
        print raw_x
        raise
