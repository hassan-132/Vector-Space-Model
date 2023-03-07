import copy
import pickle
from ast import Index
from os import terminal_size
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer    
lem = nltk.WordNetLemmatizer()
alpha = 0.001
result = {}
tokens = []
doclen = 0
index = {}
docfreq = {}
inverted = {}
tfidf = {}
idf = {}
docvector ={}
df = 0
pos = 1
#cleaning tokens
def datacleaning(x):
    if '.' in x :
        x = x.replace('.','')
        return x
    elif '_' in x :
        x = x.replace('_','')
        return x
    elif '+' in x :
        x = x.replace('+','')
        return x
    elif '-' in x :
        x = x.replace('-','')
        return x
    elif '/' in x :
        x = x.replace('/','')
        return x
    else :
        return x
#making inverted Index
def add_if_key_not_exist(dict_obj, key, value):
    if key not in dict_obj:
        inverted.update({key:value})
    else :
        append_value(inverted,key,value)

def append_value(dict_obj, key, value):
    if key in dict_obj:
        if not isinstance(dict_obj[key], list):
            dict_obj[key] = [dict_obj[key]]
        dict_obj[key].append(value)
        dict_obj[key] = list(dict.fromkeys(dict_obj[key]))
f2 = open("Stopword-List.txt",'r')
stp = f2.read()#copying stopword from file to stp
for i in range(448):
    f = open(str(i+1) + ".txt","r")
    t = f.read()#reading file content to t
    t = t.replace("-"," ")
    t = t.replace("/"," ")
    token = word_tokenize(t)#tokenizing elements
    for j in token : # particular term from a list of tokens
        if j == '.' or j == '(' or j == ',' or j == '[' or j == ']' or j == ')' or j == ':' or j==';' or j=='%' or j=='#' or j=='-' or j=='/' or j=='~' or j=='$' or j=='?' or j =='{' or j=='}' or j=='*' or j=='@':
            continue # removal of punctuation
        if j in stp : # if j term in stopword then increment doc freq and continue
            df = df + 1
            continue
        else :
            df = df + 1
            terms = lem.lemmatize(j.lower()) #lemmitizing each term
            terms = datacleaning(terms) #passing terms for cleaning
            add_if_key_not_exist(inverted,terms,i+1)
            if (i+1) not in docvector :
                docvector[i+1] = [terms]
            else :
                docvector[i+1].append(terms)
            if terms not in index:#making a TF dictionary as index
                index[terms] = {i+1 : 1}
            elif (i+1) in index[terms]:
                index[terms][i+1] = index[terms][i+1] + 1
            else :
                index[terms][i+1] = 1
    for key,value in index.items(): # Normalizing the TF as tf / len(doc)
        for key in value :
            if key == i+1 :
                value[key]  = round(float(value[key]/df),5)
    df = 0
    pos = 1
ff=open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/index.pkl",'wb')#storing termfrequency
pickle.dump(index,ff)
ff.close()
ff2=open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/inverted.pkl",'wb')#storing inverted index
pickle.dump(inverted,ff2)
ff2.close()
for key in inverted :# calculating IDF
    if not isinstance(inverted[key], list):
            inverted[key] = [inverted[key]]
    idf[key] = round(math.log(448/len(inverted[key]),10),5)
ff3=open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/idf.pkl",'wb')#storing IDF
pickle.dump(idf,ff3)
ff3.close()
tfidf = copy.deepcopy(index)
for key,value in tfidf.items():# making tfidf dictionary {term : {docid : TF * idf},.....} 
    for k in value :
        value[k] = round((value[k] * idf[key]),5)
ff4=open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/tfidf.pkl",'wb')#storing TFIDF
pickle.dump(tfidf,ff4)
ff4.close()
