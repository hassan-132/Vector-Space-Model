from msilib import text
from tkinter import *
from nltk.tokenize import word_tokenize
import nltk
import pickle

from numpy import outer
from pyparsing import White
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
q = ""
file = open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/index.pkl",'rb')#loading termFrequency
index = pickle.load(file)
file.close()
file2 = open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/inverted.pkl",'rb')#loading inverted Index
inverted = pickle.load(file2)
file2.close()
file3 = open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/idf.pkl",'rb')#loading IDF
idf = pickle.load(file3)
file3.close()
file4 = open("C:/Users/SINGAPORE TRADERS/Desktop/Semester 6/Information Retreival/IR Assignment 2/tfidf.pkl",'rb')#loading TFIDF
tfidf = pickle.load(file4)
file4.close()
f2 = open("Stopword-List.txt",'r')
stp = f2.read()#copying stopword from file to stp
query_token = []
tfq = {}
tfidfq = {}
def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb
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
def queryProcessing(q):
    for i in q :
        if i in stp:
            continue
        qr = lem.lemmatize(i.lower())
        qr = datacleaning(qr)
        query_token.append(qr)       #appending tokenized terms in querytoken list 
    for i in query_token :#for calculating frequency of each term in query
        if i not in tfq :
            tfq[i] = 1
        elif i in tfq :
            tfq[i] = tfq[i] + 1
    for i in tfq :
        tfq[i] = tfq[i] / len(q) #normalizing TFQ
        if i not in idf :
            idf[i] = 0
        tfidfq[i] = idf[i] * tfq[i] #multiplying query tf by idf to tfidf for query
    for i in tfidfq : #for procssing final result
        for j in tfidf :
            for k in tfidf[j] :
                if i == j :
                    if i not in result : #Main step multiply tfidf of query with tfidfs term for each docid
                        result[i] = {k : tfidf[j][k] * tfidfq[i] }
                    elif k in result[i] :
                        result[i][k] = tfidf[j][k] * tfidfq[i]
                    else :
                        result[i][k] = tfidf[j][k] * tfidfq[i]
    displayResult(result)
#Displaying Output
def displayResult(result):
    r = []
    for k,v in result.items() :
        for k in v :
            if v[k] >= alpha :#Checking which docs qualify to be presented to user
                r.append(k)
    r = list(set(r))
    r = str(sorted(r))
    output.insert(INSERT,r)
#Getting Query from User
def getquery():
    q = queryval.get()
    q = word_tokenize(q)
    queryProcessing(q)
#GUI Implementation    
root = Tk()
root.geometry("600x300")
root.config(bg=rgb_hack((51,51,255)))
f1 = Frame(root,bg='gold',padx=10,pady=10)
f1.pack(side=TOP)
f2 = Frame(root)
f2.pack(side=BOTTOM)
queryval = StringVar()
Label(f1,text="VSM Search Engine",bg=rgb_hack((0,102,204)),padx=10,pady=10,font = "ubuntu 16 italic",fg='white').pack()
queryentry = Entry(f1,textvariable=queryval,width=50,bg=rgb_hack((0,0,153)),fg='white')
queryentry.pack(padx=10,pady=10)
Button(f1,text="Search",command=getquery,bg='yellow',relief=SUNKEN).pack(pady=10)
output = Text(f2,width=43,height=20,bg=rgb_hack((0,51,102)),fg='white')
output.pack()
root.mainloop()

