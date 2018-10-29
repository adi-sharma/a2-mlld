import re
import sys
import json
import numpy as np
from collections import Counter
# import zipimport
# importer = zipimport.zipimporter('nltk.zip')
# nltk = importer.load_module('nltk')
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')


TRAIN_DATA_FILE_NAME = "../../data/verysmall_train.txt"
TEST_DATA_FILE_NAME = "../../data/verysmall_test.txt"

# DATA_FILE_NAME = "data/DBPedia.full.hdfs/full_test.txt"
STOPWORDS_FILE_NAME = "stopwords.txt"
# VOCAB_FILE_NAME = "vocab_verysmall.txt"
VOCAB_FILE_NAME = "vocab_full.txt"
CLASSES_FILE_NAME = "classes.txt"
TRAIN_DATA_VECTORS_FILE_NAME = "../../data/vectors_sparse_train_verysmall.txt"
TEST_DATA_VECTORS_FILE_NAME = "../../data/vectors_sparse_test_verysmall.txt"

def remove_till_first_quote(text):
    regex = r"^(.*?)\""
    text = re.sub(regex, '', text)
    return text

def remove_unicode(text):
    """Replace unicode codes like xxxx with space"""
    regex = r"(\\u....)"
    text = re.sub(regex, ' ', text)
    return text

def denoise_text(text):
    text = remove_till_first_quote(text)
    text = remove_unicode(text)
    text = remove_punctuation(text)
    return text

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(text):
    """Remove punctuation and replace with space"""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

stopwords = []
f = open(STOPWORDS_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    if(line):
        stopwords.append(line)
f.close()

vocab = {}
f = open(VOCAB_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    word = splitLine[0]
    wordId = int(splitLine[1])
    vocab[word] = wordId
f.close()

classesList = {}
f = open(CLASSES_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    docClass = splitLine[0]
    classId = int(splitLine[1])
    classesList[docClass] = classId
f.close()


dataFile = open(TRAIN_DATA_FILE_NAME, "r")
vectorsFile = open(TRAIN_DATA_VECTORS_FILE_NAME, "w")

for line in dataFile.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    classes = splitLine[0].split(',')
    document = splitLine[1]
    document = denoise_text(document)
    words = document.split()
    words = to_lowercase(words)
    new_words = []
    for word in words:
        if word in stopwords:
            continue
        new_words.append(stemmer.stem(word))

    # create a sparse vector representation
    # key will be wordId of word
    # value will be its count in the document
    sparseVector = {}
    for word in new_words:
        if word in vocab:
            wordId = int(vocab[word])
            if wordId in sparseVector:
                sparseVector[wordId] += 1
            else:
                sparseVector[wordId] = 1
    # need to get classes of each document
    classesOfDoc = []
    for docClass in classes:
        docClass = docClass.strip()
        if docClass in classesList:
            classId = int(classesList[docClass])
            classesOfDoc.append(classId)
    toWrite = {}
    toWrite['classes'] = classesOfDoc
    toWrite['vector'] = sparseVector
    vectorsFile.write(json.dumps(toWrite))
    vectorsFile.write('\n')


vectorsFile.close()
dataFile.close()


dataFile = open(TEST_DATA_FILE_NAME, "r")
vectorsFile = open(TEST_DATA_VECTORS_FILE_NAME, "w")

for line in dataFile.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    classes = splitLine[0].split(',')
    document = splitLine[1]
    document = denoise_text(document)
    words = document.split()
    words = to_lowercase(words)
    new_words = []
    for word in words:
        if word in stopwords:
            continue
        new_words.append(stemmer.stem(word))

    # create a sparse vector representation
    # key will be wordId of word
    # value will be its count in the document
    sparseVector = {}
    for word in new_words:
        if word in vocab:
            wordId = int(vocab[word])
            if wordId in sparseVector:
                sparseVector[wordId] += 1
            else:
                sparseVector[wordId] = 1
    # need to get classes of each document
    classesOfDoc = []
    for docClass in classes:
        docClass = docClass.strip()
        if docClass in classesList:
            classId = int(classesList[docClass])
            classesOfDoc.append(classId)
    toWrite = {}
    toWrite['classes'] = classesOfDoc
    toWrite['vector'] = sparseVector
    vectorsFile.write(json.dumps(toWrite))
    vectorsFile.write('\n')


vectorsFile.close()
dataFile.close()