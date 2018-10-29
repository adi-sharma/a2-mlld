import re
import sys
from collections import Counter
import zipimport
# importer = zipimport.zipimporter('nltk.zip')
# nltk = importer.load_module('nltk')
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
TRAIN_DATA_FILE_NAME = "../../data/verysmall_test.txt"
TEST_DATA_FILE_NAME = "../../data/verysmall_test.txt"
# DATA_FILE_NAME = "data/DBPedia.full.hdfs/full_train.txt"
STOPWORDS_FILE_NAME = "stopwords.txt"
# VOCAB_FILE_NAME = "vocab_verysmall.txt"
VOCAB_FILE_NAME = "vocab_full.txt"
CLASSES_FILE_NAME = "classes.txt"

def remove_till_first_quote(text):
    regex = r"^(.*?)\""
    text = re.sub(regex, '', text)
    return text

def remove_unicode(text):
    """Replace unicode codes like uxxxx with space"""
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


dataFile = open(TRAIN_DATA_FILE_NAME, "r")
dataFile2 = open(TEST_DATA_FILE_NAME, "r")
# allWords = set()
allClasses = set()
wordcountdict = {}
for line in dataFile.readlines()+dataFile2.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    classes = splitLine[0].split(',')
    for docClass in classes:
        docClass = docClass.strip()
        allClasses.add(docClass)
    document = splitLine[1]
    document = denoise_text(document)
    words = document.split()
    words = to_lowercase(words)
    for word in words:
        if word in stopwords:
            continue
        # allWords.add(stemmer.stem(word))
        word = stemmer.stem(word)
        if word in wordcountdict:
            wordcountdict[word] += 1
        else:
            wordcountdict[word] = 1

vocabFile = open(VOCAB_FILE_NAME, "w")
classesFile = open(CLASSES_FILE_NAME, "w")
# allWordsList = list(allWords)
# import pdb; pdb.set_trace()
allWordsList = [word for word in wordcountdict.keys() if wordcountdict[word]>3]
allWordsList.sort()
print(len(allWordsList))
allClassesList = list(allClasses)
allClassesList.sort()
wordId = 0
for word in allWordsList:
    vocabFile.write(word + '\t' + str(wordId) + '\n')
    wordId += 1

classId = 0
for docClass in allClassesList:
    classesFile.write(docClass + '\t' + str(classId) + '\n')
    classId += 1
