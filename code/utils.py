import os
os.chdir('/scratche/home/aditya/harshita/scratch/mlld/a2/ray_logistic/')
import re, unicodedata
import nltk
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pickle
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import ray
import time
import json
# import progressbar

print("Libraries imported")

#################################

lr_variant = 'constant'
if lr_variant =='constant':
    file_folder = '../models/local_const_lr/'
elif lr_variant =='increasing':
    file_folder = '../models/local_inc_lr/'
elif lr_variant =='decreasing':
    file_folder = '../models/local_dec_lr/'

mu = 50e-6
lr = 0.005
epochs = 2000
use_reg = False
use_sparse = True

use_tiny = False


if use_sparse:
    DATA_VECTORS_FILE_NAME = "../logistic_full/processed_data/vectors_sparse_train_full.txt"
    VALID_DATA_VECTORS_FILE_NAME = "../logistic_full/processed_data/vectors_sparse_test_full.txt"
    ALPHA = lr
    VOCAB_FILE_NAME = "../logistic_full/code/vocab_full.txt"
    LAMBDA = mu
    MAX_EPOCHS = epochs

    vocab = {}
    f = open(VOCAB_FILE_NAME, "r")
    for line in f.readlines():
        line = line.strip()
        splitLine = line.split('\t', 2)
        word = splitLine[0]
        wordId = int(splitLine[1])
        vocab[word] = wordId
    f.close()

    VOCAB_SIZE = len(vocab)
        # sparse vector data with class
    data = []
    file = open(DATA_VECTORS_FILE_NAME, "r")
    NUM_INSTANCE_TO_PROCESS = 3000000 # if this exceeds no. of lines in file its still ok
    count = 0
    for line in file.readlines():
        line = json.loads(line)
        data.append(line)
        count = count + 1
        if(count > NUM_INSTANCE_TO_PROCESS):
            break
    NUM_INSTANCE_TO_PROCESS = count
    file.close()

    # validation data
    valid_data = []
    file = open(VALID_DATA_VECTORS_FILE_NAME, "r")
    NUM_INSTANCE_TO_PROCESS_VALID = 3000000 # if this exceeds no. of lines in file its still ok
    count = 0
    for line in file.readlines():
        line = json.loads(line)
        valid_data.append(line)
        count = count + 1
        if(count > NUM_INSTANCE_TO_PROCESS_VALID):
            break
    NUM_INSTANCE_TO_PROCESS_VALID = count
    file.close()




    print("Data read, %d instances, W length %d. ALPHA = %f LAMBDA = %f MAX_EPOCHS = %d" %(NUM_INSTANCE_TO_PROCESS, VOCAB_SIZE, ALPHA, LAMBDA, MAX_EPOCHS))


    train_data = [ x['vector'] for x in data ]
    train_labels_data = [ x['classes'] for x in data ]
    test_data = [ x['vector'] for x in valid_data ]
    test_labels_data = [ x['classes'] for x in valid_data ]

    labelset = set(sum((train_labels_data+test_labels_data),[]))

    if use_tiny:
        train_data = train_data[0:2000]
        train_labels_data = train_labels_data[0:2000]
        test_data = test_data[0:1000]
        test_labels_data = test_labels_data[0:1000]

    # labelindexer = {}
    # for i, lab in enumerate(labelset):
    #     labelindexer[lab] = i

    # train_labels_data = [ [labelindexer[label] for label in labels] for labels in train_labels_data]
    # test_labels_data = [ [labelindexer[label] for label in labels] for labels in test_labels_data]

    # train_words = sum([list(x.keys()) for x in train_data], [])
    # test_words = sum([list(x.keys()) for x in test_data], [])

    # wordset = set(train_words + test_words)

    # wordindexer = {}
    # for i, word in enumerate(wordset):
    #     wordindexer[word] = i

    # pdb.set_trace()
    # print('1')
    # train_data = [ [wordindexer[word] for word in words] for words in train_data.items()]
    # print('2')
    # test_data = [ [wordindexer[word] for word in words] for words in test_data]
    # print('3')


    num_classes = len(labelset)

else:
    train_path = "../../data/verysmall_train.txt"
    test_path = "../../data/verysmall_test.txt"
    dev_path = "../../data/verysmall_devel.txt"
    supersmall_path  = "../../data/supersmall.txt"


    vocab = pickle.load(open('vocab.p','rb'))
    labelindexer = pickle.load(open('labels.p','rb'))

    print("vocab has %s words" % len(vocab))
    print("there are %s labels" % len(labelindexer))

    files = np.load(open('processed_data','rb'))
    train_data, train_labels_data, test_data, test_labels_data, dev_data, dev_labels_data = files['train_data'], files['train_labels_data'], files['test_data'], files['test_labels_data'], files['dev_data'], files['dev_labels_data']
    train_labels_data = train_labels_data.reshape(-1,1)
    test_labels_data = test_labels_data.reshape(-1,1)
    dev_labels_data = dev_labels_data.reshape(-1,1)

    VOCAB_SIZE = len(vocab)
    num_classes = len(labelindexer)

#################################
print("Train rows: %s, test rows: %s" % (len(train_data), len(test_data)))


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_quotes(text):
    return re.sub('"', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_quotes(text)
    return text
#############################################

def save_file(path, data):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # pdb.set_trace()
        pickle.dump(data, open(path, 'wb'))

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def extract_serial(i, line, labels_data, train_data):
    processedtext1 = denoise_text(line)
    tabsplit = processedtext1.split("\t")
    processedwords = nltk.word_tokenize(tabsplit[1])
    processedtext = normalize(processedwords)

    data = np.zeros(len(vocab))
    for word in processedtext:
        if word not in vocab.keys():
            print('word not found' + str(word))
        else:
            data[vocab[word]]+=1
    print(i)

    # labels = nltk.word_tokenize(tabsplit[0])
    labels = set([x.strip() for x in tabsplit[0].split(',')])
    labels_data[i] = [labelindexer[label] for label in labels]
    train_data[i] = data

# def extract(i, line, labels_data, train_data):
@ray.remote(num_return_vals=2)
def extract(line):
    processedtext1 = denoise_text(line)
    tabsplit = processedtext1.split("\t")
    processedwords = nltk.word_tokenize(tabsplit[1])
    processedtext = normalize(processedwords)
    data = np.zeros(len(vocab))
    for word in processedtext:
        if word not in vocab.keys():
            print('word not found' + str(word))
        else:
            data[vocab[word]]+=1
    # labels = nltk.word_tokenize(tabsplit[0])
    labels = set([x.strip() for x in tabsplit[0].split(',')])
    labels = [labelindexer[label] for label in labels]

    train = data
    return train, labels


def read_data(path, lines = None, using_ray = False):
    # file = open("/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt","r")
    file = open(path,"r")

    text = file.read()
    splittext = text.splitlines()
    if lines is not None:
        splittext = splittext[0:lines]


    # bar = progressbar.ProgressBar(maxval=len(splittext), \
    # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # <<<<< Insert Code here >>>>>>
    # bar.finish()
    
    
    if using_ray:

        all_data = ray.get(sum([extract.remote(line) for line in (splittext)], []))
        train_data = all_data[0::2]
        labels_data = all_data[1::2]
        # ray.get([extract.remote(line) for line in splittext])
    else:
        train_data = ['' for _ in splittext]
        labels_data = [[] for _ in splittext]
        [extract_serial(i, line, labels_data, train_data) for i, line in enumerate(splittext)]
    return train_data, labels_data

######################### ONE D and TWO D MATRIX OPERATIONS ##########################

def sparse_mult_dot(normal, sparse):

    dense_out = np.zeros((num_classes, VOCAB_SIZE))
    for i in range(0, num_classes):
        for key, value in sparse.items():
                key = int(key)
                value = float(value)
                dense_out[i][key] = normal[i]*value
        # if 
    return dense_out

# def sparse_scalar_mult(sparse, k):
#     sparse2 = sparse.copy()
#     for key in sparse2:
#         sparse2[key] *= k
#     return sparse2

# def sparse_subtract(normal, sparse):
#     for key, value in sparse.items():
#         key = int(key)
#         value = float(value)
#         normal[key] -= value
#     return normal

def sparse_mult(normal, sparse):
    # pdb.set_trace()
    ans_classes = []
    ans = 0.0
    for i in range(0,num_classes):
        for key, value in sparse.items():
            key = int(key)
            value = float(value)
            ans = ans + normal[i][key]*value
        ans_classes.append(ans)
    return ans_classes

def sparse_scalar_mult(sparse, k):
    pdb.set_trace()
    sparse2 = sparse.copy()
    for key in sparse2:
        sparse2[key] *= k
    return sparse2

def sparse_subtract(normal, sparse):
    for i in range(0,num_classes):
        for key, value in sparse.items():
            key = int(key)
            value = float(value)
            normal[i][key] -= value
    return normal

########################### USING SPARSE MATRIX OPERATIONS ########################

if use_sparse == True:
    def rmse(x):
        return np.sqrt(np.sum(np.dot(x, x.T)))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) 

    def cross_entropy(yhat, y):
        # pdb.set_trace()
        return - np.sum( (y * np.log(yhat+1e-6)) )

    def calcgradient(x, y, yhat, mu, w):
        grad = - sparse_mult_dot((y-yhat).reshape(-1,1), x)
        # pdb.set_trace()
        return grad

    def sgd(x, y, yhat, W, lr, mu, t):
        if lr_variant == 'increasing':
            lr = 0.00001 * np.log(t+1) * lr
        elif lr_variant == 'decreasing':
            lr =  np.exp(-1 * (t+1)) * lr
        
        grad = calcgradient(x, y, yhat, mu, W)
        update = grad*lr

        if use_reg:
            update = update + 2*lr*mu*W

        # print(rmse(update), rmse(-2*lr*mu*W))
        #     # print("param sum: %s, grad update: %s, reg penalty: %s" %(rmse(param), rmse(lr * grad), rmse(2 * lr * mu * param)) )

        return update

    def model(X, W):
        yhat = sparse_mult(W, X)
        yhat = softmax(yhat)
        return yhat

    def evaluate_accuracy(data, model, W, labels_data):
        numerator = 0.
        # data = data.as_in_context(model_ctx).reshape((-1,784))
        # label = label.as_in_context(model_ctx)
        # label_one_hot = nd.one_hot(label, 10)

        output = []
        for i, d in enumerate(data):
            output.append(model(d, W))

        predictions = np.argmax(output, axis=1)
        
        # Uncomment below if you pass one-hot directly
        # label_data = np.argmax(label_data, axis = 1)

        numerator += np.sum([pred in labels_data[i] for i, pred in enumerate(predictions)])
        return (numerator / len(data))


########################### USING DENSE MATRIX OPERATIONS ###########################

# if use_sparse == False:
#     def rmse(x):
#         return np.sqrt(np.sum(np.dot(x, x.T)))

#     def softmax(x):
#         e_x = np.exp(x - np.max(x))
#         return e_x / e_x.sum(axis=0) 

#     def cross_entropy(yhat, y):
#         # pdb.set_trace()
#         return - np.sum( (y * np.log(yhat+1e-6)) )

    def l2reg(W, mu):
        return - mu * np.sum(np.square(W))

#     def calcgradient(x, y, yhat, mu, w):
#         grad = - np.dot((y-yhat).reshape(-1,1), x.reshape(1,-1))
#         # pdb.set_trace()
#         return grad

#     def calcgradient_reg(x, y, yhat, mu, w):
#         grad = - (np.dot((y-yhat).reshape(-1,1), x.reshape(1,-1)) + mu*w)
#         # print(grad.shape)
#         return grad

#     def sgd(x, y, yhat, W, lr, mu, t):
#         if lr_variant == 'increasing':
#             lr = 0.00001 * np.log(t+1) * lr
#         elif lr_variant == 'decreasing':
#             lr =  np.exp(-1 * (t+1)) * lr
#         if use_reg:
#             grad = calcgradient_reg(x, y, yhat, mu, W)
#             update = -1 * lr * grad - 2 * lr * mu * W
#             # print("param sum: %s, grad update: %s, reg penalty: %s" %(rmse(param), rmse(lr * grad), rmse(2 * lr * mu * param)) )
#         else:
#             grad = calcgradient(x, y, yhat, mu, W)
#             update = -1 * lr * grad
#             # print("param sum: %s, grad update: %s" %(rmse(param), rmse(lr * grad)) )
#         return update

#     def model(X, W):
#         yhat = np.dot(W, X)
#         yhat = softmax(yhat)
#         return yhat

#     def evaluate_accuracy(data, model, W, labels_data):
#         numerator = 0.
#         # data = data.as_in_context(model_ctx).reshape((-1,784))
#         # label = label.as_in_context(model_ctx)
#         # label_one_hot = nd.one_hot(label, 10)

#         output = []
#         for i, d in enumerate(data):
#             output.append(model(d, W))

#         predictions = np.argmax(output, axis=1)
        
#         # Uncomment below if you pass one-hot directly
#         # label_data = np.argmax(label_data, axis = 1)

#         numerator += np.sum([pred in labels_data[i] for i, pred in enumerate(predictions)])
#         return (numerator / len(data))


# ######################################################
