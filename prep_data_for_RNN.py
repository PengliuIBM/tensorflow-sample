import nltk
import string
import json
import os
from collections import defaultdict
import pickle

from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer 

#Converting words to their base forms using lemmatization
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()

with open('data/yelp/yelp_academic_dataset_review.json', 'rb') as f:
	reviews = [json.loads(line.lower().strip()) for line in f]
#caculate word frequency as the sequence number for each word
word_freq = defaultdict(int)

if os.path.exists('data/yelp/word_freq.pickle'):
    with open('data/yelp/word_freq.pickle', 'rb') as f:
        word_freq = pickle.load(f)
else:
    for review in reviews:
        words = word_tokenizer.tokenize(review['text'])
        for word in words:
            if word not in stop:
                w = lemmatizer.lemmatize(word)
                word_freq[w] += 1
    with open('data/yelp/word_freq.pickle', 'wb') as f:
        pickle.dump(word_freq,f)

lower = 3
total = 50000
top_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))[:total-lower+1]        
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in top_words:
    vocab[word] = i
    i += 1
print('total vocabulary', i)     
UNKNOW_TOKEN = 0
# Get a balanced sample of positive and negative reviews

# firstly, rule out some less significant reviews to reduce the samples number
#texts = [review['text'] for review['funny'] in (0, 1) for review in reviews]
texts = [review['text'] for review in reviews]

# Convert our 5 classes into 2 (negative or positive)
# to avoid confusion bewteen class0 and padding 0, use class 1 and 2 other than class 0 and 1.
binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
balanced_texts = []
balanced_labels = []
limit = 100000  # Change this to grow/shrink the dataset
neg_pos_counts = [0, 0]
for i in range(len(texts)):
    polarity = binstars[i]
    if neg_pos_counts[polarity] < limit:
        balanced_texts.append(texts[i])
        balanced_labels.append(binstars[i])
        neg_pos_counts[polarity] += 1

###create features sequence
data_x = []
data_y = []
max_sent_in_doc = 30
max_word_in_sent = 30
num_classes = 2
for key, review in enumerate(balanced_texts):
    doc = []
    sents = sent_tokenizer.tokenize(review)
    for i, sent in enumerate(sents):
        if i < max_sent_in_doc:
            word_to_index = []
            for j, word in enumerate(word_tokenizer.tokenize(sent)):
                if j < max_word_in_sent:
                    if word not in stop:
                        w = lemmatizer.lemmatize(word)
                        ex = vocab.get(w, UNKNOW_TOKEN)
                        if ex != UNKNOW_TOKEN:                                           
                            word_to_index.append(ex) 
            doc.append(word_to_index)	
    label = balanced_labels[key]
    labels = [0] * num_classes
    labels[label] = 1
    data_y.append(labels)
    data_x.append(doc)

#print(data_x)
#print(data_y)
pickle.dump((data_x, data_y), open('data/yelp/yelp_data', 'wb'))     





























