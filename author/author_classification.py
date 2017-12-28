import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import random
import pickle
from collections import Counter
import os
import pandas as pd
import tensorflow as tf
import reprlib

def pprint(s):
    print(reprlib.repr(s))

stop_words = set(stopwords.words('english'))
tp = .1
auth_index = {
    "EAP": 0,
    "MWS": 1,
    "HPL": 2
    }

def create_lexicon(table, min_thresh, max_thresh):
    lex = []
    for index, row in table.iterrows():
        text = row['text']
        words = word_tokenize(text)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        lex += list(words)
    lex = [WordNetLemmatizer().lemmatize(x) for x in lex]
    counts = Counter(lex)

    final_lex = []

    for word in counts:
        if max_thresh > counts[word] > min_thresh:
            final_lex.append(word)

    return final_lex


def create_train_feature_and_labels(table, lexicon):
    features = []
    for index, row in table.iterrows():
        text = row['text']
        auth = auth_index[row['author']]
        label = [0, 0, 0]
        label[auth] = 1
        
        words = word_tokenize(text.lower())
        words = [WordNetLemmatizer().lemmatize(x) for x in words if x.isalpha()]
        f = np.zeros(len(lexicon))
        for word in words:
            w = word.lower()
            if w in lexicon:
                i = lexicon.index(w)
                f[i] += 1
        if np.sum(f) == 0:
            print("zero vector")
            print(text)
            f = list(f)
        else:
            f = list(np.multiply(1.0/np.sum(f), f))
            
        features.append([f, label])
    random.shuffle(features)
    features = np.array(features)
    return features

def create_kaggle_features(table, lexicon):
    features = []
    for index, row in table.iterrows():
        text = row['text']
        tag = row['id']

        words = word_tokenize(text.lower())
        words = [WordNetLemmatizer().lemmatize(x) for x in words if x.isalpha()]
        f = np.zeros(len(lexicon))
        for word in words:
            w = word.lower()
            if w in lexicon:
                i = lexicon.index(w)
                f[i] +=1
        if np.sum(f) == 0:
            print("zero vector")
            print(text)
            f = list(f)
        else:
            f = list(np.multiply(1.0/np.sum(f), f))

            
        features.append([tag, f])
    return features


print("reading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
lex = create_lexicon(train, 5, 1000)
print("length of lexicon", str(len(lex)))
features = create_train_feature_and_labels(train, lex)

train_x = list(features[int(len(features)*tp):,0])
train_y = list(features[int(len(features)*tp):,1])

test_x = list(features[:int(len(features)*tp),0])
test_y = list(features[:int(len(features)*tp),1])

#print("creating kaggle array")
#kaggle_data = create_kaggle_features(test, lex)

print("starting neural net part")

n_nodes_hl1 = 9500
n_nodes_hl2 = 9500
#n_nodes_hl3 = 6500
#n_nodes_hl4 = 4500


n_classes = 3
batch_size = 100
hm_epochs = 40

with tf.name_scope("batches"):
    x = tf.placeholder('float')
    y = tf.placeholder('float')

with tf.name_scope("weights-biases"):
    hidden_1_layer = {'f_fum':n_nodes_hl1,
                      'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum':n_nodes_hl2,
                      'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}


    output_layer = {'f_fum':None,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes])),}


def neural_network_model(data):
    with tf.name_scope("layer-1"):
        l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)
    with tf.name_scope("layer-2"):
        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

    with tf.name_scope("output"):
        output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output

kaggle_answers = []
        
def train_neural_network(x):
    prediction = neural_network_model(x)
    with tf.name_scope("optimization"):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.name_scope("testing"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log\\2")
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:   
        sess.run(init)
        writer.add_graph(sess.graph)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                            y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            a, s = sess.run([accuracy, merged_summary], feed_dict={x:test_x, y:test_y})
            writer.add_summary(s, epoch)
            print('Accuracy: ', a)
            
        #for item in kaggle_data:
        #    feature = item[1]
        #    inp = [feature]
        #    ans = sess.run([prediction], feed_dict = {x:inp})
        #    kaggle_answers.append([item[0], ans])

	    
train_neural_network(x)

sub = pd.DataFrame(index=np.arange(0, len(kaggle_answers)), columns=['id', 'EAP', 'HPL', 'MWS'])

for index, item in enumerate(kaggle_answers):
    tag = item[0]
    clas = item[1][0][0]
    m = min(clas)
    if m < 0:
        clas = [1 + c - m for c in clas]
            
    clas = [c/sum(clas) for c in clas]
    sub.loc[index] = [tag, clas[0], clas[2], clas[1]]

sub.to_csv('submission.csv', index=False)
