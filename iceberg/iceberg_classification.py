import reprlib
import psycopg2
import pandas.io.sql as psql
import pandas as pd
import numpy as np
import random
import tensorflow as tf

def pprint(s):
    print(reprlib.repr(s))

run_number = input("run number please: ")
tp = .1
connect_str = "dbname='postgres' user='postgres' host='localhost' " + \
                "password='{}'".format(input("enter postgres password for postgres user:"))

conn = psycopg2.connect(connect_str)
cur = conn.cursor()

#cur.execute("select * from icebergtrain")
#table = cur.fetchall()
#pprint(table)

print("reading the database...")
df = psql.read_sql("select * from icebergtrain", conn)

def create_train_feature_and_labels(table):
    features = []
    for index, row in table.iterrows():
        band1 = row['band1']
        band2 = row['band2']
        label = [0, 0]
        label[row['is_iceberg']] = 1
        
        
        
        features.append([[band1, band2], label])
    random.shuffle(features)
    features = np.array(features)
    return features

print("creating features...")
features = create_train_feature_and_labels(df)

train_x = features[int(len(features)*tp):,0]
train_x = [list(x) for x in train_x]
train_y = list(features[int(len(features)*tp):,1])

test_x = features[:int(len(features)*tp),0]
test_x = [list(x) for x in test_x]
test_y = list(features[:int(len(features)*tp),1])



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


n_classes = 3
batch_size = 100
hm_epochs = 200
keep_rate = 0.95

filters1 = 64
filters2 = 128

with tf.name_scope("placeholders"):
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder('float')
    y = tf.placeholder('float')




def neural_network_model(x):

    with tf.name_scope("weights-biases"):
        weights = {'W_conv1': tf.Variable(tf.random_normal([4,4,2,filters1])),
                    'W_conv2': tf.Variable(tf.random_normal([4,4,filters1,filters2])),
                    'W_fc': tf.Variable(tf.random_normal([19*19*filters2,2048])),
                    'out': tf.Variable(tf.random_normal([2048, 2]))}
        
        biases = {'B_conv1': tf.Variable(tf.random_normal([filters1])),
                  'B_conv2': tf.Variable(tf.random_normal([filters2])),
                  'B_fc': tf.Variable(tf.random_normal([2048])),
                  'out': tf.Variable(tf.random_normal([2]))}
    with tf.name_scope("input"):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, shape=[-1, 75, 75, 2])

    with tf.name_scope("convolution-1"):
        c1 = conv2d(x, weights['W_conv1']) + biases['B_conv1']
        c1 = tf.nn.relu(c1)
        c1 = maxpool2d(c1)
        c1 = tf.nn.dropout(c1, keep_prob)
    with tf.name_scope("convolution-2"):
        c2 = conv2d(c1, weights['W_conv2']) + biases['B_conv2']
        c2 = tf.nn.relu(c2)
        c2 = maxpool2d(c2)
        c2 = tf.nn.dropout(c2, keep_prob)
    with tf.name_scope("fully-connected"):
        fc = tf.reshape(c2, [-1, 19*19*filters2])
        fc = tf.matmul(fc, weights['W_fc'])
        fc = fc + biases['B_fc']
        fc = tf.nn.relu(fc)
        fc = tf.nn.dropout(fc, keep_prob)

    with tf.name_scope("output"):
        output = tf.matmul(fc, weights['out']) + biases['out']

    return output

kaggle_answers = []
        
def train_neural_network(x):
    prediction = neural_network_model(x)
    with tf.name_scope("optimizing"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.name_scope("testing"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log\\" + str(run_number))
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
                #pprint(batch_y)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                              y: batch_y,
                                                              keep_prob: 0.95})
                epoch_loss += c
                i+=batch_size
            
            a, s = sess.run([accuracy, merged_summary], feed_dict={x:test_x, y:test_y, keep_prob: 1.0})
            print("Accuracy: {}".format(a))
            writer.add_summary(s, epoch)
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)


print("starting network part")
train_neural_network(x)


