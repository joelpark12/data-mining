import numpy as np
import pandas as pd
import math
import sqlalchemy
import psycopg2
import string
import nltk
from nltk.corpus import stopwords
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:@localhost/authors')
#Read data into Pandas data frame
train = pd.read_sql_table('train', engine)
#limit = 20
for index, row in train.iterrows():
    text = nltk.word_tokenize(row['text'])
    #Remove stop words
    new = [word for word in text if word not in stopwords.words('english')]
    new = ' '.join(new)
    new = new.translate(str.maketrans('','',string.punctuation))
    train.set_value(index, 'text',  new)
##    if index >= limit:
##        break
