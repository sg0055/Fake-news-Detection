# -*- coding: utf-8 -*-


! pip install -q kaggle

from google.colab import files

files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets list

! pip install update kaggle

! kaggle competitions download -c fake-news

#!pip install -U -q PyDrive
#from colab_util import *
!cp "/content/train.csv" -r "/content/drive/My\Drive/"

!unzip /content/train.csv.zip
!unzip /content/test.csv.zip

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



df=pd.read_csv('/content/train.csv')
#df.dropna()

print(df.shape)
#df.head()
#df['label']=df['label'].replace([0,1],['Real','Fake'])
df.head()

labels=df.label

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
#x_train.dropna()
#x_test.dropna()

shape(x_train)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train.apply(lambda x: np.str_(x))) 
tfidf_test=tfidf_vectorizer.transform(x_test.apply(lambda x: np.str_(x)))

passc=PassiveAggressiveClassifier(max_iter=50)
model=passc.fit(tfidf_train,y_train)

model.summary()

y_pred=passc.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

print(df['text'][3])

"""Pipeline 

"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values=np.nan, strategy='constant')
x_train_preprocess = x_train.apply(lambda x: np.str_(x))

pipeline = Pipeline(steps=[('tfidf',TfidfVectorizer(stop_words='english', max_df=0.7)),
                           ('model',PassiveAggressiveClassifier(max_iter=50))])
pipeline.fit(x_train_preprocess,y_train)

y_pred=pipeline.predict(x_test.apply(lambda x: np.str_(x)))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

from joblib import dump

dump(pipeline, filename='news_classification.joblib')

from joblib import load

pipeline2 = load('news_classification.joblib')

y_pred=pipeline2.predict(x_test.apply(lambda x: np.str_(x)))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm=confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

import matplotlib.pyplot as plt
plt.hist(y_test)

plt.hist(y_pred)

"""

---
**Kaggle Submission**


---

"""

ka_test=pd.read_csv('/content/test.csv')
ka_test1=ka_test['text']

ka_test.head()

tf_test=tfidf_vectorizer.transform(ka_test1.apply(lambda x: np.str_(x)))

submit=passc.predict(tf_test)
plt.hist(submit)

data=[]
for i in range(0,5200):
  data=data+[[ka_test['id'][i],submit[i]]]

submit2=pd.DataFrame(data, columns = ['id', 'label']) 
#print(data)

np.savetxt('submit.csv',submit2, delimiter=',', fmt='%d', header='id,label', comments='')

features=tfidf_vectorizer.get_feature_names()

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):

  row = np.squeeze(Xtr[row_id].toarray())
  return top_tfidf_feats(row, features, top_n)

print(top_feats_in_doc(tfidf_train,feasures,0))

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_feats_in_doc(Xtr, features,1, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

plot_tfidf_classfeats_h(top_feats_by_class(tfidf_train,labels,feasures))

"""TF-IDF CODES

"""

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

python --version





"""Fake news  detection using Neural Networks

"""

import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K 
K.clear_session()

df=pd.read_csv('/content/train.csv')
df = df.fillna(' ')
df.count()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index
vocab_size=len(word_index)
print(vocab_size)

# Padding data

sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

split = 0.3
split_n = int(round(len(padded)*(1-split),0))

train_data = padded[:split_n]
train_labels = df['label'].values[:split_n]
test_data = padded[split_n:]
test_labels = df['label'].values[split_n:]

labels=df.label
train_data, test_data, train_labels, test_labels = train_test_split(padded, labels, test_size=0.2, random_state=7)

print(len(test_labels))
print(len(padded))

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \
    -O /tmp/glove.6B.100d.txt
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;
print(len(coefs))

embeddings_matrix = np.zeros((vocab_size+1, 100));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 100, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(type(train_data))

#tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(train_data, train_labels, epochs=5, batch_size=100, validation_data=(test_data, test_labels))

print("Training Complete")

prediction= model.predict(test_data)

pred=[]
for p in prediction:
  if p>=0.5:
    pred+=[1]
  else:
    pred+=[0]

from sklearn.metrics import accuracy_score,confusion_matrix
score=accuracy_score(test_labels,pred)
print(f'Accuracy: {round(score*100,2)}%')

cm=confusion_matrix(test_labels,pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()



"""Kaggle Submission

"""

ka_test=pd.read_csv('/content/test.csv')
ka_test = ka_test.fillna(' ')


ka_test.head()

tokenizer.fit_on_texts(ka_test['text'])
word_index = tokenizer.word_index
vocab_size=len(word_index)
print(vocab_size)

# Padding data

sequences = tokenizer.texts_to_sequences(ka_test['text'])
padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
