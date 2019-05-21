"""
LSTM model for mortgage document classification
Created by Ashwini Tonge
"""
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.externals import joblib

from LSTM_Seq import LSTMSEQ



def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion) #should be int
    print(ratio)
    X_train = matrix[ratio:]
    X_test =  matrix[:ratio]
    Y_train = target[ratio:]
    Y_test =  target[:ratio]
    return X_train, X_test, Y_train, Y_test


# load Train dataset
df = pandas.read_csv("document-classification-test-master/shuffled-full-set-hashed.csv")
#dataset = df.values

print(df.head())
print(df.info())


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 58627
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
df.document_text = df.document_text.astype(str)
tokenizer.fit_on_texts(df['document_text'].values)
joblib.dump(tokenizer, "Sequence_model/token.pkl")
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['document_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
print('Shape of data tensor:', X.shape)

Y = pandas.get_dummies(df['document_type']).values
Y_h = pandas.get_dummies(df['document_type'])
print('Shape of label tensor:', Y.shape)
print(Y_h.head())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
#model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(14, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)

epochs = 12
batch_size = 64

#history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#joblib.dump(model, "Sequence_model/lstm.pkl")
model.save("Sequence_model/lstm.h5")

