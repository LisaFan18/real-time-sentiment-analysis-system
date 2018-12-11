import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout,Flatten, concatenate, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# load data
df = pd.read_csv('sentiment140_clean.csv',index_col=0)
print(df.head(5))
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()

#Train and Test Sets
x = df.text # input
y = df.target # label
SEED = 2000
# 80% data for training, 20% data for test/validate
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.2, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)



# Sequential text data
# split each word in a sentence
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
#get a sequential representation of each sentence
sequences = tokenizer.texts_to_sequences(x_train)
sequences_val = tokenizer.texts_to_sequences(x_validation)
len(tokenizer.word_index)

# Find out the maximum length of all sentences for padding
# length = []
# for x in x_train:
#     length.append(len(x.split()))
# max(length) # 40

# Keras zero-pads at the beginning,  padding size is 45
x_train_seq = pad_sequences(sequences, maxlen=45)
x_val_seq = pad_sequences(sequences_val, maxlen=45)
print('Shape of data tensor:', x_train_seq.shape)
x_train_seq[:5]




# word2vector
# load 2 word2vector model to extract word vector forms
# CBOW means Continuous Bag Of Words
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
# skip-gram model
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')
#len(model_ug_cbow.wv.vocab.keys())

# combine two word2vector models so that each word will have 200 dimension vector representation
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))


### Build a Matrix
# the most num_words frequent words in the training set
num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# test Matrix
np.array_equal(embedding_matrix[6] ,embeddings_index.get('you'))


####  CNN
tweet_input = Input(shape=(45,), dtype='int32')

tweet_encoder = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()

model_path="../sentiment_training_pipeline/output/CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train_seq, y_train, batch_size=32, epochs=5,
                     validation_data=(x_val_seq, y_validation), callbacks = [checkpoint])

# reloaded model
# loaded_CNN_model = load_model('CNN_best_weights.02-0.8333.hdf5')
# loaded_CNN_model.evaluate(x=x_val_seq, y=y_validation)