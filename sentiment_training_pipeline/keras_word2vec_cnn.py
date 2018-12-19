import numpy as np
import pandas as pd
import time
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint

# Initialize parameters
SEED = 2000
MAX_SENTENCE_LENGTH = 45
EMBEDDING_DIM = 200 #
MODEL_PATH = '../sentiment_training_pipeline/output/dp/'

#
TRAIN_SIZE = 0.7
TEST_SIZE = 0.5
MAX_NB_WORDS = 50000 # 100000
BATCH_SIZE = 128
EPOCHES = 4

# TRAIN_SIZE = 0.8
# TEST_SIZE = 0.5
# MAX_NB_WORDS = 50000
# BATCH_SIZE = 128
# EPOCHES = 2

# Load raw data
df = pd.read_csv('/home/zhengjuan/project/cs543/stock-prediction-system-master/sentiment_training_pipeline/sentiment140_clean.csv',index_col=0)
print(df.head(6))
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()


# Split data
x = df.text   # input
y = df.target # label
x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x, y, train_size=TRAIN_SIZE, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=TEST_SIZE, random_state=SEED)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_val),
                                                                             (len(x_val[y_val == 0]) / (len(x_val)*1.))*100,
                                                                            (len(x_val[y_val == 1]) / (len(x_val)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


# Preprocess by TOkenizer and Pad_sequence
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# Required before using `texts_to_sequences`
tokenizer.fit_on_texts(x_train)

#compute a sequential representation of each sentence
seq_tr = tokenizer.texts_to_sequences(x_train)
seq_val = tokenizer.texts_to_sequences(x_val)
#len(tokenizer.word_index)

# Find out the maximum length of all sentences for padding
# length = []
# for x in x_train:
#     length.append(len(x.split()))
# max(length) # 40

# Keras zero-pads at the beginning,  padding size is 45
x_train_seq = pad_sequences(seq_tr, maxlen=MAX_SENTENCE_LENGTH)
x_val_seq = pad_sequences(seq_val, maxlen=MAX_SENTENCE_LENGTH)
print('Shape of data tensor:', x_train_seq.shape)


# Load Word2vec models (CBOW_Continuous Bag Of Words, Skip-gram) for word embeddings
w2v_model_cbow = KeyedVectors.load(MODEL_PATH + 'w2v_model_cbow')
w2v_model_sg = KeyedVectors.load(MODEL_PATH + 'w2v_model_sg')
#len(w2v_model_cbow.wv.vocab.keys())

# Combine CBOW and SG
embeddings_index = {}
for w in w2v_model_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(w2v_model_cbow.wv[w],w2v_model_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))


### Build Embedding Matrix
# the most num_words frequent words in the training set
embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# test Matrix
#np.array_equal(embedding_matrix[6] ,embeddings_index.get('you'))


#### Build 1D CNN
tweet_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
# Embedding layer
#embedded_seq = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SENTENCE_LENGTH, trainable=True)(tweet_input)
embedded_seq = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SENTENCE_LENGTH, trainable=False)(tweet_input)

bigram_branch = Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu', strides=1)(embedded_seq)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1)(embedded_seq)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=1)(embedded_seq)
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

model_path="../sentiment_training_pipeline/output/dp/CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

start_time = time.time()
model.fit(x_train_seq, y_train, batch_size=BATCH_SIZE, epochs=EPOCHES,
                     validation_data=(x_val_seq, y_val), callbacks = [checkpoint])
end_time = time.time()

print("Training takes: " , (end_time - start_time)/60, "mins")



# predict
seq_ts = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(seq_ts, maxlen=MAX_SENTENCE_LENGTH)
print("input text only showing top 5 rows  ")
print(x_test[:5])
print("input label only showing top 5 rows  ")
print(y_test[:5])
y_pred = model.predict(x_test_seq)
print("output prediction only showing top 5 rows ", )
print(y_pred[:5])
