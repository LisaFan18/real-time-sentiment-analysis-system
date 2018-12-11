import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

plt.style.use('fivethirtyeight')
df = pd.read_csv('sentiment140_clean.csv',index_col=0)
print(df.head(5))
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()
x = df.text
y = df.target
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
# print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
#                                                                              (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
#                                                                             (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
# print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
#                                                                              (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
#                                                                             (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
# print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
#                                                                              (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
#                                                                             (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')
cores = multiprocessing.cpu_count()

model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

print("start to save model................")

#modelPath = '../sentiment_training_pipeline/output/dp/'
model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')

print("end to save model................")