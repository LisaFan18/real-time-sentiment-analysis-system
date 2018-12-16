import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

# Load raw data
df = pd.read_csv('sentiment140_clean.csv',index_col=0)
#print(df.head(5))
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
#df.info()

# Initialize parameters
SEED = 2000
MODEL_PATH = '../sentiment_training_pipeline/output/dp/'
TRAIN_SIZE = 0.995
TEST_SIZE = 0.5

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

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train,x_val,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')
cores = multiprocessing.cpu_count()

w2v_model_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
w2v_model_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    w2v_model_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    w2v_model_cbow.alpha -= 0.002
    w2v_model_cbow.min_alpha = w2v_model_cbow.alpha

w2v_model_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
w2v_model_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    w2v_model_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    w2v_model_sg.alpha -= 0.002
    w2v_model_sg.min_alpha = w2v_model_sg.alpha

print("start to save model................")

w2v_model_cbow.save(MODEL_PATH + 'w2v_model_cbow')
w2v_model_sg.save(MODEL_PATH + 'w2v_model_sg')

print("end to save model................")