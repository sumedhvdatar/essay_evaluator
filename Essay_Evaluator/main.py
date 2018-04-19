import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression,SGDRegressor,LinearRegression
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.naive_bayes import MultinomialNB
import gensim
from gensim.models.doc2vec import LabeledSentence
from tqdm import tqdm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import utils
SEED = 2000
class Prepocessing:

    def text_process(self,mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation

        nopunc = [char for char in mess if char not in string.punctuation]

        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        string_array = nopunc.split()
        l = []
        for word in string_array:
            if str.isalpha(word):
                l.append(word)
            else:
                continue

        nopunc = " ".join(str(item) for item in l)

        # Now just remove any stopwords
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    def get_vectors(self,model, corpus, size):
        vecs = np.zeros((len(corpus), size))
        n = 0
        i = 0
        for essay in corpus:
            prefix = 'all_' + str(i)
            vecs[n] = model.docvecs[prefix]
            n += 1
            i = i + 1
        return vecs

    def labelize_tweets_ug(self,tweets, label):
        result = []
        prefix = label
        i = 0
        for t in tweets:
            result.append(LabeledSentence(t, [prefix + '_%s' % i]))
            i = i + 1
        return result

    def process(self,filename,mode):
        contents1 = pd.read_csv(filename, encoding="ISO-8859-1")
        contents = pd.DataFrame(contents1)
        # print(contents['domain1_score'][0])

        # contents_test = pd.read_csv('test1.csv', encoding="ISO-8859-1")
        print(contents['domain1_score'].shape)
        # contents['essay'].to_csv('out.csv')
        bag_of_words = []
        i = 0
        for essay in contents['essay']:
            print(i)
            split_list = self.text_process(essay)
            # print(split_list)
            bag_of_words.append(split_list)
            i = i + 1
        print(bag_of_words)
        x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(bag_of_words,contents['domain1_score'], test_size=.02)
        # print(y_validation_and_test.real)
        all_x_w2v = self.labelize_tweets_ug(bag_of_words, 'all')

        model_ug_dbow = gensim.models.Doc2Vec(dm=0, size=500, negative=5, min_count=2, workers=2, alpha=0.065, min_alpha=0.065)
        model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)]) \

        for epoch in range(60):
            model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
            model_ug_dbow.alpha -= 0.002
            model_ug_dbow.min_alpha = model_ug_dbow.alpha
        print("done")
        train_vecs_dbow = self.get_vectors(model_ug_dbow, x_train, 500)
        validation_vecs_dbow = self.get_vectors(model_ug_dbow, x_validation_and_test, 500)
        # Machine learning
        if mode == "test":
            filename2 = 'finalized_model.pkl'
            loaded_model = pickle.load(open(filename2, 'rb'))
            # print(loaded_model.predict)
            print(essays_tfidf[0])
            print(contents['domain1_score'][0])
            
            result = loaded_model.predict(essays_tfidf[0])
            print(result)
            # all_predictions = essay_score_prediction_model.predict(essays_tfidf)
            # print(all_predictions)
        elif mode == "train":
            clf = LinearRegression()
            clf.fit(train_vecs_dbow, y_train.real)
            predict = clf.predict(validation_vecs_dbow)
            i = 0
            j = 0
            for p in predict:
                if(int(p) == y_validation_and_test.real[i]):
                    j = j + 1
                    i = i + 1
                # print(str(int(p))+" | "+str(y_validation_and_test.real[i]))
                else:
                    i = i + 1
            accuracy = (j/i)*100
            print(accuracy)

p = Prepocessing()
p.process("train1.csv","train")
# p.process("test1.csv","test")


#Text pre-processing



