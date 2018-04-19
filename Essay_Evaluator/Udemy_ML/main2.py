from collections import Counter
import pandas
from nltk.corpus import stopwords
import numpy
from sklearn.linear_model import LogisticRegression
contents = pandas.read_csv("train1.csv",encoding="ISO-8859-1")
headlines = contents['essay']
# headlines = [
#     "PretzelBros, airbnb for people who like pretzels, raises $2 million",
#     "Top 10 reasons why Go is better than whatever language you use.",
#     "Why working at apple stole my soul (I still love it though)",
#     "80 things I think you should do immediately if you use python.",
#     "Show HN: carjack.me -- Uber meets GTA"
# ]

# Find all the unique words in the headlines.
unique_words = list(set(" ".join(headlines).split(" ")))
def make_matrix(headlines, vocab):
    matrix = []
    for headline in headlines:
        # Count each word in the headline, and make a dictionary.
        counter = Counter(headline)
        # Turn the dictionary into a matrix row using the vocab.
        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pandas.DataFrame(matrix)
    df.columns = unique_words
    return df

# print(make_matrix(headlines, unique_words))

import re

# Lowercase, then replace any non-letter, space, or digit character in the headlines.
new_headlines = [re.sub(r'[^\w\s\d]','',h.lower()) for h in headlines]
# Replace sequences of whitespace with a space character.
new_headlines = [re.sub("\s+", " ", h) for h in new_headlines]

#############################################################3
unique_words = list(set(" ".join(new_headlines).split(" ")))
# We've reduced the number of columns in the matrix a bit.
# print(make_matrix(new_headlines, unique_words))

unique_words = list(set(" ".join(new_headlines).split(" ")))
# Remove stopwords from the vocabulary.
unique_words = [w for w in unique_words if w not in stopwords.words('english')]

# We're down to 34 columns, which is way better!
# print(make_matrix(new_headlines, unique_words))

from sklearn.feature_extraction.text import CountVectorizer

# Construct a bag of words matrix.
# This will lowercase everything, and ignore all punctuation by default.
# It will also remove stop words.
vectorizer = CountVectorizer(lowercase=True, stop_words="english")

matrix = vectorizer.fit_transform(headlines)
# We created our bag of words matrix with far fewer commands.
# print(matrix.todense())

# Let's apply the same method to all the headlines in all 100000 submissions.
# We'll also add the url of the submission to the end of the headline so we can take it into account.
# submissions['full_test'] = submissions["headline"] + " " + submissions["url"]
full_matrix = vectorizer.fit_transform(contents["essay"])
# print(full_matrix.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Convert the upvotes variable to binary so it works with a chi-squared test.
col = contents["domain1_score"].copy(deep=True)
col_mean = col.mean()
col[col < col_mean] = 0
col[(col > 0) & (col > col_mean)] = 1

# Find the 1000 most informative columns
selector = SelectKBest(chi2, k=1000)
selector.fit(full_matrix, col)
top_words = selector.get_support().nonzero()

# Pick only the most informative columns in the data.
chi_matrix = full_matrix[:,top_words[0]]

# Our list of functions to apply.
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(contents["essay"].apply(func))

# Convert the meta features to a numpy array.
meta = numpy.asarray(columns).T
columns = []

# Convert the submission dates column to datetime.
# submission_dates = pandas.to_datetime(submissions["submission_time"])

# Transform functions for the datetime column.
transform_functions = [
    lambda x: x.year,
    lambda x: x.month,
    lambda x: x.day,
    lambda x: x.hour,
    lambda x: x.minute,
]

# Apply all functions to the datetime column.
# for func in transform_functions:
#     columns.append(submission_dates.apply(func))

# Convert the meta features to a numpy array.
non_nlp = numpy.asarray(columns).T

# Concatenate the features together.
features = numpy.hstack([meta, chi_matrix.todense()])

# Our list of functions to apply.
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(contents["essay"].apply(func))

# Convert the meta features to a numpy array.
meta = numpy.asarray(columns).T

features = numpy.hstack([meta, chi_matrix.todense()])

from sklearn.linear_model import Ridge
import numpy
import random

train_rows = 1200
# Set a seed to get the same "random" shuffle every time.
random.seed(1)

# Shuffle the indices for the matrix.
indices = list(range(features.shape[0]))
random.shuffle(indices)

# Create train and test sets.
train = features[indices[:train_rows], :]
test = features[indices[train_rows:], :]
train_upvotes = contents["domain1_score"].iloc[indices[:train_rows]]
test_upvotes = contents["domain1_score"].iloc[indices[train_rows:]]
train = numpy.nan_to_num(train)

# Run the regression and generate predictions for the test set.
reg = LogisticRegression()
reg.fit(train, train_upvotes)
predictions = reg.predict(test)
# print(predictions)
i = 0
for test in test_upvotes:
    print(str(test)+" | "+str(predictions[i]))
    i = i + 1


