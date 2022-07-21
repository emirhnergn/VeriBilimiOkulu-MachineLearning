#%%
text = """
A Scandal in Bohemia! 01
The Red-header League, 2
A Case, of Identity 33
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

text


#%%
# Test Preprocessing
import pandas as pd
from warnings import filterwarnings
import nltk
filterwarnings("ignore")

# String to DF
v_text = text.split("\n")
v = pd.Series(v_text)
text_vector = v[1:len(v)]
mdf = pd.DataFrame(text_vector, columns = ["Stories"])
mdf

# Upper / Lower Case
d_mdf = mdf.copy()
list1 = [1,2,3]
str1 = " ".join(str(i) for i in list1)
str1

d_mdf["Stories"].apply(lambda x : " ".join(x.lower() for x in x.split()))
d_mdf2 = d_mdf["Stories"].apply(lambda x : " ".join(x.lower() for x in x.split()))

# Punctuation
d_mdf2 = d_mdf2.str.replace("[^\w\s]","")

# Remove Numeric
d_mdf2 = d_mdf2.str.replace("\d","")

# Stopwords
from nltk.corpus import stopwords
sw = stopwords.words("english")
d_mdf3 = d_mdf2.apply(lambda x : " ".join(x for x in x.split() if x not in sw))

# Remove Low Usage Words
delete = pd.Series(" ".join(d_mdf3).split()).value_counts()[-2:]
d_mdf4 = d_mdf3.apply(lambda x : " ".join(x for x in x.split() if x not in delete))

# Tokenization
import textblob
from textblob import TextBlob
d_mdf4.apply(lambda x : TextBlob(x).words)

# Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
d_mdf4.apply(lambda x : " ".join(st.stem(i) for i in x.split()))

# Lemmatization
from textblob import Word
d_mdf5 = d_mdf4.apply(lambda x : " ".join(Word(i).lemmatize() for i in x.split()))
#%%
# N-Gram
text = """Backgammon is one of the oldest known board games.
 Its history can be traced back nearly 5,000 years to archeological 
 discoveries in the Middle East. It is a two player game where each player has 
 fifteen checkers which move between twenty-four points according 
 to the roll of two dice."""
TextBlob(text).ngrams(3)

# Part of Speech Tagging
d_mdf5.apply(lambda x : TextBlob(x).tags)

# Chunking (Shallow Parsing)
pos = TextBlob(text).tags
reg_exp = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_exp)
results = rp.parse(pos)
#print(results)
#results.draw()

# Named Entity Recognition
from nltk import word_tokenize, pos_tag, ne_chunk
print(ne_chunk(pos_tag(word_tokenize(text))))



#%%
# Letter / Character Count
b_mdf = d_mdf.copy()

b_mdf["Letter Count"] = b_mdf["Stories"].str.len()
b_mdf

# Word Count
a = "scandal in a bohemia"
len(a.split())

b_mdf["Word Count"] = b_mdf["Stories"].apply(lambda x : len(str(x).split(" ")))

# Special Characters
b_mdf["Special Characters"] =  b_mdf["Stories"].apply(lambda x :len([x for x in x.split() 
                                      if x.startswith("Adventure")]))

# Catching Numeric Values
b_mdf["Numeric Characters"] =  b_mdf["Stories"].apply(lambda x :len([x for x in x.split() 
                                      if x.isdigit()]))


#%%
# Text Visualization
import pandas as pd
data = pd.read_csv("train.tsv", sep="\t")
#data.info()

data["Phrase"] = data["Phrase"].apply(lambda x : " ".join(x.lower() for x in x.split()))

# Punctuation
data["Phrase"] = data["Phrase"].str.replace("[^\w\s]","")

# Remove Numeric
data["Phrase"] = data["Phrase"].str.replace("\d","")

# Stopwords
from nltk.corpus import stopwords
sw = stopwords.words("english")
data["Phrase"] = data["Phrase"].apply(lambda x : " ".join(x for x in x.split() if x not in sw))

# Remove Low Usage Words
delete = pd.Series(" ".join(data["Phrase"]).split()).value_counts()[-1000:]
data["Phrase"] = data["Phrase"].apply(lambda x : " ".join(x for x in x.split() if x not in delete))

# Lemmatization
from textblob import Word
data["Phrase"] = data["Phrase"].apply(lambda x : " ".join(Word(i).lemmatize() for i in x.split()))

data["Phrase"].head(10)


#%%
tf1 = data["Phrase"].apply(lambda x:
                             pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ["words", "tf"]

# Visualization
a = tf1[tf1["tf"] > 1000]
a.plot.bar(x= "words", y= "tf")
# %%
# WordCloud
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = data["Phrase"][0]
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("Wordcloud.png");

#%% 
# Full Text
text = " ".join(i for i in data.Phrase)

wordcloud = WordCloud(max_font_size=50,
                      max_words=200,
                      background_color="white").generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


#%%
# Sentiment Analysis and Classification Models
from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd
import numpy as np
import xgboost, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

data = pd.read_csv("train.tsv", sep = "\t")

data["Sentiment"].replace(0, value = "negative", inplace = True)
data["Sentiment"].replace(1, value = "negative", inplace = True)
data["Sentiment"].replace(2, value = "neutral" , inplace = True)
data["Sentiment"].replace(3, value = "positive", inplace = True)
data["Sentiment"].replace(4, value = "positive", inplace = True)

data.groupby("Sentiment").count()

#%%
# Preprocessing Text

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]

# Lower Case
df["text"] = df["text"].apply(lambda x : " ".join(x.lower() for x in x.split()))

# Punctuation
df["text"] = df["text"].str.replace("[^\w\s]","")

# Remove Numeric
df["text"] = df["text"].str.replace("\d","")

# Stopwords
from nltk.corpus import stopwords
sw = stopwords.words("english")
df["text"] = df["text"].apply(lambda x : " ".join(x for x in x.split() if x not in sw))

# Remove Low Usage Words
delete = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x : " ".join(x for x in x.split() if x not in delete))

# Lemmatization
from textblob import Word
df["text"] = df["text"].apply(lambda x : " ".join(Word(i).lemmatize() for i in x.split()))
#%%
# Feature Engineering

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df["text"],
                                                                    df["label"],
                                                                    test_size = 0.33,
                                                                    random_state=(42))
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(Y_train)
y_test  = encoder.fit_transform(Y_test)

# Count Vectors

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

x_train_count = vectorizer.transform(X_train)
x_test_count  = vectorizer.transform(X_test)

# TF-IDf
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(X_train)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(X_train)
x_test_tf_idf_word  = tf_idf_word_vectorizer.transform(X_test)

# ngram level tf_idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(X_train)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(X_train)
x_test_tf_idf_ngram  = tf_idf_ngram_vectorizer.transform(X_test)

# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char",
                                          ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(X_train)

x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(X_train)
x_test_tf_idf_chars  = tf_idf_chars_vectorizer.transform(X_test)

#%%

def getAccuracy(model, params = {}):
    f_model = model(**params)
    f_model.fit(x_train_count, y_train)
    accuracy = model_selection.cross_val_score(f_model,
                                               x_test_count,
                                               y_test,
                                               cv = 10).mean()
    print(model.__name__, "Count Vector Accuracy Score:",accuracy)
    f_model = model(**params)
    f_model.fit(x_train_tf_idf_word, y_train)
    accuracy = model_selection.cross_val_score(f_model,
                                               x_test_tf_idf_word,
                                               y_test,
                                               cv = 10).mean()
    print(model.__name__, "Word-Level TF-IDF Accuracy Score:",accuracy)
    f_model = model(**params)
    f_model.fit(x_train_tf_idf_ngram, y_train)
    accuracy = model_selection.cross_val_score(f_model,
                                               x_test_tf_idf_ngram,
                                               y_test,
                                               cv = 10).mean()
    print(model.__name__, "NGRAM-Level TF-IDF Accuracy Score:",accuracy)
    f_model = model(**params)
    f_model.fit(x_train_tf_idf_chars, y_train)
    accuracy = model_selection.cross_val_score(f_model,
                                               x_test_tf_idf_chars,
                                               y_test,
                                               cv = 10).mean()
    print(model.__name__, "Character-Level TF-IDF Accuracy Score:",accuracy)

getAccuracy(linear_model.LogisticRegression, params = {"max_iter" : 30})
getAccuracy(naive_bayes.MultinomialNB)
getAccuracy(ensemble.RandomForestClassifier, params={"n_estimators" : 10,
                                                     "max_leaf_nodes" : 10})
getAccuracy(xgboost.XGBClassifier)


#%%
# Predict from Given Text
def transformTest(text):
    textSeries = pd.Series(text)
    
    # Lower Case
    textSeries = textSeries.apply(lambda x : " ".join(x.lower() for x in x.split()))
    
    # Punctuation
    textSeries = textSeries.str.replace("[^\w\s]","")
    # Remove Numeric
    textSeries = textSeries.str.replace("\d","")

    # Stopwords
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    textSeries = textSeries.apply(lambda x : " ".join(x for x in x.split() if x not in sw))

    # Lemmatization
    from textblob import Word
    textSeries = textSeries.apply(lambda x : " ".join(Word(i).lemmatize() for i in x.split()))
    
    v = CountVectorizer()
    v.fit(X_train)
    textSeries = v.transform(textSeries)
    
    return textSeries


text_positive = "this film is very nice and good i like it"
text_neutral  = "fine i guess"
text_negative = "very bad look at this shit i hate it"

predictModel = linear_model.LogisticRegression()
predictModel.fit(x_train_count, y_train)
print("'",text_positive,"' Prediction:",predictModel.predict(transformTest(text_positive))[0])
print("'",text_neutral, "' Prediction:",predictModel.predict(transformTest(text_neutral))[0])
print("'",text_negative,"' Prediction:",predictModel.predict(transformTest(text_negative))[0])

# 2 -> Positive
# 1 -> Neutral
# 0 -> Negative

#%%











