import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# Local paths
csv1_path = r'c:\Users\דניאל\Google Drive\active_learning_project\finance_en_content_all.csv'
csv1_labels_path = r'c:\Users\דניאל\Google Drive\active_learning_project\finance_en_labels_all.csv'
csv2_path = r'c:\Users\דניאל\Google Drive\active_learning_project\finance_sites_from_classifier.csv'
categories_path = r'c:\Users\דניאל\Google Drive\active_learning_project\categories.txt'

# Read first .csv file
csv1 = pd.read_csv(csv1_path)

#Filter non-English domains
csv1 = csv1[csv1.language == 'en']
csv1 = csv1.drop(columns='language')

# Add labels
labels_csv1 = pd.read_csv(csv1_labels_path)
csv1 = csv1.merge(labels_csv1, on='domain', how='inner')

# Read second .csv file
csv2 = pd.read_csv(csv2_path)

# Remove irrelevant domains (1018 = 'Account Suspended')
csv2 = csv2[csv2.classification != 1018]

# Unify & merge tables
csv2 = csv2.rename(columns={'classification': 'label', 'Domain Content': 'content'})
data = pd.concat([csv1, csv2], sort=False)
del csv1, csv2

# Match category to label
categories = pd.read_csv(categories_path, quotechar='\'', names=['label','category'])
categories = dict(zip(categories.label, categories.category))
data['category'] = data.label.apply(lambda label: categories[label])

# Filtering nulls
data = data[~data.content.isna()]

# Filtring domains with infiseable lengh of content (which are also probably garbage)
data = data[data.content.apply(len) < 500000]

# Fitering categories which are too small
too_small = data.category.value_counts()[data.category.value_counts() < 150]
# too_small includes: 'Cashing Checks', 'Credit Restoration Services', 'Crypto Currency'
data = data[~data.category.isin(['Cashing Checks', 'Credit Restoration Services', 'Crypto Currency'])]
data = data.reset_index(drop=True)

# Tokenizing
data['content'] = data.content.apply(nltk.word_tokenize)

# Filtering non-alphabetic tokens or single letter tokens, and normalizing case
data['content'] = data.content.apply(lambda tokens: [t.lower() for t in tokens if t.isalpha() and len(t) > 1])

# Removing stop words
stop_words = set(nltk.corpus.stopwords.words('english')) 
data['content'] = data.content.apply(lambda tokens: [t for t in tokens if t not in stop_words])

# Loading word2vec pretraind model
import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")

# Removing word that are not in the model
vocabulary = word_vectors.vocab
data['content'] = data.content.apply(lambda tokens: [t for t in tokens if t in vocabulary])

# Vectorizing content using tokens counts
def dummy(doc):  # walkaround to enable passing tokens to CountVectorizer
    return doc
cv = CountVectorizer(tokenizer=dummy,
                     preprocessor=dummy)
vectorized_content = cv.fit_transform(data.content)

# Computing tf-idf matrix
tfidf = TfidfTransformer()
tfidf_content = tfidf.fit_transform(vectorized_content)

# weighting token counts using tf-idf matrix
vectorized_content = vectorized_content.multiply(tfidf_content)

# Converting tokens to feature vectors using word2vec embeddings
w2v = cv.get_feature_names()
w2v = [word_vectors.get_vector(x) for x in w2v]
w2v = np.array(w2v)
vectorized_content = vectorized_content.dot(w2v)


