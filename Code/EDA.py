### import section ###
 
import math
import os
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

######################

### data read section ###

# NOTE: replace with your own data folder path
DATA_FOLDER = r'C:\Users\Ofir\Downloads'

# WARNING: this will take up to 3.0GB of your RAM -> might get your computer stuck
# read content df
content_df = pd.read_csv(os.path.join(DATA_FOLDER,'finance_en_content_all.csv'))
# read labels df
labels_df = pd.read_csv(os.path.join(DATA_FOLDER,'finance_en_labels_all.csv'))
# read categories df - match between label (number) to category (name)
categories_df = pd.read_csv(os.path.join(DATA_FOLDER,'categories.txt'),names=['label','category'])
categories_df['label'] = categories_df['label'].apply(lambda num_str_dirty: int(num_str_dirty.replace("'", "")))
categories_df['category'] = categories_df['category'].apply(lambda str_dirty: str_dirty.replace("'", ""))
# unit to 1 df:
content_labels_df = pd.merge(content_df,labels_df,on='domain')
content_labels_categories_df = pd.merge(content_labels_df,categories_df,on='label')

# delete the unnecessary dataframes
del content_df
del labels_df

# read the second csv and merge it to the first one
new_content_df = pd.read_csv(os.path.join(DATA_FOLDER,'finance_sites_from_classifier.csv'))
new_content_df.columns = ['domain', 'label','content']
new_content_labels_categories_df = pd.merge(new_content_df,categories_df,on='label')
df = pd.concat([new_content_labels_categories_df, content_labels_categories_df], sort = True)
df['language'] = df.language.fillna('en')
df = df.reset_index(drop=True)
df = df[['domain', 'content', 'language', 'label', 'category']]

# delete the old saperate data frames, and check the label distribution again:
del content_labels_categories_df
del new_content_labels_categories_df
del new_content_df

#######################

### EDA code ###


# create a dictionary that match between label(key) to category (value)
categories = pd.read_csv(os.path.join(DATA_FOLDER,'categories.txt'), quotechar='\'', names=['label','category'])
categories = dict(zip(categories.label, categories.category))

# show the entire dataset header:
print(df.head())

## cleaning:

# delete the "Account suspended" category (Accidentally brought by the company members),<br>
df = df[df.category != 'Account Suspended']
# In addition, filter the categories with less than 150 samples 
df = df[~df.category.isin(['Cashing Checks', 'Credit Restoration Services', 'Crypto Currency'])]
# print the size of the df after the filter process 
print(len(df.columns),len(df.index))
# plot the distribution of the categories:
plt.figure(figsize=(8, 8)) 
ax = df.category.value_counts().plot(kind='barh')

for p in ax.patches:
    w = p.get_bbox().bounds[2]
    h = p.get_bbox().bounds[1]
    ax.annotate(int(w), xy = (w, h))

# checking and visualize the missing values in the data: 
sns.heatmap(df.isnull().T.astype(int))
print(df.isnull().sum())


# label distribution for the missing values:
plt.figure(figsize=(15,5)) # this creates a figure 10 inch wide, 5 inch high
sns.countplot(df[df['content'].isna()]['category'])
plt.xticks(rotation=-90)
plt.show()

# filter out the contentless sites and print the size of the filtered df:
df.dropna(axis=0, inplace=True)
print(len(df.columns),len(df.index))


#language distribution:
plt.figure(figsize=(15,5)) # this creates a figure 10 inch wide, 5 inch high
sns.countplot(df['language'])
plt.xticks(rotation=-45)
plt.show()


#filter the english sites
df = df[df['language'] == 'en']
print(len(df.columns),len(df.index))

##

## finding potential features (around the sites' content):
 
# label distribution by the content length (number of chars) of each site:

df['content_len'] = df['content'].apply(len)
# boxplot showing the content length distribution per category
plt.figure(figsize=(16, 8))
ax = sns.boxplot(x='category', y='content_len', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()
# filter the far outliers (that exceed 750K chars):
plt.figure(figsize=(16, 8))
df = df[df['content_len'] <= 750000]
ax = sns.boxplot(x='category', y='content_len', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()

# label distribution by the domain postfix of each site:

# domain postfix extraction
def get_domain_postfix(domain_str):
    return domain_str.split('.')[-1]
df['domain_postfix'] = df['domain'].apply(get_domain_postfix)
# countplot of the categories distribution by domain postfix, showing top 10 popular postfix
plt.figure(figsize=(16, 8))
ax = sns.countplot(x="domain_postfix", hue="category", data=df,
                   order=df.domain_postfix.value_counts().iloc[:10].index)
ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()

##

## find potential features in the content itself:

# find the most common words of the sites
# for the EDA  we will show the graphs for 3 categories only, due to memory issues and graphic visibility.
# for using the EDA code for all the categories use df instead of short_df (in addition, some of the code need to adjust 
# Tokenizing
import nltk
short_df = df.loc[df['category'].isin(['Forex','Financial Institution', 'Loans'])]
short_df['words'] = short_df.content.apply(nltk.word_tokenize)
short_df.words.head()
# filter the non-alphabetic words, the stop words and the extensions(lemmatize): 
short_df['words'] = short_df.words.apply(lambda tokens: [t.lower() for t in tokens if t.isalpha() and len(t) > 1])
# using the stop words vocabulary of the nltk package
stop_words = set(nltk.corpus.stopwords.words('english')) 
short_df['words'] = short_df.words.apply(lambda tokens: [t for t in tokens if t not in stop_words])
# lemmatizing
from nltk.stem import WordNetLemmatizer
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
short_df['words'] = short_df.words.apply(lambda words: [lemmatizer.lemmatize(w) for w in words])

# Vectorizing using tokens counts
from sklearn.feature_extraction.text import CountVectorizer
# walkaround function to enable passing tokens to CountVectorizer
def dummy(doc):  
    return doc
# for each site, create a vector of the words' indexes and the number of their occurence  
cv = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
cv.fit(short_df.words)
cv.get_feature_names()
vectorized_content = cv.transform(short_df.words)
#the 10 most frequent words in the whole dataset
#can be replace to other numbers..
number_of_words = 10
short_df.reset_index(inplace=True)
short_df_indices = short_df.index.tolist()
apperance_words_frequency_array = vectorized_content[short_df_indices].sum(axis=0)
words_freq = [(word, apperance_words_frequency_array[0, idx],idx) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:number_of_words])

#the 10 most frequent words distribution by category:

most_common_index = []
most_common_words= []
most_common_amount = []
for i in range(number_of_words):
    most_common_words.append(words_freq[:number_of_words][i][0])
    most_common_amount.append(words_freq[:number_of_words][i][1]) 
    most_common_index.append(words_freq[:number_of_words][i][2])


categories = short_df.category.unique().tolist()
amount_by_category = []
temp = pd.DataFrame()
for category in categories:
    short_df_indices_by_category = short_df.index[short_df.category == category].tolist()  
    apperance_words_frequency_array_by_category = vectorized_content[short_df_indices_by_category].sum(axis=0)
    apperance_words_frequency_array_by_category = [i/len(short_df_indices_by_category) for i in apperance_words_frequency_array_by_category]
    amount_by_category.append([apperance_words_frequency_array_by_category[0].tolist()[0][i] for i in most_common_index]) 
    temp = pd.concat([temp,pd.DataFrame(data={'word':most_common_words,'category': [category]*len(most_common_words), 'avg': amount_by_category[categories.index(category)]})])

temp.reset_index(inplace=True,drop=True)
temp.sort_values(by='avg', axis=0, ascending=False)

# visualize:
fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x="word", y="avg", hue="category", data=temp)
for item in ax.get_xticklabels():
    item.set_rotation(45)
plt.title('Ten most common words in all data set - distrbution for 3 categories')


# the 10 common words for each category relative to their appearance in the other categories:
number_of_words_by_category = 10
most_common_by_category =[]
most_common_by_category_word = []

for category in categories:
    subset_indices = short_df[short_df.category == category].index.tolist()
    apperance_words_frequency_array = vectorized_content[subset_indices].sum(axis=0)

    words_freq = [(word, apperance_words_frequency_array[0, idx]/len(subset_indices),idx) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    most_common_by_category.append([freq[2] for freq in words_freq[:number_of_words_by_category]])
    most_common_by_category_word.append([freq[0] for freq in words_freq[:number_of_words_by_category]])

print('Ten most common words for each label is:')
for inx,category in enumerate(categories):
    print("category - ",category,':')
    print(most_common_by_category_word[inx])
    print('----------------------------------------')


# visualize:
my_dict = {}
temp_df = list()
for i in range(len(categories)):
    for category in categories:
        index = short_df[short_df.category == category].index.tolist()
        temp_vec = vectorized_content[index,:]
        apperance_words_frequency_array = temp_vec[:,most_common_by_category[i]].sum(axis=0).tolist()[0]
        my_dict[category] = list(map(lambda x: x/len(index), apperance_words_frequency_array))
        
    my_dict['word']=most_common_by_category_word[i]
    temp_df.append(pd.DataFrame(my_dict))
    my_dict.clear()

df0 = pd.melt(temp_df[0],value_vars=categories,id_vars='word')
df1 = pd.melt(temp_df[1],value_vars=categories,id_vars='word')
df2 = pd.melt(temp_df[2],value_vars=categories,id_vars='word')

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df0)
plt.title('Ten most common words - {}'.format(categories[0]))

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df1)
plt.title('Ten most common words - {}'.format(categories[1]))

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df2)
plt.title('Ten most common words - {}'.format(categories[2]))

# 10 least common word in each category relative to their appearance in the other categories:

number_of_words_by_category = 10
most_none_common_by_category =[]
most_none_common_by_category_word = []

for category in categories:
    subset_indices = short_df[short_df.category == category].index.tolist()
    apperance_words_frequency_array = vectorized_content[subset_indices].sum(axis=0)

    words_freq = [(word, apperance_words_frequency_array[0, idx]/len(subset_indices),idx) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=False)
    
    most_none_common_by_category.append([freq[2] for freq in words_freq[:number_of_words_by_category]])
    most_none_common_by_category_word.append([freq[0] for freq in words_freq[:number_of_words_by_category]])

print('Ten most uncommon words for each label is:')
for inx,category in enumerate(categories):
    print("category - ",category,':')
    print(most_none_common_by_category_word[inx])
    print('----------------------------------------')

# visualize:

my_dict = {}
temp_df = list()
for i in range(len(categories)):
    for category in categories:
        index = short_df[short_df.category == category].index.tolist()
        temp_vec = vectorized_content[index,:]
        apperance_words_frequency_array = temp_vec[:,most_none_common_by_category[i]].sum(axis=0).tolist()[0]
        my_dict[category] = list(map(lambda x: x/len(index), apperance_words_frequency_array))
        
    my_dict['word']=most_none_common_by_category_word[i]
    temp_df.append(pd.DataFrame(my_dict))
    my_dict.clear()

df0 = pd.melt(temp_df[0],value_vars=categories,id_vars='word')
df1 = pd.melt(temp_df[1],value_vars=categories,id_vars='word')
df2 = pd.melt(temp_df[2],value_vars=categories,id_vars='word')

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df0)
plt.title('Ten least common words - {}'.format(categories[0]))

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df1)
plt.title('Ten least common words - {}'.format(categories[1]))

fig_size = (11.7, 8.27)
fig, ax = plt.subplots(figsize=fig_size)
sns.barplot(x='word',hue='variable',y='value',data=df2)
plt.title('Ten least common words - {}'.format(categories[2]))


