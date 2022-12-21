import pandas as pd
import ast



apphistory_file_name='~/Documents/Segmentation/Segmentation-AppHistory/50k_androidid_with_apphistory.csv'

# Read data into papers
print(apphistory_file_name)
apphistory = pd.read_csv(apphistory_file_name)



# Print out the first rows of papers
apphistory.head(10)

# Load the regular expression library
import re

remove_values=['com.clarocolombia.contenedor','com','android','samsung','contenedor','apps','app','google','system','manager','sec','facebook','telcel']

print(apphistory.dtypes)
# Remove punctuation
apphistory['packages']=apphistory['packages'].apply(ast.literal_eval)
print(apphistory)
print(apphistory.dtypes)
apphistory.loc[:, 'packages'] = apphistory['packages'].replace(remove_values, '',regex=True)
apphistory['packages_text'] = apphistory['packages'].apply(lambda x: ','.join(x))
apphistory.loc[:, 'packages_text'] = apphistory['packages_text'].replace(remove_values, '',regex=True)
# Convert the titles to lowercase
#apphistory['packages_processed'].map(lambda x: x.lower())

apphistory.head(10)
# Import the wordcloud library
from wordcloud import WordCloud

# print(','.join(apphistory['packages_text']))

# Join the different processed titles together.
long_string = ','.join(list(apphistory['packages_text'].values))

# Create a WordCloud object
# wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

# # Generate a word cloud
# wordcloud.generate(long_string)

# # Visualize the word cloud
# wordcloud.to_file('file.png')

data = apphistory.packages_text.values.tolist()
import gensim
import gensim.corpora as corpora


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(data))      
# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])

from pprint import pprint

# number of topics
num_topics = 15

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os
# Visualize the topics
# pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('/home/camilo/Documents/Segmentation/Segmentation-AppHistory/ldavis_prepared_'+str(num_topics))

print(LDAvis_data_filepath)
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared  = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, '/home/camilo/Documents/Segmentation/Segmentation-AppHistory/ldavis_prepared_'+ str(num_topics) +'.html')

LDAvis_prepared
