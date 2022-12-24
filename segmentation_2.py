import pandas as pd
import ast
import os
import psycopg2
from psycopg2.sql import SQL, Identifier
from psycopg2.sql import Literal as LiteralSQL

from dotenv import load_dotenv

load_dotenv(override=True)

connection=psycopg2.connect(
                user=os.getenv("DB_USER_CITUS"),
                password=os.getenv("DB_PASSWORD_CITUS"),
                dbname=os.getenv("DB_NAME_CITUS"),
                port=os.getenv("DB_PORT_CITUS"),
                host=os.getenv("DB_HOST_CITUS"),
                application_name="etl_migrations",
                sslmode="disable"
            )


save_to_csv_string_stout = SQL(
    """COPY (select androidid, jsonb_agg(appname) as packages
    from brand_data_columnar_apphistory
    group by androidid
    limit {limit})
    to STDOUT WITH
    DELIMITER ','
    CSV HEADER"""
)

AMOUNT_TO_EXTRACT=os.getenv("AMOUNT_TO_EXTRACT")
LIMIT=AMOUNT_TO_EXTRACT

script_path: str = os.path.dirname(os.path.abspath(__file__))

csv_filename=os.path.join(script_path,f"output_{LIMIT}.csv")
with connection as con:
    with con.cursor() as cursor:
        with open(csv_filename, "w") as open_method:
            cursor.copy_expert(
                save_to_csv_string_stout.format(
                    limit=LiteralSQL(LIMIT),
                ),
                open_method,
            )
apphistory_file_name=csv_filename

# Read data into papers
print(apphistory_file_name)
apphistory = pd.read_csv(filepath_or_buffer=apphistory_file_name,
            delimiter=",",
            encoding_errors="replace",
            encoding="utf-8")



# Print out the first rows of papers
apphistory.head(10)

# Load the regular expression library
import re

#remove_values=['com.clarocolombia.contenedor','com','android','samsung','contenedor','apps','app','google','system','manager','sec','facebook','telcel','orca']

print(apphistory.dtypes)
# Remove punctuation
apphistory['packages']=apphistory['packages'].apply(ast.literal_eval)
print(apphistory)
print(apphistory.dtypes)
#apphistory.loc[:, 'packages'] = apphistory['packages'].replace(remove_values, '',regex=True)
apphistory['packages_text'] = apphistory['packages'].apply(lambda x: ','.join(x))
#apphistory.loc[:, 'packages_text'] = apphistory['packages_text'].replace(remove_values, '',regex=True)
# Convert the titles to lowercase
#apphistory['packages_processed'].map(lambda x: x.lower())

apphistory.head(10)
# Import the wordcloud library

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
# print(corpus[:1][0][:30])

from pprint import pprint

# number of topics
AMOUNT_TO_EXTRACT=os.getenv("TOPICS_TO_MODEL")
num_topics = AMOUNT_TO_EXTRACT

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

LDAvis_data_filepath=os.path.join(script_path,'ldavis_prepared_'+str(num_topics))

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

LDAvis_data_filepath_html=os.path.join(script_path,'ldavis_prepared_'+str(num_topics)+'.html')
pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath_html )
print(f"HTML file output {LDAvis_data_filepath_html}")
