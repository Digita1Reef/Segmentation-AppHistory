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



# Data processing
import pandas as pd
import numpy as np
# Text preprocessiong
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP

# Initiate UMAP

umap_model = UMAP(n_neighbors=15, 
                  n_components=5, 
                  min_dist=0.0, 
                  metric='cosine', 
                  random_state=100)
# Initiate BERTopic
topic_model = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)
# Run BERTopic model
topics, probabilities = topic_model.fit_transform(apphistory['packages_text'])

#SHOW TOPICS
topic_model.get_topic_info()
#print(topic_model.get_topic(0))

AMOUNT_TO_EXTRACT=os.getenv("TOPICS_TO_MODEL")
TOPICS=int(AMOUNT_TO_EXTRACT)

#BARCHART
fig=topic_model.visualize_barchart(top_n_topics=TOPICS)
barchart_file=os.path.join(script_path,'barchart_file_topics-'+str(TOPICS)+'.html')
print(barchart_file)
with open(barchart_file, "w") as open_method:
    fig.write_html(open_method)

#TERM_RANK
term_rank=topic_model.visualize_term_rank()
term_rank_file=os.path.join(script_path,'term_rank_file_topics-'+str(TOPICS)+'.html')

with open(term_rank_file, "w") as open_method:
    term_rank.write_html(open_method)
print(term_rank_file)
#print(term_rank)



#REPRESENTATIVE_DOCS
representative_docs=topic_model.get_representative_docs()
print(representative_docs)
#print(term_rank_file)
#print(term_rank)


#TOPIC_DISTANCE

# topic_distance=topic_model.visualize_topics(top_n_topics=TOPICS)
# topic_distance_file=os.path.join(script_path,'topic_distance_file_topics-'+str(TOPICS)+'html')
# with open(topic_distance_file, "w") as open_method:
#     topic_distance.write_html(open_method)
# print(topic_distance_file)
#print(topic_distance)

#TOPIC_DISTRIBUTION

topic_distribution=topic_model.visualize_distribution(topic_model.probabilities_[0], min_probability=0.015)
topic_distribution_file=os.path.join(script_path,'topic_distribution_file_topics-'+str(TOPICS)+'.html')
with open(topic_distribution_file, "w") as open_method:
    topic_distribution.write_html(open_method)
print(topic_distribution_file)
#print(topic_distribution)