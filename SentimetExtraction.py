
# coding: utf-8

# Set up HDFS and Google credentials

# In[1]:


# In[2]:

from __future__ import print_function
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./imdb-e9e7ce7a779d.json"
os.environ["HDFSCLI_CONFIG"]="./.hdfscli.cfg"
os.environ["HADOOP_CONF_DIR"]="/opt/hadoop-3.1.0/etc/hadoop"
sc.environment["GOOGLE_APPLICATION_CREDENTIALS"]="./imdb-e9e7ce7a779d.json"


# List filenames of reviews from HDFS and parallelize in preparation from processing
from hdfs import Config
import pandas as pd

#9870
#from hdfs3 import HDFileSystem
#hdfs = HDFileSystem(u"localhost", 9000)


hdfs_prefix="hdfs://localhost:9000/"
reviews_path="/user/lmrd/reviews"

    
def getReviewFilenames(path):
    client = Config().get_client('dev')
    files = client.list(path)
    
    files_df = pd.DataFrame(data={"fileNames":files})
    
    files_df["index"] = files_df["fileNames"].str.extract('(.*)_.*\..*', expand = True).apply(pd.to_numeric)

    files_df = files_df.sort_values(by="index")
   # print(files_df)
    
    return [hdfs_prefix+path+"/"+f for f in files_df["fileNames"]]

pos_files_rdd = sc.parallelize(getReviewFilenames(reviews_path+"/pos"))
#pos_files_rdd.collect()
# Parallelise the reviews and use Google NLP API to extract entities and related sentiment.

# In[3]:


# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from functools import reduce

def collectEntities(x, y):
    # The first reduce call doesn't pass a list for x, so we need to check for that.
    if not isinstance(x, list):
        x=[x]
        

    xd = dict(x)
    #print(xd)
    
    if not isinstance(y, list):
        y = [y]
        
    for ye in y:
        if ye[0] in xd:
            xd[ye[0]] = (xd[ye[0]]+ye[1])/2
        else:
            xd[ye[0]] = ye[1]
    
    return [o for o in xd.items()]
        

import re

def extractEntitiesSetiment(fileObj):
    # Instantiates a client
    client = language.LanguageServiceClient()
    
    review_contents = fileObj
        
    #print(review_contents)
    
    document = types.Document(content = review_contents, 
                             type=enums.Document.Type.PLAIN_TEXT)
    
    entities = client.analyze_entity_sentiment(document=document, encoding_type="UTF8")
    
    # Make sure we have no duplicate entities. If we do, average their sentiment.
    justLetters = re.compile("[^a-z ]")
    response = [o for o in zip([justLetters.sub("", entity.name.lower()) for entity in entities.entities], [entity.sentiment.score * entity.sentiment.magnitude for entity in entities.entities])]
    response = sorted(response, key=lambda x: x[0])
    response = reduce(collectEntities, response)
    
    return response



file_objs = map(lambda f: ' '.join(list(sc.hadoopFile(f, "org.apache.hadoop.mapred.TextInputFormat", 
                                              "org.apache.hadoop.io.Text", 
                                              "org.apache.hadoop.io.Text").values().collect())), pos_files_rdd.collect())

#file_objs = sc.parallelize(file_objs)
#entity_documents_info = file_objs.map(extractEntitiesSetiment)


#entity_documents_info.cache()
#print(entity_documents_info.take(1))
#print(entity_documents_info.count())
# In[14]:


tf = sc.wholeTextFiles("hdfs://localhost:9000/user/lmrd/reviews/pos_toy")


# In[16]:


import re
from pyspark.sql.types import *

def extractEntitiesSetiment2(fileObj):
    # Instantiates a client
    client = language.LanguageServiceClient()
    
    review_contents = fileObj[1]
        
    #print(review_contents)
    
    document = types.Document(content = review_contents, 
                             type=enums.Document.Type.PLAIN_TEXT)
    
    entities = client.analyze_entity_sentiment(document=document, encoding_type="UTF8")
    
    # Make sure we have no duplicate entities. If we do, average their sentiment.
    justLetters = re.compile("[^a-z ]")
    response = [o for o in zip([justLetters.sub("", entity.name.lower()) for entity in entities.entities], [entity.sentiment.score * entity.sentiment.magnitude for entity in entities.entities])]
    response = sorted(response, key=lambda x: x[0])
    response = reduce(collectEntities, response)
    
    return (fileObj[0], response)

def extractOrdering(rec):
    filenameRegexp = ".*/([0-9]*)_.*\.txt$"
    r = re.search(filenameRegexp, rec[0])

    return (int(r.groups()[0]), rec[1])
    #hdfs://localhost:9000/user/lmrd/reviews/pos/3467_7.txt

#sc.broadcast(filenameRegexp)
filesRdd = tf.map(extractOrdering)

#schema1 = StructType([
#    StructField("ID", IntegerType(), False),
#    StructField("GENRE", ArrayType(
#            StructField("ENTITY", StringType(), False), 
#            StructField("SENTIMENT", FloatType(), False)), nullable=True)])

entity_documents_info = spark.createDataFrame(filesRdd.map(extractEntitiesSetiment2), schema=["ID", "ENTITIY_SENTIMENT"])
#entity_documents_info = filesRdd.map(extractEntitiesSetiment2)
#entity_documents_info.show(5)


# In[10]:


filesRdd.take(5)
#entity_documents_info.show(5)
#entity_documents_info.take(5)


# Load genre information from file (previously collected using IMDB API)

# In[17]:


import pickle
import pandas as pd
import base64
from functools import reduce

def decodeGenre(x):
    try: 
        g = pickle.loads(base64.b64decode(x[2:-1]), encoding="bytes") 
        if (len(g)==0):
            return ["NA"]
        else:
            return g
    except:
        return ["NA"]    
        
        
genres = pd.read_csv("Data/genres_train_urls_pos.csv", sep="\t", index_col=0, usecols=[1, 2, 3])
#print(genres.head())
genres = genres.fillna(value="b''")
genres["GENRE"] = genres["GENRE"].apply(decodeGenre) 

# Get list of unique genre values
#unique_genres = set(reduce(lambda x, y: x+y, genres["GENRE"].values))
#print(unique_genres)

#print(genres)
#print(genres[["ID", "GENRE"]])
#z = zip(genres["ID"], genres["GENRE"])


#genres_rdd = sc.parallelize([(int(k)-1, v[0], v[1]) for (k, v) in genres.iteritems()])

schema = StructType([
    StructField("FILM_ID", IntegerType(), True),
    StructField("GENRE", ArrayType(StringType(), containsNull=True), True)])

genres_df = spark.createDataFrame(genres, schema)

from pyspark.sql.functions import monotonically_increasing_id

# This will return a new DF with all the columns + id
genres_df = genres_df.withColumn("ID", monotonically_increasing_id()).limit(10)

genres_df.show()
#genres_rdd.collect()


# In[18]:


entity_documents_info = entity_documents_info.alias("df1").join(genres_df.alias("df2"), entity_documents_info.ID == genres_df.ID).select(["df1.*", "df2.FILM_ID", "df2.GENRE"])

entity_documents_info.show(5)


# Zip the document-entity-sentiment rdd with the genre rdd.
# There should be exactly the same number of reviews as records in the genres rdd.
entity_documents_info = entity_documents_info.zip(genres_rdd)

#entity_documents_info.collect()
# Group documents by genre

# In[25]:


def separateGenres(rec):
    print(len(rec))
    return [[genre, rec[0]] for genre in rec[1][1]]

def separateGenres2(rec):
    return [[genre, e, s] for (e, s) in rec[0] for genre in rec[1][1]]

def separateGenres3(rec):
    print(rec)
    return [[genre, e, s] for (e, s) in rec.ENTITIY_SENTIMENT for genre in rec.GENRE]
    
#grouped_entities = entity_documents_info.flatMap(separateGenres).reduceByKey(collectEntities)
grouped_entities = entity_documents_info.rdd.flatMap(separateGenres3)

grouped_entities_df = spark.createDataFrame(data=grouped_entities, schema=["genre", "entity", "sentiment"])
#grouped_entities_df.show()
grouped_entities_df.cache()


# In[27]:


grouped_entities_df.show()


# In[28]:


from pyspark.sql import Row
from pyspark.sql.functions import collect_list

#def removeSentiment(x):
#    entities = list()
#    for xe in x:
#        entities.append(xe[0])
#        
#    return entities
#
#grouped_entity_words = grouped_entities.values().map(removeSentiment)

grouped_entity_words = grouped_entities_df.select(["genre", "entity"]).groupBy("genre").agg(collect_list("entity").alias("entities"))
grouped_sentiment = grouped_entities_df.select(["genre", "sentiment"]).groupBy("genre").agg(collect_list("sentiment").alias("sentiment"))
#grouped_entity_words.show()
#grouped_sentiment.show()


# In[31]:


grouped_sentiment.show()


# In[32]:


from pyspark.ml.feature import CountVectorizer, IDF

# remove sentiment info for use by hashingTF/tfif

# Load documents (one per line).

countVec = CountVectorizer(inputCol="entities", outputCol="tf")
cvmodel = countVec.fit(grouped_entity_words)

tf = cvmodel.transform(grouped_entity_words)
tf.show()
#sc.broadcast(hashingTF)

# While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# First to compute the IDF vector and second to scale the term frequencies by IDF.
tf.cache()
idf = IDF(inputCol="tf", outputCol="tfidf").fit(tf)
tfidf = idf.transform(tf)
tfidf.show()
# spark.mllib's IDF implementation provides an option for ignoring terms
# which occur in less than a minimum number of documents.
# In such cases, the IDF for these terms is set to 0.
# This feature can be used by passing the minDocFreq value to the IDF constructor.
# idfIgnore = IDF(minDocFreq=2).fit(tf)
# tfidfIgnore = idfIgnore.transform(tf)


# In[33]:


from pyspark.sql import Row
from pyspark.sql.functions import explode
import numpy as np

vocab = tfidf.select(["genre", "tfidf"])
genreVocabs = dict()

for genre in vocab.collect():
    genreName = genre.genre
    
    t=genre.tfidf
    genreVocabs[genreName] = t
    
globalVocab = list(cvmodel.vocabulary)
    
sc.broadcast(globalVocab)
sc.broadcast(genreVocabs)

def remapEntitiesByTfidf(row):
    tfidfMappings = genreVocabs[row.genre]
    tfIndex = globalVocab.index(row.entity)
    tfidf = tfidfMappings[tfIndex]
    
    return Row(genre=row.genre, entity=row.entity, tfidf=float(tfidf), vocabIndex=int(tfIndex))
    
genreCorpora=dict()

for genre in genreVocabs.keys():
    genreEntities = tfidf.where(tfidf.genre==genre).select("genre", explode("entities").alias("entity"))
    
    #genreEntities.show()
    
    #data = genreEntities.rdd.map(remapEntitiesByTfidf)

    entitiesByTfidf = spark.createDataFrame(data=genreEntities.rdd.map(remapEntitiesByTfidf), schema=["entity", "genre", "tfidf", "vocabIndex"])
    #entitiesByTfidf.show()
    entitiesByTfidf = entitiesByTfidf.join(grouped_entities_df, on=["genre", "entity"], how="inner" ).groupBy(["genre", "entity", "tfidf", "vocabIndex"]).avg("sentiment").sort("tfidf", ascending=False)
    
    genreCorpora[genre] = entitiesByTfidf
    
genreCorpora["Comedy"].show()


# In[55]:


entitiesByTfidf.collect()


# In[249]:


grouped_words_tfidf = grouped_entities.values().zip(tfidf)

grouped_words_tfidf.collect()


# In[243]:




collectedTFDF = tfidf.collect()
print(len(collectedTFDF))
i=1
for d in grouped_entity_words.collect():
    print("Document ", i, ":")
    
    tfidfscores = [collectedTFDF[i-1][hashingTF.indexOf(d[e])] for e in range(0, len(d))]
    
    for e in range(0, len(d)):
        print(d[e]+"("+str(collectedTFDF[i-1][hashingTF.indexOf(d[e])])+") ", end="")
        
    i+=1
    print(" ")
    


# In[48]:


import pickle
import pandas as pd
import base64

def decodeGenre(x):
    try: 
        return pickle.loads(base64.b64decode(x[2:-1]), encoding="bytes") 
        
    except:
        return []    
        
        
genres = pd.read_csv("id_pos_genre_train.csv", sep="\t", index_col=0, usecols=[1, 2, 3])
genres = genres.fillna(value="b''")
genres["GENRE_t"] = genres["GENRE"].apply(decodeGenre) 
print(genres["GENRE_t"])

# url_df = pd.read_csv("id_pos_genre_train_prot2_stringdump.csv", delimiter="\t")
# #temp = url_df["GENRE"].apply(lambda x: print(x))#pickle.loads(x.encode()))#
# print(pickle.loads(url_df.iloc[0,3][2:-1].encode("utf-8")))
#f = open("id_pos_genre_train.csv", mode="rb")

#o=pickle.load(f)

#f.close()

#print(len(o))
#print(o)

#print(o.loc[o["ID"]=="0453418"])
##print(o["ID"]=="0453418")
#print(["Comedy" in l for l in o["GENRE"]])
#o.loc[["Comedy" in l for l in o["GENRE"]]]

# pickle.dumps(o.iloc[0, 2])


# In[33]:


url_df.head()


# In[45]:


type(pickle.dumps(o.iloc[0, 2]))


# In[228]:


test = [("a", 1), ("b", 2), ("c", 3), ("a", 4), ("c", 6)]

test_sorted = sorted(test, key=lambda x: x[0])
print(test_sorted)

print(reduce(collectEntities, test_sorted))
#entity_documents_info.collect()

