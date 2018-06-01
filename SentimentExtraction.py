#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:18:17 2018

@author: etienne
"""

import os
import re
import pickle
import pandas as pd
import base64
from functools import reduce
from pyspark.sql import Row
from pyspark.sql.functions import explode
import numpy as np

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from functools import reduce

from pyspark import SparkContext, SparkConf

#conf = SparkConf().setAppName('MyFirstStandaloneApp')
#sc = SparkContext(conf=conf)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./imdb-e9e7ce7a779d.json"
os.environ["HDFSCLI_CONFIG"]="./.hdfscli.cfg"
os.environ["HADOOP_CONF_DIR"]="/opt/hadoop-3.1.0/etc/hadoop"
sc.environment["GOOGLE_APPLICATION_CREDENTIALS"]="./imdb-e9e7ce7a779d.json"


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
    print(files_df)
    
    return [hdfs_prefix+path+"/"+f for f in files_df["fileNames"]]

pos_files_rdd = sc.parallelize(getReviewFilenames(reviews_path+"/pos")[0:10])
#pos_files_rdd.collect()


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

file_objs = sc.parallelize(file_objs)
entity_documents_info = file_objs.map(extractEntitiesSetiment)
entity_documents_info.cache()
#print(entity_documents_info.take(1))
#print(entity_documents_info.count())



def decodeGenre(x):
    try: 
        return pickle.loads(base64.b64decode(x[2:-1]), encoding="bytes") 
        
    except:
        return ["NA"]    
        
        
genres = pd.read_csv("id_pos_genre_train.csv", sep="\t", index_col=0, usecols=[1, 2, 3])
genres = genres.fillna(value="b''")
genres["GENRE"] = genres["GENRE"].apply(decodeGenre) 

# Get list of unique genre values
#unique_genres = set(reduce(lambda x, y: x+y, genres["GENRE"].values))
#print(unique_genres)

#print(genres[["ID", "GENRE"]])

genres_rdd = sc.parallelize([o for o in zip(genres["ID"], genres["GENRE"])])

genres_rdd.collect()

entity_documents_info = entity_documents_info.zip(genres_rdd)

#entity_documents_info.collect()
def separateGenres(rec):
    print(len(rec))
    return [[genre, rec[0]] for genre in rec[1][1]]

def separateGenres2(rec):
    return [[genre, e, s] for (e, s) in rec[0] for genre in rec[1][1]]

#grouped_entities = entity_documents_info.flatMap(separateGenres).reduceByKey(collectEntities)
grouped_entities = entity_documents_info.flatMap(separateGenres2)

grouped_entities_df = spark.createDataFrame(data=grouped_entities, schema=["genre", "entity", "sentiment"])
#grouped_entities_df.show()
grouped_entities_df.cache()

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

