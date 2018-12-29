import sys
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import subprocess
import numpy
import pickle
import numpy as np
import pandas as pd
import re
import pymysql
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from flask import Flask, request, jsonify

# db = pymysql.connect("localhost","root","","internship" )

# cursor = db.cursor()
db = pymysql.connect("db4free.net","gunjanrl","gunj@nrl","id4165564_olcade")
cursor = db.cursor()
cursor.execute("SELECT description from newcourses;")
courses_desc2 = cursor.fetchall()
train = pd.read_sql_query("select * from newcourses;", db)

course_ids=list(train["courseid"])
courses=train["coursetitle"]
courses_desc=train["coursubtitle"]+train["description"]

kickdesc= pd.Series(courses_desc.tolist()).astype(str)
def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1

kickdesc= kickdesc.map(desc_clean)

stop = set(stopwords.words('english'))

kickdesc = [[x for x in x.split() if x not in stop] for x in kickdesc]

stemmer = SnowballStemmer(language='english')

kickdesc = [[stemmer.stem(x) for x in x] for x in kickdesc]

kickdesc = [[x for x in x if len(x) > 2] for x in kickdesc]

kickdesc = [' '.join(x) for x in kickdesc]

tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(kickdesc)

Total=cosine_similarity(tfidf_train,tfidf_train)

def get_similar_courses(id):
    db = pymysql.connect("localhost","root","","internship")
    train = pd.read_sql_query("select * from newcourses;", db)

    courses=train["coursetitle"] 
    courses_desc=train["coursubtitle"]+train["description"]
    c_id=list(train["courseid"])
    index=c_id.index(id)

    num_of_recommendations=6
    
    top_idx = np.argsort(Total[index])[-num_of_recommendations:]
    top_values = [Total[index][i] for i in top_idx]
    
    top_idx=top_idx[::-1]
    top_values=top_values[::-1]
    
    top_idx=np.delete(top_idx,0)
    top_values=np.delete(top_values,0)

    odds = []
    primes = odds

    for i in range(0,num_of_recommendations-1):
        primes.append(c_id[top_idx[i]])
    
    return(primes)   
app = Flask(__name__)

@app.route("/update_matrix_db", methods=['GET'])
def update_matrix_db():
    cursor.execute("TRUNCATE TABLE cosine_similarity_table;")
    for course_id in course_ids:
        similar_courses = get_similar_courses(str(course_id))
        '''print(similar_courses[0],'hi')
        print(similar_courses[4],'hello')
        #for course in similar_courses:'''
        cursor.execute("INSERT INTO `cosine_similarity_table`(`course_id`, `similar_1`,`similar_2`,`similar_3`,`similar_4`,`similar_5`) VALUES ('"+course_id+"', '"+similar_courses[0]+"','"+similar_courses[1]+"','"+similar_courses[2]+"','"+similar_courses[3]+"','"+similar_courses[4]+"')")
    cursor.close()
    print('hi')	
    db.commit() 
    return jsonify({"stat":True})

if __name__ == '__main__':
    # app.run(debug = True)
    app.debug = True
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0',port = port) 
