# word2vec
import gensim.models.keyedvectors as word2vec
def loadModel(num_words):
    model = word2vec.KeyedVectors.load_word2vec_format('C:\\Users\\Darko\\Downloads\\is\\GoogleNews-vectors-negative300.bin', binary=True, limit=num_words)
    return model


# loadiranje na mnozhestvoto
import pandas as pd
full_set = pd.read_csv("D:\\Darko\\Fax\\IS\\ProektNew\\data job posts.csv", header=0, delimiter=",", quotechar='"')


# formatiranje na tekst
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
lem = WordNetLemmatizer()
ps = PorterStemmer()

def filtered_words(sample, method, result):
    letters_only = re.sub("[^a-zA-Z]", " ", sample) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))
    if method == "normal":
        meaningful_words = [w for w in words if not w in stops]
    elif method == "lemma":
        meaningful_words = [lem.lemmatize(w) for w in words if not w in stops]
    else:
        meaningful_words = [lem.lemmatize(ps.stem(w)) for w in words if not w in stops]
    if result == "string":
        return " ".join(meaningful_words)
    else:
        return meaningful_words
    
    
num_features = 300

import numpy as np
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
	#u set poso bilo pobrzo taka
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(posts, model, num_features):
    counter = 0.
    postFeatureVecs = np.zeros((len(posts), num_features), dtype="float32")
    for post in posts:
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(posts))
        postFeatureVecs[int(counter)] = makeFeatureVec(post, model, num_features)
        counter = counter + 1.
    return postFeatureVecs


# kreiranje na vektori od mnozestvoto
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def createVectors(train_set, method, features):   
    if method == "tf-idf":
        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = features)
    else:
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = features)
    train_data_features = vectorizer.fit_transform(train_set)
    train_data_features = train_data_features.toarray()
    return train_data_features
        

# klasteriranje
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from spherecluster import SphericalKMeans
import time
def clustering(train_set, method, params):  
    start = time.time()
    if method == "KmeansEuc":
        kmeans_clustering = KMeans(n_clusters = params)
        idx = kmeans_clustering.fit_predict(train_set)
    elif method == "KmeansCos":
        kmeans_clustering = SphericalKMeans(n_clusters = params)
        idx = kmeans_clustering.fit_predict(train_set)
    elif method == "Agg":
        kmeans_clustering = AgglomerativeClustering(n_clusters = params, linkage = "average")
        idx = kmeans_clustering.fit_predict(train_set)
    elif method == "DBSCAN":
        kmeans_clustering = DBSCAN(min_samples = params, eps = 0.2)
        idx = kmeans_clustering.fit_predict(train_set)
        print idx    
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."
    word_centroid_map = dict(zip(titles, idx))
    silhoette = metrics.silhouette_score(train_set, idx)
    chs = metrics.calinski_harabaz_score(train_set, idx)
    return word_centroid_map, silhoette, chs

    
# gledanje na najcesti zborovi vo site naslovi, sample=lista od listi od zborovi
def mostFrequent(sample, model):
    if len(sample)==0:
        return 0
    words = []
    for title in sample:
        if isinstance(title, basestring):
            for word in filtered_words(title, "lemma", "list"):
                words.append(word)
    all_words_freq = nltk.FreqDist(words)
    text_file.write("najcesti zborovi za klasterov %s\n\n" % all_words_freq.most_common(3))
    for common_word in all_words_freq.most_common(3):
        #if common_word[0] == "english" or common_word[0] == "adminstrator" or common_word[0] == "qa":
        #    return 0.5
        if common_word[0] in index2word_set:
            similar = [tup[0] for tup in model.most_similar(common_word[0])]
            text_file.write("%s\n" % similar)
    countCommon = 0
    countUncommon = 0
    newTitles = []
    newTitle = ''
    for title in sample:
        #text_file.write("%s\n" % title)
        #print "tuka"
        common = False
        if not isinstance(title, basestring):
            #print "tuka1"
            newTitles.append(title)
            countUncommon += 1
            #text_file.write("se odbira stariot naslov-ne e string title %s\n")
            continue
        words = filtered_words(title, "lemma", "list")
        for common_word in all_words_freq.most_common(3):
            if common == True: break
            if common_word[0] in words:
                common = True
                newTitle = common_word[0]
                #text_file.write("imame common word %s\n" % newTitle)
                break
            if common_word[0] in index2word_set: 
                similar = [tup[0] for tup in model.most_similar(common_word[0])]
                #text_file.write("%s\n" % similar)
                for word in words:
                    if word in similar:
                        common = True
                        newTitle = str(common_word[0])+'-'+str(word)
                        #text_file.write("imame slicen word %s\n" % newTitle)
                        break
        if common == True:
            #print "tuk2"
            newTitles.append(newTitle)
            countCommon += 1
        else:
            #print "tuk3"
            newTitles.append(title)
            countUncommon += 1
            #text_file.write("se odbira stariot naslov\n")
    score = float(countCommon)/(countCommon+countUncommon)
    text_file.write("%d\n%d\n%.2f\n" % (countCommon, countUncommon, score))
    string = ""
    for title in newTitles:
        string += str(title)+','
    string = string[:-1]
    text_file.write("%s\n" % string)
    return score

if __name__ == '__main__':
    
    # kreiranje na recenici od mnozestvoto
    clean_train_set = []
    titles = []
    for i in xrange(1, full_set.shape[0]):
        if((i+1) % 1000 == 0):
            print "Sample %d of %d\n" % (i+1, full_set.shape[0])
        titles.append(full_set["Title"][i])                                                          
        clean_train_set.append(filtered_words(full_set["jobpost"][i], "lemma", "string"))
        
    #vectoredDocs = getAvgFeatureVecs(clean_train_set, model, num_features)
    vectoredDocs = createVectors(clean_train_set, "tf-idf", 300)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 10)
    data_train = pca.fit_transform(vectoredDocs)
    
    del vectoredDocs
    del full_set
    
    #ako e so average vectors ova treba pred getavgfeatures
    model = loadModel(60000)
    print "finished loading model"
    index2word_set = set(model.wv.index2word)
    
    for j in xrange(6, 10):
        for klasteringName in ["Agg", "KmeansCos", "KmeansEuc"]:
            string = 'D:\\Darko\\Fax\\IS\\ProektNew\\Final2\\'+klasteringName+'_'+str(j)+'kl.txt'
            text_file = open(string, "w")
            word_centroid_map, sil, chs = clustering(data_train, klasteringName, j)
            text_file.write("%f, %f\n\n" % (sil, chs))
            
            #print word_centroid_map.values()
            #klasters = len(set(word_centroid_map.values()))
            summ = 0. 
            lengths = []   
            for cluster in xrange(0, j):
                posts = []
                for i in xrange(0,len(word_centroid_map.values())):
                    if(word_centroid_map.values()[i] == cluster):
                        posts.append(word_centroid_map.keys()[i])
                #text_file.write("%s\n\n" % posts)
                lengths.append(len(posts))
                summ += mostFrequent(posts, model)
            npLen = np.array(lengths)
            text_file.write("%.2f\n" % (summ/float(j)))
            text_file.write("%.2f %.2f\n" % (float(np.mean(npLen)), float(np.std(npLen))))
            
            text_file.close()