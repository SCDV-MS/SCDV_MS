import time;
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture
import time;
import gc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
import sys
from random import uniform
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from gensim.models.keyedvectors import KeyedVectors
import os, pdb
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pickle

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def cluster_GMM(num_clusters, word_vectors,covar_type):
    # Initalize a GMM object and use it for clustering.
    clf = GaussianMixture(n_components=num_clusters,
                          covariance_type=covar_type, init_params='kmeans', max_iter=100)
    # Get cluster assignments.
    start = time.time()
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print "Clustering Done...", time.time() - start, "seconds"
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments.
    joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
    print "Cluster Assignments Saved..."

    joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
    print "Probabilities of Cluster Assignments Saved..."
    return (idx, idx_proba)


def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print "Cluster Model Loaded..."
    return (idx, idx_proba)


def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors.

    prob_wordvecs = {}
    for word in word_centroid_map:
        if(word in word_idf_dict):
            prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
            for index in range(0, num_clusters):
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model[word] * \
                                                                                       word_centroid_prob_map[word][index] * \
                                                                                       word_idf_dict[word]

    return prob_wordvecs


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension,
                                     word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    global min_no
    global max_no

    for word in wordlist:
        if(word in prob_wordvecs):
            bag_of_centroids += prob_wordvecs[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids





# Set number of clusters.
#int(sys.argv[2])
for sparsity_k in [7]:
  covar_type ="full"
  for min_count in [5]:
    print covar_type
    print "min_count",min_count
    print "sparsity_k",sparsity_k
    num_clusters = 40
    WORD_EMBED_DIR_DOC2VEC = "Word_Vectors/doc2vecC"
    INPUT_DIR = "data/"
    embedding_type = "doc"  # "sen" #doc and sen are two embedding type where embedding has
    sparsity = 1  # 0 and 1 are two values
    if embedding_type == "doc":
        model_name = "wordvectors_reuters_polysemy_"+str(min_count)+".txt"
    else:
        model_name = "wordvectors_reuters_nonpoly_sen.txt"
    model = KeyedVectors.load_word2vec_format(os.path.join(WORD_EMBED_DIR_DOC2VEC, model_name), binary=False)
    word_vectors = model.syn0

    all = pd.read_pickle('data/all.pkl')
    idx, idx_proba = cluster_GMM(num_clusters, word_vectors,covar_type)

    start = time.time()



    num_features = 200  # int(sys.argv[1])  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words




    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    idx_name = "gmm_latestclusmodel_len2alldata.pkl"
    idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
    #idx, idx_proba = read_GMM(idx_name, idx_proba_name)
    #pdb.set_trace()
    print sum(sum(idx_proba))
    index = np.argsort(np.argsort(idx_proba, axis=1), axis=1) < (num_clusters - sparsity_k)
    print (index[0])
    idx_proba[index] = 0
    print sum(sum(idx_proba))
    #continue
    #pdb.set_trace()
    #idx_proba = (idx_proba/(np.sum(idx_proba,axis=1).reshape(17565,1)))
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

    # Computing tf-idf values.
    traindata = []
    for i in range(0, len(all["text"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["text"][i], True)))

    tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
    tfidfmatrix_traindata = tfv.fit_transform(traindata)
    featurenames = tfv.get_feature_names()
    idf = tfv._tfidf.idf_

    # Creating a dictionary with word mapped to its idf value
    print "Creating word-idf dictionary for Training set..."

    word_idf_dict = {}
    for pair in zip(featurenames, idf):
        word_idf_dict[pair[0]] = pair[1]
        for each_num in ["first","second","third","fourth","fifth"]:
            word_idf_dict[pair[0]+each_num] = pair[1]

    # Pre-computing probability word-cluster vectors.
    prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

    temp_time = time.time() - start
    print "Creating Document Vectors...:", temp_time, "seconds."

    # Create train and text data.
    lb = MultiLabelBinarizer()
    Y = lb.fit_transform(all.tags)
    train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

    train = DataFrame({'text': []})
    test = DataFrame({'text': []})

    train["text"] = train_data.reset_index(drop=True)
    test["text"] = test_data.reset_index(drop=True)
    train_docs, Y_train, test_docs, Y_test = pickle.load(
        open(os.path.join(INPUT_DIR, "train_text_annotate_sets.pkl"), "rb"))
    # gwbowv is a matrix which contain normalised normalised gwbowv.
    gwbowv = np.zeros((train["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    min_no = 0
    max_no = 0
    for review in train["text"]:
        # Get the wordlist in each text article.
        words = train_docs[counter]
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                           word_centroid_prob_map, num_features, word_idf_dict,
                                                           featurenames, num_clusters, train=True)
        counter += 1
        if counter % 1000 == 0:
            print "Train text Covered : ", counter

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

    endtime_gwbowv = time.time() - start
    print "Created gwbowv_train: ", endtime_gwbowv, "seconds."

    gwbowv_test = np.zeros((test["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test["text"]:
        # Get the wordlist in each text article.
        words = test_docs[counter]
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                                word_centroid_prob_map, num_features, word_idf_dict,
                                                                featurenames, num_clusters)
        counter += 1
        if counter % 1000 == 0:
            print "Test Text Covered : ", counter

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_gmm_sparse.npy"

    print "Making sparse..."
    # Set the threshold percentage for making it sparse.
    percentage = 0.04
    min_no = min_no * 1.0 / len(train["text"])
    max_no = max_no * 1.0 / len(train["text"])
    print "Average min: ", min_no
    print "Average max: ", max_no
    thres = (abs(max_no) + abs(min_no)) / 2
    thres = thres * percentage

    # Make values of matrices which are less than threshold to zero.
    temp = abs(gwbowv) < thres
    #gwbowv[temp] = 0

    temp = abs(gwbowv_test) < thres
    #gwbowv_test[temp] = 0

    # saving gwbowv train and test matrices
    np.save(gwbowv_name, gwbowv)
    np.save(test_gwbowv_name, gwbowv_test)

    endtime = time.time() - start
    print "Total time taken: ", endtime, "seconds."

    print "********************************************************"
    print "Fitting One vs Rest SVM classifier to labeled cluster training data..."

    start = time.time()

    del traindata,test["text"],train["text"],train_data, test_data,model, word_vectors,word_idf_dict,all, prob_wordvecs,word_centroid_prob_map,tfv,tfidfmatrix_traindata,featurenames ,idf
    gc.collect()

    param_grid = [
        {'estimator__C': np.arange(20, 80, 20)}
        # {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.01, 0.05, 0.001, 0.005,  0.0001], 'kernel': ['rbf']},
    ]
    scores = ['f1_weighted']  # , 'accuracy', 'recall', 'f1']
    for score in scores:
        strt = time.time()
        print "# Tuning hyper-parameters for", score, "\n"
        clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C=100.0), n_jobs=15), param_grid, cv=5, n_jobs=10,
                           scoring='%s' % score)
        clf = OneVsRestClassifier(LogisticRegression(C=80.0), n_jobs=15)
        clf = clf.fit(gwbowv, Y_train)

        pred = clf.predict(gwbowv_test)
        pred_proba = clf.predict_proba(gwbowv_test)
        #pdb.set_trace()
        K = [1, 3, 5]

        for k in K:
            Total_Precision = 0
            Total_DCG = 0
            norm = 0
            for i in range(k):
                norm += 1 / math.log(i + 2)

            loop_var = 0
            for item in pred_proba:
                classelements = sorted(range(len(item)), key=lambda i: item[i])[-k:]
                classelements.reverse()
                precision = 0
                dcg = 0
                loop_var2 = 0
                for element in classelements:
                    if Y_test[loop_var][element] == 1:
                        precision += 1
                        dcg += 1 / math.log(loop_var2 + 2)
                    loop_var2 += 1

                Total_Precision += precision * 1.0 / k
                Total_DCG += dcg * 1.0 / norm
                loop_var += 1
            print "Precision@", k, ": ", Total_Precision * 1.0 / loop_var
            print "NDCG@", k, ": ", Total_DCG * 1.0 / loop_var

        print "Coverage Error: ", coverage_error(Y_test, pred_proba)
        print "Label Ranking Average precision score: ", label_ranking_average_precision_score(Y_test, pred_proba)
        print "Label Ranking Loss: ", label_ranking_loss(Y_test, pred_proba)
        print "Hamming Loss: ", hamming_loss(Y_test, pred)
        print "Weighted F1score: ", f1_score(Y_test, pred, average='weighted')
        #pdb.set_trace()
        # print "Total time taken: ", time.time()-start, "seconds."

    endtime = time.time()
    print "Total time taken: ", endtime - start, "seconds."

    print "********************************************************"