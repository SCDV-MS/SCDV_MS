import time
import warnings
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
import os
from numpy import float32
from gensim.models.keyedvectors import KeyedVectors
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
import cPickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize


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


for covar_type in ["full"]:
  for sparsity_k in [2]:
    print covar_type
    print sparsity_k
    num_features = 200
    start = time.time()
    num_clusters = 40
    WORD_EMBED_DIR_DOC2VEC = "Word_Vectors/doc2vecC"
    embedding_type = "doc"  # "sen" #doc and sen are two embedding type where embedding has
    sparsity = 1  # 0 and 1 are two values
    model_name = "Doc2VecC_Polysemy_200.txt"

    model = KeyedVectors.load_word2vec_format(os.path.join(WORD_EMBED_DIR_DOC2VEC, model_name), binary=False)
    word_vectors = model.syn0

    #all = pd.read_pickle('data/all.pkl')


    # Load train data.
    train = pd.read_csv('data/train_v2.tsv', header=0, delimiter="\t")
    # Load test data.
    test = pd.read_csv('data/test_v2.tsv', header=0, delimiter="\t")
    all = pd.read_csv('data/all_v2.tsv', header=0, delimiter="\t")
    num_clusters = 40#int(sys.argv[2])
    idx, idx_proba = cluster_GMM(num_clusters, word_vectors, covar_type)
    # Set number of clusters.

    # Uncomment below line for creating new clusters.
    #idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    # idx_name = "gmm_latestclusmodel_len2alldata.pkl"
    # idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
    # idx, idx_proba = read_GMM(idx_name, idx_proba_name)
    print sum(sum(idx_proba))
    index = np.argsort(np.argsort(idx_proba, axis=1), axis=1) < (num_clusters - sparsity_k)
    idx_proba[index]=0
    print sum(sum(idx_proba))
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

    # Computing tf-idf values.
    traindata = []
    for i in range(0, len(all["news"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["news"][i], True)))

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

    # gwbowv is a matrix which contains normalised document vectors.
    gwbowv = np.zeros((train["news"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    min_no = 0
    max_no = 0
    INPUT_DIR ="data"
    e = open(os.path.join(INPUT_DIR, '20_newsgroup_tokenized_multisense_sentences_1000_train.txt'), "r")
    g = open(os.path.join(INPUT_DIR, "20_newsgroup_tokenized_multisense_sentences_1000_test.txt"), "r")
    train_lines = e.readlines()
    train_lines_mod = [line[:-1].split() for line in train_lines]
    test_lines = g.readlines()
    test_lines_mod = [line[:-1].split() for line in test_lines]
    for review in train["news"]:
        # Get the wordlist in each news article.
        words = train_lines_mod[counter]
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                           word_centroid_prob_map, num_features, word_idf_dict,
                                                           featurenames, num_clusters, train=True)
        counter += 1
        if counter % 1000 == 0:
            print "Train News Covered : ", counter

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

    gwbowv_test = np.zeros((test["news"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test["news"]:
        # Get the wordlist in each news article.
        words = test_lines_mod[counter]
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                                word_centroid_prob_map, num_features, word_idf_dict,
                                                                featurenames, num_clusters)
        counter += 1
        if counter % 1000 == 0:
            print "Test News Covered : ", counter

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_gmm_sparse.npy"

    print "Making sparse..."
    # Set the threshold percentage for making it sparse.
    percentage = 0.04
    min_no = min_no * 1.0 / len(train["news"])
    max_no = max_no * 1.0 / len(train["news"])
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
    print "SDV created and dumped: ", endtime, "seconds."
    print "Fitting a SVM classifier on labeled training data..."

    param_grid = [
        {'C': np.arange(0.1, 5, 0.1)}]
    scores = ['accuracy']#, 'recall_micro', 'f1_micro', 'precision_micro', 'recall_macro', 'f1_macro', 'precision_macro',
              #'recall_weighted', 'f1_weighted', 'precision_weighted']  # , 'accuracy', 'recall', 'f1']
    for score in scores:
        strt = time.time()
        print "# Tuning hyper-parameters for", score, "\n"
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring='%s' % score)
        clf.fit(gwbowv, train["class"])
        print "Best parameters set found on development set:\n"
        print clf.best_params_
        print "Best value for ", score, ":\n"
        print clf.best_score_
        Y_true, Y_pred = test["class"], clf.predict(gwbowv_test)
        print "Report"
        print classification_report(Y_true, Y_pred, digits=6)
        print "Accuracy: ", clf.score(gwbowv_test, test["class"])
        print "Time taken:", time.time() - strt, "\n"
    endtime = time.time()
    print "Total time taken: ", endtime - start, "seconds."

    print "********************************************************"

