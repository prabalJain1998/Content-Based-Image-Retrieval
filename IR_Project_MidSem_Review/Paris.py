import os
import cv2
import copy
import csv
import random
import pickle
import numpy as np
import pandas as pd
import itertools
from scipy.stats import randint
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern

images_path = 'Paris/'


def read_images(datapath):
    labels = ['defense', 'eiffel', 'general', 'invalides', 'louvre', 'moulinrouge', 'museedorsay', 'notredame',
              'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']
    mapping = {'defense': 0, 'eiffel': 1, 'general': 2, 'invalides': 3, 'louvre': 4, 'moulinrouge': 5, 'museedorsay': 6,
               'notredame': 7, 'pantheon': 8, 'pompidou': 9, 'sacrecoeur': 10, 'triomphe': 11}
    images = []
    Imglabels = []
    num1 = 224
    num2 = 224
    for label in labels:
        path = os.path.join(datapath, label)
        for img in os.listdir(path):
            print(os.path.join(path, img))
            img = cv2.imread(os.path.join(path, img))
            new_img = cv2.resize(img, (num2, num1))
            images.append(new_img)
            Imglabels.append(mapping[label])
    return np.array(images), np.array(Imglabels)


def get_hog(images):
    result = np.array([hog(img, block_norm='L2') for img in images])

    return result


def get_sift(images):
    # SIFT descriptor for 1 image
    def get_image_sift(image, vector_size=15):
        alg = cv2.xfeatures2d.SIFT_create()
        kps = alg.detect(image, None)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        # Making descriptor of same size
        # Descriptor vector size is 128
        needed_size = (vector_size * 128)
        if len(kps) == 0:
            return np.zeros(needed_size)

        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

        return dsc

    # SIFT descriptor for all images
    features = []
    for i, img in enumerate(images):
        dsc = get_image_sift(img)
        features.append(dsc)

    result = np.array(features)

    return result


def get_kaze(images):
    # KAZE descriptor for 1 image
    def get_image_kaze(image, vector_size=32):
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if len(kps) == 0:
            return np.zeros(needed_size)

        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()

        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc

    # KAZE descriptor for all images
    features = []
    for i, img in enumerate(images):
        dsc = get_image_kaze(img)
        features.append(dsc)

    result = np.array(features)

    return result


def combine_features(features, horizontal=True):
    """
    Array of features [f1, f2, f3] where each fi is a feature set
    eg. f1=rgb_flat, f2=SIFT, etc.
    """
    if horizontal:
        return np.hstack(features)
    else:
        return np.vstack(features)


def norm_features_minmax(train):
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train = min_max_scaler.fit_transform(train)

    return norm_train


def get_lbp(images):
    result = np.array([local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 10, 3).flatten() for img in images])

    return result


if __name__ == "__main__":
    full_x, full_y = read_images(images_path)
    print(full_x.shape)
    print(full_y.shape)
    np.save('Paris_X_train', full_x)
    np.save('Paris_label', full_y)

    full_x = np.load('Paris_X_train.npy')
    print(full_x.shape)

    labels = np.load('Paris_label.npy')
    temp_count = 0
    count = {0: 0}
    j = 1
    for i in range(len(labels) - 1):
        if labels[i + 1] == labels[i]:
            temp_count += 1
        else:
            count[j] = temp_count + 1
            temp_count = temp_count + 1
            j += 1
    count[12] = full_x.shape[0] - 1
    print(count)
    print(labels[6331])
    # cv2.imshow('Query Image', full_x[517])
    # cv2.waitKey(0)

    # HOG Features
    hog_features = get_hog(full_x)
    print(hog_features.shape)
    np.save('Paris_HOG', hog_features)

    lbp_features = get_lbp(full_x)
    lbp_features = np.load('Paris_LBP.npy')
    print(lbp_features.shape)
    np.save('Paris_LBP', lbp_features)

    sift_features = get_sift(full_x)
    sift_features = np.load('Paris_SIFT.npy')
    print(sift_features.shape)
    np.save('Paris_SIFT', sift_features)

    kaze_features = get_kaze(full_x)
    kaze_features = np.load('Paris_KAZE.npy')
    print(kaze_features.shape)
    np.save('Paris_KAZE', kaze_features)

    # Normalization
    hog_features = np.load('Paris_HOG.npy')
    norm_hog_features = norm_features_minmax(hog_features)
    print(norm_hog_features.shape)
    np.save('Paris_norm_HOG',norm_hog_features)

    hog_features = np.load('Caltech_HOG.npy')
    norm_hog_features = norm_features_minmax(hog_features)
    print(norm_hog_features.shape)
    np.save('Caltech_norm_HOG', norm_hog_features)

    lbp_features = np.load('Paris_LBP.npy')
    norm_lbp_features = norm_features_minmax(lbp_features)
    print(norm_lbp_features.shape)
    np.save('Paris_norm_LBP', norm_lbp_features)

    sift_features = np.load('Paris_SIFT.npy')
    norm_sift_features = norm_features_minmax(sift_features)
    print(norm_sift_features.shape)
    np.save('Paris_norm_SIFT', norm_sift_features)

    kaze_features = np.load('Paris_KAZE.npy')
    norm_kaze_features = norm_features_minmax(kaze_features)
    print(norm_kaze_features.shape)
    np.save('Paris_norm_KAZE', norm_kaze_features)

    # PCA
    hog_norm_features = np.load('Paris_norm_HOG.npy')
    pca = PCA(n_components=1500)
    pca_hog_features = pca.fit_transform(hog_norm_features)
    np.save('Paris_HOG_PCA', pca_hog_features)
    pca_hog_features = np.load('Paris_HOG_PCA.npy')
    print(pca_hog_features.shape)

    hog_norm_features = np.load('Caltech_norm_HOG.npy')
    pca = PCA(n_components=1500)
    pca_hog_features = pca.fit_transform(hog_norm_features)
    np.save('Paris_HOG_PCA.npy', pca_hog_features)
    # pca_hog_features = np.load('Paris_HOG_PCA.npy')
    print(pca_hog_features.shape)

    sift_norm_features = np.load('Paris_norm_SIFT.npy')
    print(sift_norm_features.shape)
    pca = PCA(n_components=400)
    pca_sift_features = pca.fit_transform(sift_norm_features)
    np.save('Paris_SIFT_PCA', pca_sift_features)
    pca_sift_features = np.load('Paris_SIFT_PCA.npy')
    print(pca_sift_features.shape)

    kaze_norm_features = np.load('Paris_norm_KAZE.npy')
    print(kaze_norm_features.shape)
    pca = PCA(n_components=500)
    pca_kaze_features = pca.fit_transform(kaze_norm_features)
    np.save('Paris_KAZE_PCA', pca_kaze_features)
    pca_kaze_features = np.load('Paris_KAZE_PCA.npy')
    print(pca_kaze_features.shape)

    lbp_norm_features = np.load('Paris_norm_LBP.npy')
    print(lbp_norm_features.shape)
    pca = PCA(n_components=3000)
    pca_lbp_features = pca.fit_transform(lbp_norm_features)
    np.save('Paris_LBP_PCA', pca_lbp_features)
    pca_lbp_features = np.load('Paris_LBP_PCA.npy')
    print(pca_lbp_features.shape)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.show()

    pca_surf_features = np.load('Paris_SURF_PCA.npy')
    print(pca_surf_features.shape)

    # Combine Features

    features_pca_5500 = None

    for t in (pca_hog_features, pca_sift_features, pca_kaze_features, pca_lbp_features, pca_surf_features):
        if features_pca_5500 is None:
            features_pca_5500 = t
        else:
            features_pca_5500 = combine_features([features_pca_5500, t])

    print(features_pca_5500.shape)
    np.save('Paris_combined_5500', features_pca_5500)

    features_pca_2500 = None
    for t in (pca_hog_features, pca_sift_features, pca_kaze_features, pca_surf_features):
        if features_pca_2500 is None:
            features_pca_2500 = t
        else:
            features_pca_2500 = combine_features([features_pca_2500, t])

    print(features_pca_2500.shape)
    np.save('Paris_combined_2500', features_pca_2500)

    surf_norm_features = np.load('Paris_norm_SURF.npy')

    print(labels.shape)

    # LDA
    lda = LDA()
    lda_features = lda.fit_transform(hog_norm_features, labels)
    np.save('Caltech_HOG_LDA', lda_features)
    print(lda_features.shape)

    lda_hog_features = np.load('Paris_HOG_LDA.npy')
    lda_sift_features = np.load('Paris_SIFT_LDA.npy')
    lda_kaze_features = np.load('Paris_KAZE_LDA.npy')
    lda_surf_features = np.load('Paris_SURF_LDA.npy')

    # Combine LDA Features
    features_lda= None
    for t in (lda_hog_features, lda_sift_features, lda_kaze_features, lda_surf_features):
        if features_lda is None:
            features_lda = t
        else:
            features_lda = combine_features([features_lda, t])

    print(features_lda.shape)
    np.save('Paris_combined_LDA', features_lda)

    query_image = cv2.imread('paris_eiffel_000284.jpg')
    query_new_img = cv2.resize(query_image, (224, 224))
    cv2.imshow('query_image', query_image)
    cv2.waitKey(0)

    paris_combined_lda = np.load('Paris_combined_LDA.npy')
    paris_combined_pca = np.load('Paris_combined_5500.npy')
    paris_predict = np.load('Paris_combined_LDA.npy')
    paris_match = np.load('Paris_combined_2500.npy')
    print(paris_predict.shape)
    print(paris_match.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    print(labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(paris_predict, labels, test_size=0.2)
    print(y_train)
    model1 = XGBClassifier()
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    pickle.dump(model1, open("XGB_model", 'wb'))

    class_report = classification_report(y_test, y_pred1, output_dict=True)
    print('Precision =', class_report['macro avg']['precision'])
    print('Recall = ', class_report['macro avg']['recall'])
    print('F1-score =', class_report['macro avg']['f1-score'])
    print('Accuracy =', class_report['accuracy'])
    acc1 = accuracy_score(y_pred1, y_test)
    print(acc1)

    index = [count[1] + 17]
    test_labels = []
    query_image = []
    query_image_match = []
    show = full_x[index[0]]

    for i in range(len(index)):
        query_image.append(paris_predict[index[i]])
        query_image_match.append(paris_match[index[i]])
    query_image = np.array(query_image)
    query_image_match = np.array(query_image_match)
    print(query_image.shape)
    print(query_image_match.shape)

    for i in range(len(index)):
        paris_match = np.delete(paris_match, index[i], axis=0)
        paris_predict = np.delete(paris_predict, index[i], axis=0)
        full_x = np.delete(full_x, index[i], axis=0)
    print(paris_predict.shape)

    for i in range(len(index)):
        test_labels.append(labels[index[i]])

    for i in range(len(index)):
        labels = np.delete(labels, index[i], axis=0)
    print(labels.shape)

    model = XGBClassifier()
    model.fit(paris_predict, labels)
    y_pred = model.predict(query_image)

    print(y_pred)
    acc = accuracy_score(y_pred, test_labels)
    print(acc)

    dist = {}
    for i in range(count[y_pred[0]], count[y_pred[0] + 1]):
        dist1 = spatial.distance.cosine(paris_match[i], query_image_match[0])
        dist[i] = dist1

    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    print(len(dist))
    print(dist)
    cv2.imwrite('Query Image.jpg', show)
    for i in range(10):
        val = list(dist.keys())[i]
        s = "image1_"+str(i)+".jpg"
        cv2.imwrite(s, full_x[val])
        cv2.waitKey(0)

    # Combine LDA Features
    query_new_features_lda = None
    for t in (pca_hog_features, pca_sift_features, pca_kaze_features, lda_surf_features):
        if query_new_features_lda is None:
            query_new_features_lda = t
        else:
            query_new_features_lda = combine_features([query_new_features_lda, t])

    print(query_new_features_lda.shape)
