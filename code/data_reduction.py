from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def pca_reducer(values, ncomponents = 6):
    normalized = MinMaxScaler().fit_transform(values)
    pca = PCA(n_components = ncomponents)
    pca.fit(normalized.T)

    return  MinMaxScaler().fit_transform(pca.components_.T)


def pls_reducer(values, y, ncomponents = 6):
    normalized_values = MinMaxScaler().fit_transform(values)
    X = normalized_values

    pls = PLSRegression(n_components=ncomponents)
    pls_score = pls.fit_transform(X, y)

    return MinMaxScaler().fit_transform(pls_score[0])