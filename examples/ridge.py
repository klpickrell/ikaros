#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, cross_val_score

from mlxtend.classifier import StackingClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn import over_sampling, pipeline as pl
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from functools import partial

def sample_loss( estimator, X, y ):
    return cross_val_score(estimator,
                           X,
                           y=y,
                           scoring='roc_auc',
                           cv=2).mean()

def _main():

    RANDOM_STATE=42

    X, y = make_classification(n_samples=10000,
                               n_features=10,
                               n_informative=8,
                               n_redundant=2)


    encoder = LabelEncoder().fit(y)
    X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2)
    y_train = encoder.fit_transform(y_train)

    num_neighbors = 1
    num_svc = 1
    num_forests = 1

    estimators = []
    estimators.extend( [ KNeighborsClassifier() for i in xrange(num_neighbors) ] )
    estimators.extend( [ SVC() for i in xrange(num_svc) ] )
    estimators.extend( [ RandomForestClassifier() for i in xrange(num_forests) ] )

    print( 'building StackingClassifier' )
    model = StackingClassifier( classifiers=estimators,
                                meta_classifier=LogisticRegression() )
    resampler = over_sampling.SMOTE(random_state=RANDOM_STATE, k_neighbors=20)
    X_u,y_u = resampler.fit_sample(X_train,y_train)
    if encoder:
        model.fit( X_u, encoder.fit_transform(y_u) )
    else:
        model.fit( X_u, y_u )

    classification_report([('StackingClassifier',model)],X_test,encoder.fit_transform(y_test))
    print( sample_loss( model, X_test, encoder.fit_transform(y_test) ) )

    return 0

def classification_report(estimators,X,y):
    '''
    classification_report: Generate confusion matrices, call sklearn.metrics.classification_report for
    all estimators on X and y
    '''
    for name, estimator in estimators:
        print( '-'*20 + name + '-'*20 )
        preds = estimator.predict(X)
        print( 'cohen kappa: {}'.format(metrics.cohen_kappa_score(preds, y)) )
        print( metrics.classification_report(preds, y) )



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
