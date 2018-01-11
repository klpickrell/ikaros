#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import lightgbm.sklearn as lgb
from catboost import CatBoostClassifier

from ikaros.pipeline import IPipeline
from ikaros.stages   import SequentialStage, StackedStage
from ikaros.sampling import discrete, continuous

from functools import partial

def sample_loss( estimator, X, y ):
    return cross_val_score(estimator,
                           X,
                           y=y,
                           scoring='roc_auc',
                           cv=2).mean()

def _main():

    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=8,
                               n_redundant=2)

    s1 = SequentialStage()
    s1.add( VarianceThreshold, params={ 'threshold' : continuous( 0, 0.05 ) } )
    s1.add( StandardScaler )
#    s1.add( SMOTE, params={ 'k_neighbors' : discrete( [ 5, 6, 7 ] ) } )

    s2 = StackedStage()
    s2.add( KNeighborsClassifier, params={ 'n_neighbors' : discrete( xrange(5,30,8) ),
                                           'weights' : discrete( [ 'uniform', 'distance' ] ) } )
    s2.add( RandomForestClassifier, params={ 'n_estimators' : discrete( xrange(10,500,200) ) } )
    s2.add( LogisticRegression, params={ 'C' : continuous(0.001,10) } )

    ik = IPipeline( stages=[s1,s2], loss=partial(sample_loss,X=X,y=y), verbose=True )

    ik.fit( X, y )
    ik.optimize(10)

    print( '{}: {}'.format(ik.best_score_, ik.best_estimator_.get_params()) )

    return 0



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
