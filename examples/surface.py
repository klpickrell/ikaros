#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
#    s1.add( VarianceThreshold, params={ 'threshold' : continuous( 0, 0.05 ) } )
    s1.add( StandardScaler )

    s2 = StackedStage()
    s2.add( KNeighborsClassifier, params={ 'n_neighbors' : 5,
                                           'weights' : 'uniform' } )
    s2.add( RandomForestClassifier, params={ 'n_estimators' : 350 } )
    s2.add( SVC, params={ 'gamma' : continuous(1,4) } )
    s2.add( LogisticRegression, params={ 'C' : continuous(0.001,10) } )

    ik = IPipeline( stages=[s1,s2], loss=partial(sample_loss,X=X,y=y), verbose=True )

    ik.fit( X, y )
    ik.optimize(n_iterations=30)

    bpipeline = ik.pipelines[0]
    yest = np.array(bpipeline.stage_loss)
    Xvest = np.array([(i['SVC__gamma'],i['LogisticRegression__C']) for i in bpipeline.stage_space])

    bpipeline.bootstrap(bootstrap=100)

    y = np.array(bpipeline.stage_loss)
    X = np.array([(i['SVC__gamma'],i['LogisticRegression__C']) for i in bpipeline.stage_space])

    optimum = y[y.argmax()]
    optimum_X = X[y.argmax()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title( 'sample loss surface' )
    ax.set_xlabel( 'gamma')
    ax.set_ylabel( 'C' )
    ax.set_xlim(1,4)
    ax.set_ylim(0,10)
    cp = ax.tricontourf(X[:,0], X[:,1], y, 100, cmap='viridis')
    plt.colorbar(cp)
    ax.scatter(X[:,0], X[:,1], color='blue', s=20)
    ax.scatter(optimum_X[0], optimum_X[1], marker='*', c='gold', s=150)
    ax.annotate('{}'.format(optimum), xy=(optimum_X[0], optimum_X[1]), xytext=(optimum_X[0]+0.2, optimum_X[1]+0.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )

    plt.savefig( '../images/kernelsurface.png' )
    plt.show()

    return 0



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
