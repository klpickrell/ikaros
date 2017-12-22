import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, accuracy_score, roc_curve, precision_recall_fscore_support, classification_report
from sklearn.datasets import make_classification, make_circles, make_moons
from gp import bayesian_optimization

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class GentleApplicator(OneHotEncoder):
    def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float64, sparse=True, handle_unknown='error', loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=1e-7, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto'):

        super(GentleApplicator, self).__init__(n_values=n_values,
                                               categorical_features='all',
                                               dtype=np.float64,
                                               sparse=True,
                                               handle_unknown='error')
        self._forest = GradientBoostingClassifier(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start,
            presort=presort)

        self._encoder = OneHotEncoder()

    def get_params(self, deep=False):
        out = super(GentleApplicator, self).get_params(deep=deep)
        out.update( self._forest.get_params(deep=deep) )
        return out

    def fit( self, X, y=None, **fit_params ):
        self._forest.fit(X,y,**fit_params)
        return super(GentleApplicator,self).fit(X,y)

    def transform( self, X, y=None, **fit_params ):
        return super(GentleApplicator,self).transform( self._forest.apply(X)[:,:,0] )

    def fit_transform( self, X, y=None, **fit_params ):
        self._forest.fit(X,y,**fit_params)
        return super(GentleApplicator,self).fit_transform( self._forest.apply(X)[:,:,0] )


def sample_loss( search_params, pipeline ):
    return cross_val_score( pipeline(param_grid=search_params) )

def _main():
    
    estimators = [ ('variance', VarianceThreshold()),
                   ('standardize', StandardScaler()),
                   ('gboost', GentleApplicator()),
                   ('lr', LogisticRegression()) ]

    pipeline = Pipeline(estimators)
    search_params = { 'variance__threshold' : np.linspace( 0.95*(1-0.95), 0, 2 ),
                      'gboost__n_estimators' : range(5,500),
                      'lr__C' : [0.001,0.01,0.1,1,10,100],
                      'lr__penalty' : [ 'l1', 'l2' ]
                    }

#    X,y = make_classification(10000)
#    X,y = make_moons(10000)
    X,y = make_circles(10000)

#    bayesian_optimization( 5, sample_loss

#    X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.8)
#    search = GridSearchCV( pipeline, param_grid=search_params, verbose=10 )
#    search.fit( X, y )


    print( 'best estimator:' )
    print( search.best_estimator_ )
    print( 'best params:' )
    print( search.best_params_ )
    print( 'best score:' )
    print( search.best_score_ )
    print( 'the best' )
    print( '---optimized gradientboosting kernel approximation pipeline on circles---' )
    print( classification_report( y_test, search.predict( X_test ).round() ) )

    print( '---logistic regression on circles---' )
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    print( classification_report( y_test, lr.predict(X_test).round() ) )

#    print( '---default gradientboosting kernel approximation---' )
#    grd = GradientBoostingClassifier( n_estimators=10 )
#    encoder = OneHotEncoder()
#    lr = LogisticRegression()
#    grd.fit(X_train,y_train)
#    encoder.fit( grd.apply(X_train)[:,:,0] )
#    lr.fit( encoder.transform(grd.apply(X_train)[:,:,0]), y_train )
#    y_true, y_pred = y_test, lr.predict( encoder.transform(grd.apply(X_test)[:,:,0]) ).round()
#    print(classification_report(y_true, y_pred))

#    estimators = [('standardize', StandardScaler()), ('gboost', GradientBoostingClassifier()), ('clf', LogisticRegression())]
#    pipeline = Pipeline(estimators)
#
#    param_grid = dict(gboost__n_estimators=range(5,50),
#                      clf__C=[0.01, 0.1, 10, 100])
#
#    grid_search = GridSearchCV(pipeline, param_grid=param_grid)

#    scores = ['precision', 'recall', 'f1']
#    scores = ['f1']
#    
#    for score in scores:
#        print("# Tuning hyper-parameters for %s" % score)
#        print()
#    
#        clf = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='{}_macro'.format(score) )
#
#        clf.fit(X_train, y_train)
#    
#        print("Best parameters set found on development set:")
#        print()
#        print(clf.best_params_)
#        print()
#        print("Grid scores on development set:")
#        print()
#        means = clf.cv_results_['mean_test_score']
#        stds = clf.cv_results_['std_test_score']
#        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#            print("%0.3f (+/-%0.03f) for %r"
#                  % (mean, std * 2, params))
#        print()
#    
#        print("Detailed classification report:")
#        print()
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print()
#        y_true, y_pred = y_test, clf.predict(X_test)
#        print(classification_report(y_true, y_pred))
#        print()


#    predictions = lr.predict_proba( encoder.transform(grd.apply(X_test)[:,:,0]) )[:,1]
#    fpr, tpr, _ = roc_curve(y_test, predictions)
#
#    fig, (ax1,ax2) = plt.subplots( ncols=2 )
#    ax1.plot(fpr,tpr,label='GBT + LR')
#    x = np.linspace(0,2*np.pi,400)
#    ax2.plot(x, np.sin(x**2))
#    plt.show()
    return 0


if __name__ == '__main__':
    import sys; sys.exit( _main() )
