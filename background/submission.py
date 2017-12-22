#!/usr/bin/env python
'''
submission.py - Spark Cognition data science challenge submission

There are quite a few lines in this script, but most of them are related to the initial exploratory
analysis I did before settling on a model.  I've tried to keep the most relevant pieces at the top of the file
for clarity.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from sparsesvd import sparsesvd

from tqdm import tqdm
from kernels import ClassificationKernel
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, validation_curve
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from imblearn import over_sampling, pipeline as pl

import lightgbm.sklearn as lgb
from catboost import CatBoostClassifier

from scipy import stats

RANDOM_STATE=42

def _main():
    '''
    _main() - build the best model and output the predictions on marketing_test.csv to 'submission.csv'
    '''
    df_train,df_test = load_data()
#    explore( df_train, df_test )
    X,y = df_train.drop('responded',axis=1), df_train['responded']
    encoder = LabelEncoder().fit(y)
    _,model = method7( X, y, encoder=encoder )
    targets = model.predict(df_test)
    columns = df_test.columns
    df_test = df_test.drop( columns, axis=1 )
    df_test['responded'] = encoder.inverse_transform(targets)
    df_test.to_csv( 'submission.csv' )
    return 0

def load_data():
    '''
    load_data:  load and tidy training and test data
    '''
    df_train, df_test = pd.read_csv( 'marketing_training.csv' ), pd.read_csv('marketing_test.csv', index_col=0)
    df_train = df_train.fillna(-999).sample(frac=1.0)
    df_test = df_test.fillna(-999)
    ignore = set([ 'responded' ])
    active_columns = set(df_train.columns).intersection(set(df_test.columns)).difference(ignore)
    print( 'label encoding categorical features...' )
    for column in tqdm(active_columns):
        if df_train[column].dtype == 'object':
            df_train[column] = df_train[column].apply(str)
            df_test[column] = df_test[column].apply(str)

            encoder = LabelEncoder()
            train_values = set(df_train[column].unique())
            test_values = set(df_test[column].unique())
            encoder.fit( list(train_values.union(test_values)) )
            df_train[column] = encoder.transform( df_train[column] )
            df_test[column] = encoder.transform( df_test[column] )

    return df_train, df_test


def explore( df_train, df_test ):
    '''
    explore:  do some exploratory analysis
    '''
    anova_test( df_train.drop('responded',axis=1), df_test )
    X,y = df_train.drop('responded',axis=1), df_train['responded']
    X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2)
    y_train = LabelEncoder().fit_transform(y_train)
    estimators = [ method1(X_train,y_train),
                   method2(X_train,y_train),
                   method3(X_train,y_train),
                   method4(X_train,y_train),
                   method7(X_train,y_train),
                   method8(X_train,y_train) ]
#    oversample_plots('oversampled.png', estimators, X, LabelEncoder().fit_transform(y))
    roc_curves('roc_curves.png', estimators, X_test, LabelEncoder().fit_transform(y_test))
    classification_reports(estimators,X_test,LabelEncoder().fit_transform(y_test))


def roc_curves( filename, estimators, X, y ):
    '''
    roc_curves: generate ROC curves for all estimators passed in on X and y
    '''
    figure = plt.figure()
    for name, estimator in estimators:
        y_prob = estimator.predict_proba(X)[:,1]
        roc_figure(figure, y_prob, y, label=name)

    plt.suptitle( 'ROC for some models' )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig( filename )

def roc_figure( figure, y_prob, y, label=None ):
    fpr, tpr, _ = metrics.roc_curve(y, y_prob)
    plt.plot(fpr,tpr,label=label)

def anova_test( dfl, dfr ):
    '''
    anova_test:  perform ANOVA on dfl and dfr and report any significantly differing features,
                 also generate a violin plot for inspecting the train/test distributions
    '''
    F, p = stats.f_oneway(dfl, dfr)
    df = pd.DataFrame(zip(dfr.columns,p), columns=['column','pvalue'])
    hetero = df[df['pvalue'] < 0.05]
    print( 'check on these columns: {}'.format(hetero) )
    for index, row in hetero.iterrows():
        column,pvalue = row['column'],row['pvalue']
        print( '{}: p={}'.format(column,pvalue) )
        print( '\tSet 1:' )
        print( '\t{}'.format(dfl[column].describe()) )
        print( '\tSet 2:' )
        print( '\t{}'.format(dfr[column].describe()) )

    columns = hetero['column'].values.tolist()
    d1 = dfl[columns].copy()
    d1['dataset'] = 'train'
    d2 = dfr[columns].copy()
    d2['dataset'] = 'test'
    plt.figure()
    g = sns.PairGrid( pd.concat((d1,d2)), 
                      x_vars=['dataset'],
                      y_vars=columns,
                      aspect=0.75, size=3.5 )
    g.map(sns.violinplot, palette='pastel')
    plt.savefig('violinplots.png')

def oversample_plots(filename, estimators, X, y):
    '''
    oversample_plots: How does oversampling affect performance?  
                      Plot Cohen's kappa as a function of oversampling on various estimators
    '''
    scorer = metrics.make_scorer(metrics.cohen_kappa_score)
    resampler = over_sampling.SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
#    resampler = over_sampling.RandomOverSampler(ratio='minority')
    for name, estimator in estimators:
        pipeline = pl.make_pipeline(resampler, StandardScaler(), estimator)
        
        param_range = range(5, 30)
#        param_range = [ 5, 10, 30 ]
        train_scores, test_scores = validation_curve(
                                       pipeline, X, y, param_name="smote__k_neighbors",
                                       param_range=param_range,
                                       scoring=scorer,
                                       cv=3)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        plt.plot(param_range, test_scores_mean, label='smote_k')
        ax.fill_between(param_range, test_scores_mean + test_scores_std,
                        test_scores_mean - test_scores_std, alpha=0.2)
        idx_max = np.argmax(test_scores_mean)
        plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
                    label=r'max: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))
        
        plt.title("Cohen's kappa vs smote k-neighbors ({})".format(name))
        plt.xlabel("k-neighbors")
        plt.ylabel("Cohen's kappa")
        
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
        plt.legend(loc="best")
        plt.savefig('{}_'.format(name)+filename)

def classification_reports(estimators,X,y):
    '''
    classification_reports: Generate confusion matrices, call sklearn.metrics.classification_reports for
                            all estimators on X and y
    '''
    for name, estimator in estimators:
        print( '-'*20 + name + '-'*20 )
        preds = estimator.predict(X)
        print( 'cohen kappa: {}'.format(metrics.cohen_kappa_score(preds, y)) )
        print( metrics.classification_report(preds, y) )

def method1( X, y ):
    '''
    method1: build a simple linear model pipeline
    '''
    pipeline = Pipeline( [ ('standardize', StandardScaler()),
                           ('linear', LogisticRegression()) ] )
    pipeline.fit(X,y)
    return ('LogisticRegression',pipeline)

def method2( X, y ):
    '''
    method2: build a simple random forest pipeline
    '''

    print('RandomForestClassifier')
    pipeline = Pipeline( [ ('standardize', StandardScaler()),
                           ('randomforestclassifier', RandomForestClassifier(n_estimators=350)) ] )
    pipeline.fit(X,y)
    return ('RandomForestClassifier',pipeline)

def method3( X, y ):
    '''
    method3: build a gradient boosting pipeline but weight samples by information entropy of class
    '''
    print('GradientBoostingClassifier (sample weighted)')
    from collections import Counter
    frequencies = Counter(y)
    y_weights = [ -np.log(frequencies[i]/float(len(y))) for i in y ]
    pipeline = Pipeline( [ ('standardize', StandardScaler()),
                           ('gradientboostingclassifier', GradientBoostingClassifier(n_estimators=350)) ] )
    params = { 'gradientboostingclassifier__sample_weight' : y_weights }
    pipeline.fit(X,y,**params)
    return ('GradientBoostingClassifier (sample weighted)', pipeline)

def method4( X, y ):
    '''
    method4: build default gradient boosting pipeline
    '''
    print('GradientBoostingClassifier')
    pipeline = Pipeline( [ ('standardize', StandardScaler()),
                           ('gradientboostingclassifier', GradientBoostingClassifier(n_estimators=350)) ] )
    pipeline.fit(X,y)
    return ('GradientBoostingClassifier',pipeline)

def method6(X, y):
    '''
    method6: grid search pipeline using Light-GBM and Catboost
    '''
    estimators = [ ('variance', VarianceThreshold()),
                   ('standardize', StandardScaler()),
#                   ('gboost', ClassificationKernel()),
                   ('lboost', lgb.LGBMClassifier()) ]
#                   ('catboost', CatBoostClassifier()),
#                   ('lr', LogisticRegression()) ]

    pipeline = Pipeline(estimators)

    preproc_search_params = { 'variance__threshold' : np.linspace( 0.95*(1-0.95), 0, 2 ) }
    gboost_search_params = {
        'gboost__n_estimators' : [50,100,150],
        'gboost__subsample' : [ 0.5, 1.0 ],
        'gboost__learning_rate' : [ 0.1, 0.03 ]
        }
    lboost_search_params = {
        'lboost__max_bin' : [ 400 ],
        'lboost__n_estimators' : [ 550 ],
        'lboost__colsample_bytree' : [ 0.5 ],
        'lboost__reg_lambda' : [ 0,1,10 ],
        'lboost__reg_alpha' : [ 0,1,10 ],
        'lboost__learning_rate' : [ 0.002 ],
        'lboost__num_leaves' : [ 130 ],
        'lboost__min_child_weight' : [ 0.06 ],
        'lboost__min_child_samples' : [4]
        }
    catboost_search_params = {
        'catboost__loss_function' : ['MAE'],
        'catboost__eval_metric' : [ 'MAE'],
        'catboost__iterations' : [ 500, 600, 630, 650 ],
        'catboost__bagging_temperature' : [ 1 ],
        'catboost__max_ctr_complexity' : [ 4, 6 ]
        }

    linear_search_params = { 
        'lr__C' : [0.001,0.01,0.1,1,10,100],
        'lr__penalty' : [ 'l1', 'l2' ]
        }
    search_params = {}
#    search_params.update(preproc_search_params)
    search_params.update(lboost_search_params)
#    search_params.update(gboost_search_params)
#    search_params.update(linear_search_params)
    search = GridSearchCV( pipeline, param_grid=search_params, verbose=10 )
    search.fit( X, y )
    print( 'best found estimator:' )
    print( search.best_estimator_ )
    print( 'best found params:' )
    print( search.best_params_ )
    print( 'best score:' )
    print( search.best_score_ )

def method7( X, y, encoder=None ):
    '''
    method7: build stack combining KNN, R-forest and LGBM with LogisticRegression meta-estimator
    '''
    num_neighbors = 1
    num_lgbm = 1
    num_forests = 1

    estimators = []
    estimators.extend( [ KNeighborsClassifier() for i in xrange(num_neighbors) ] )
    estimators.extend( [ lgb.LGBMClassifier() for i in xrange(num_lgbm) ] )
    estimators.extend( [ RandomForestClassifier() for i in xrange(num_forests) ] )

    print( 'building StackingCVClassifier' )
    model = StackingCVClassifier( classifiers=estimators,
                                  meta_classifier=LogisticRegression() )
    resampler = over_sampling.SMOTE(random_state=RANDOM_STATE, k_neighbors=20)
    X_u,y_u = resampler.fit_sample(X,y)
    if encoder:
        model.fit( X_u, encoder.fit_transform(y_u) )
    else:
        model.fit( X_u, y_u )
    return ('StackingCV', model)

def method8(X, y):
    '''
    method8: catboost averaging
    '''

    num_ensembles = 5
    ensemble = [ CatBoostClassifier( iterations=650, 
                                     learning_rate=np.random.choice([0.01,0.03]),
                                     depth=np.random.choice([4,5,6]), 
                                     l2_leaf_reg=3,
                                     loss_function='Logloss',
                                     eval_metric='Logloss',
                                     random_seed=np.random.randint(int(1e9))) for i in xrange(num_ensembles) ]

    estimators = [ ('catboost_{}'.format(i),f) for i,f in enumerate(ensemble) ]
    meta = VotingClassifier( estimators=estimators, voting='soft' ) # not recommended for uncalibrated probabilities
    meta.fit(X,y)
    return ('Catboost x5', meta)

def gridsearch(X,y):
    '''
    gridsearch: build a stacking classifier and gridsearch for optimal parameters
    '''
    num_neighbors = 1
    num_lgbm = 1
    num_forests = 1

    estimators = []
    estimators.extend( [ KNeighborsClassifier() for i in xrange(num_neighbors) ] )
    estimators.extend( [ lgb.LGBMClassifier() for i in xrange(num_lgbm) ] )
    estimators.extend( [ RandomForestClassifier() for i in xrange(num_forests) ] )

    parameters = {}
    for i in range(1,num_neighbors+1):
#    parameters['kneighborsclassifier-{}__n_neighbors'.format(i)] = scrandint(1,50)
#    parameters['kneighborsclassifier__n_neighbors'] = scrandint(1,50)
        parameters['kneighborsclassifier__n_neighbors'] = [5,10]

    for i in range(1,num_lgbm+1):
    #    parameters['lgbmclassifier-{}__n_estimators'.format(i)] = scrandint(10,6000)
    #    parameters['lgbmclassifier-{}__learning_rate'.format(i)] = [0.1,0.03,0.01,0.005]
    #    parameters['lgbmclassifier-{}__reg_alpha'.format(i)] = uniform(0,30)
    #    parameters['lgbmclassifier-{}__reg_lambda'.format(i)] = uniform(0,30)
    #    parameters['lgbmclassifier-{}__max_bin'.format(i)] = scrandint(10,10000)
    
    #    parameters['lgbmclassifier__n_estimators'] = scrandint(10,6000)
    #    parameters['lgbmclassifier__learning_rate'] = [0.1,0.03,0.01]
    #    parameters['lgbmclassifier__reg_alpha'] = uniform(0,30)
    #    parameters['lgbmclassifier__reg_lambda'] = uniform(0,30)
    #    parameters['lgbmclassifier__max_bin'] = scrandint(10,10000)
    
        parameters['lgbmclassifier__max_depth'] = [ -1]#,6 ]
        parameters['lgbmclassifier__num_leaves'] = [ 32,64,80 ]
        parameters['lgbmclassifier__min_child_samples'] = [ 10,100 ]
    
    #    parameters['lgbmclassifier__n_estimators'] = [ 500, 1000, 2000 ]
    #    parameters['lgbmclassifier__learning_rate'] = [0.03,0.01]
    
        parameters['lgbmclassifier__reg_alpha'] = [ 1.0]#, 10.0 ]
        parameters['lgbmclassifier__reg_lambda'] = [ 1.0]#, 10.0 ]
        parameters['lgbmclassifier__max_bin'] = [ 250, 500 ]

    for i in range(1,num_forests+1):
        parameters['randomforestclassifier__n_estimators'] = [ 350 ]

    parameters['meta-logisticregression__C'] = [ 0.01, 0.1, 1.0 ]

    n_search_iter=10
#    #model = RandomizedSearchCV(stack,param_distributions=parameters,n_iter=n_search_iter,verbose=10,refit=True,scoring='f1')
    model = GridSearchCV(stack, param_grid=parameters, verbose=10, scoring=metrics.make_scorer(metrics.cohen_kappa_score))
#
    X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2)
    X_res,y_res = over_sampling.RandomOverSampler(ratio='minority').fit_sample(X_train,y_train)
    X_res = StandardScaler().fit_transform(X_res)

    model.fit(X_res,LabelEncoder().fit_transform(y_res))

    print( 'best found estimator:' )
    print( model.best_estimator_ )
    print( 'best found params:' )
    print( model.best_params_ )
    print( 'best score:' )
    print( model.best_score_ )




if __name__ == '__main__':
    import sys
    sys.exit( _main() )
