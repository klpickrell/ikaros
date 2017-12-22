"""
:mod:`ikaros.pipeline` implements pipeline stages with bayesian optimization
"""

import numpy as np
import sklearn.gaussian_process as gp

from itertools import product

from stages import SequentialStage, StackedStage
from sampling import discrete, continuous
from sklearn.base import ClassifierMixin

class IPipeline(ClassifierMixin):
    def __init__( self, stages, loss ):
        self.stages = stages
        self.target_spaces = []
        self.loss_fn = loss
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def fit( self, X, y=None ):
        stage_spaces = []
        for stage in self.stages:
            stage_spaces.append( self._product_stages(stage) )

        self.target_spaces.extend( product( *stage_spaces ) )


    def transform( self, X ):
        if self.best_estimator_:
            return self.best_estimator_.transform(X)
        else:
            raise TypeError( 'need to call fit before transform' )

    def fit_transform( self, X, y=None ):
        self.fit(X,y)
        return self.transform(X)

    def _product_stages( self, stage ):
        psets = []
        for operation, params in stage:
            if not params:
                psets.append( [ (operation,params) ] )
                continue

            allparams = []
            for name, param in params.iteritems():
                if type(param) is discrete:
                    allparams.append( [ (name,p) for p in param.levels ] )
                else:
                    allparams.append( [ (name,param) ] )
            allops = []
            for pset in product(*allparams):
                allops.append( (operation,dict(pset)) )
            psets.append(allops)
        target_spaces = [ p for p in product(*psets) ]
        return target_spaces

