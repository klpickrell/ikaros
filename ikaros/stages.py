"""
:mod:`ikaros.stages` implements supported pipeline stages
"""

from copy import deepcopy
from mlxtend.classifier import StackingClassifier

from sampling import discrete, continuous

class BaseStage(object):
    def __init__( self, steps=None ):
        if steps:
            self.steps = steps
        else:
            self.steps = []
    
    def add( self, classtype, **params ):
        if 'params' in params:
            self.steps.append( (classtype,params['params']) )
        else:
            self.steps.append( (classtype,params) )

    def project( self, **parameter_hints ):
        projected_steps = []
        projected_cparams = {}
        cparams_bounds = {}
        for classtype, params in deepcopy(self.steps):
            new_params = {}
            for key, value in params.iteritems():
                _param_name = '{}__{}'.format(classtype.__name__,key)
                if _param_name in parameter_hints:
                    new_params[key] = parameter_hints[_param_name]
                    if type(value) is continuous:
                        projected_cparams[_param_name] = parameter_hints[_param_name]
                        cparams_bounds[_param_name] = value.bounds()
                    continue
                else:
                    v = value
                    if type(value) in (discrete,continuous):
                        v = value.sample()
                    new_params[key] = v
                    if type(value) is continuous:
                        projected_cparams[_param_name] = v
                        cparams_bounds[_param_name] = value.bounds()
            projected_steps.append( classtype(**new_params) )
        return projected_steps, projected_cparams, cparams_bounds
                    

    def __iter__(self):
        return iter(self.steps)


class SequentialStage(BaseStage):
    def __init__( self, steps=None ):
        super(SequentialStage, self).__init__(steps=steps)

class StackedStage(BaseStage):
    def __init__( self, steps=None ):
        super(StackedStage, self).__init__(steps=steps)
    def project( self, **parameter_hints ):
        projected_steps, projected_cparams, cparams_bounds = super(StackedStage,self).project(**parameter_hints)
        return [ StackingClassifier( classifiers=projected_steps[:-1],
                                     meta_classifier=projected_steps[-1] ) ], projected_cparams, cparams_bounds

