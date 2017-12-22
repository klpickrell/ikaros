"""
:mod:`ikaros.stages` implements supported pipeline stages
"""

class BaseStage(object):
    def __init__( self ):
        self.steps = []
    
    def add( self, classtype, **params ):
        if 'params' in params:
            self.steps.append( (classtype,params['params']) )
        else:
            self.steps.append( (classtype,params) )

    def __iter__(self):
        return iter(self.steps)


class SequentialStage(BaseStage):
    def __init__( self ):
        super(SequentialStage, self).__init__()

class StackedStage(BaseStage):
    def __init__( self ):
        super(StackedStage, self).__init__()
    
