"""
:mod:`ikaros.sampling` implements supported stage sampling strategies
"""
import numpy as np

class base_sampler(object):
    def __init__( self ):
        pass
    def sample( self, n=1 ):
        pass

class continuous(base_sampler):
    def __init__( self, low, high ):
        super(continuous, self).__init__()
        self.low = low
        self.high = high
    def sample( self ):
        return np.random.uniform(self.low,self.high)
    def bounds( self ):
        return (self.low,self.high)

class discrete(base_sampler):
    def __init__( self, levels ):
        super(discrete, self).__init__()
        self.levels = levels
    def sample( self ):
        np.random.choice(self.levels)
    
