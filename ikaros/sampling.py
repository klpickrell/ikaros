"""
:mod:`ikaros.sampling` implements supported stage sampling strategies
"""
import numpy as np

class base_sampler(object):
    def __init__( self ):
        pass

class continuous(base_sampler):
    def __init__( self, low, high ):
        super(continuous, self).__init__()
        self.low = low
        self.high = high

class discrete(base_sampler):
    def __init__( self, levels ):
        super(discrete, self).__init__()
        self.levels = levels
    
