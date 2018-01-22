"""
:mod:`ikaros.pipeline` implements pipeline stages with bayesian optimization
"""

import numpy as np
import sklearn.gaussian_process as gp

from itertools import product, chain
from tqdm import tqdm
from collections import OrderedDict

from stages import BaseStage
from sampling import discrete, continuous
from sklearn.base import ClassifierMixin
from scipy.stats import norm
from scipy.optimize import minimize

from imblearn.pipeline import make_pipeline

class BayesPipeline(ClassifierMixin):
    def __init__( self, stages, loss ):
        self.stages = stages
        self.loss_fn = loss
        self.stage_space, self.stage_space_bounds, self.stage_estimators, self.stage_loss = [], {}, [], []
        self.best_estimator_ = None

    def bootstrap( self, **fit_params ):
        bootstrap = fit_params.get('bootstrap',3)
        for _ in xrange(bootstrap):
            thisstage = []
            thisparams = OrderedDict()
            thisbounds = OrderedDict()
            for stage in self.stages:
                _estimator, _cparams, _bounds = stage.project()
                thisstage.extend(_estimator)
                thisparams.update(_cparams)
                thisbounds.update(_bounds)
            estimator = make_pipeline(*thisstage)
            self.stage_estimators.append(estimator)
            self.stage_loss.append(self.loss_fn(estimator))
            self.stage_space.append(thisparams)
            self.stage_space_bounds.update(thisbounds)

    def optimize( self, n_iterations, alpha=1e-5, epsilon=1e-7 ):
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

        xp = np.array( [ p.values() for p in self.stage_space ] )
        bounds = np.array(self.stage_space_bounds.values())
        yp = np.array( self.stage_loss )
        for n in range(n_iterations):
            model.fit(xp, yp)
            next_p = _next_sample(_expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)
            # remove near duplicates
            if np.any(np.abs(next_p - xp) <= epsilon):
                next_p = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

            thisstage = []
            thisparams = OrderedDict()
            thisbounds = OrderedDict()
            parameter_hints = { k:None for k in self.stage_space[0].keys() }
            for key, v in zip(parameter_hints.keys(),next_p):
                parameter_hints[key] = v

            for stage in self.stages:
                _estimator, _cparams, _bounds = stage.project(**parameter_hints)
                thisstage.extend(_estimator)
                thisparams.update(_cparams)
                thisbounds.update(_bounds)

            estimator = make_pipeline(*thisstage)
            self.stage_estimators.append(estimator)
            self.stage_loss.append(self.loss_fn(estimator))
            self.stage_space.append(thisparams)

            xp = np.array( [ p.values() for p in self.stage_space ] )
            yp = np.array(self.stage_loss)

        idx = np.argmax(self.stage_loss)
        self.best_estimator_ = self.stage_estimators[idx]

    def fit( self, X, y=None, **fit_params ):
        self.bootstrap(fit_params=fit_params)
        return self

    def transform( self, X ):
        if self.best_estimator_:
            return self.best_estimator_.transform(X)
        else:
            raise TypeError( 'need to call fit before transform' )

    def fit_transform( self, X, y=None ):
        self.fit(X,y)
        return self.transform(X)

class IPipeline(ClassifierMixin):
    def __init__( self, stages, loss, verbose=False ):
        self.stages = stages
        self.target_spaces = []
        self.loss_fn = loss
        self.verbose = verbose
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def fit( self, X, y=None ):
        stage_spaces = []
        for stage in self.stages:
            stage_spaces.append( self._product_stages(stage) )

        self.target_spaces = stage_spaces
        self.pipelines = [ BayesPipeline(space, self.loss_fn) for space
                           in product(*self.target_spaces) ]
        if self.verbose:
            print( 'bootstrapping pipelines...' )
            for p in tqdm(self.pipelines):
                p.fit(X,y)
        else:
            for p in self.pipelines:
                p.fit(X,y)

        self.best_score_ = 0.0
        for bp in self.pipelines:
            idx = np.argmax(bp.stage_loss)
            if bp.stage_loss[idx] > self.best_score_:
                self.best_score_ = bp.stage_loss[idx]
                self.best_estimator_ = bp.stage_estimators[idx]


    def optimize( self, n_iterations, depth=1, alpha=1e-5, epsilon=1e-7 ):
        op = lambda x: x
        if self.verbose:
            print( 'optimizing pipelines...' )
            op = tqdm
        for p in op(self.pipelines):
                p.optimize(n_iterations)

        index = np.argsort([np.max(bp.stage_loss) for bp in self.pipelines])[-depth:]
        op = lambda x: x
        if self.verbose:
            print( 'performing stage 2 optimization...' )
            op = tqdm
        for i in op(index):
            bp = self.pipelines[int(i)]
            bp.optimize(n_iterations*depth)


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
        target_spaces = [ stage.__class__(list(p)) for p in product(*psets) ]
        return target_spaces

def _expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement

def _next_sample(acquisition_func, 
                gaussian_process, 
                evaluated_loss, 
                greater_is_better=False,
                bounds=(0, 10), n_restarts=25):
    """ _next_sample
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x
