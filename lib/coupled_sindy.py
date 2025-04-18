"""
MIT License

Copyright (c) 2025 Justin Kreikemeyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Based on 
Burrage, Pamela M., Hasitha N. Weerasinghe, and Kevin Burrage. "Using a library of chemical reactions to fit systems of ordinary differential equations to agent-based models: a machine learning approach." Numerical Algorithms 96.3 (2024): 1063-1077.
"""

from lib.reaction import Reaction
from lib.reaction_library import ReactionLibrary, get_model_intersection
from lib.dataset import Dataset

from typing import List, Tuple
from copy import deepcopy
from collections import namedtuple

from scipy.optimize import least_squares, nnls, lsq_linear
from sklearn.linear_model import Lasso, LinearRegression, LassoCV, LinearRegression, Ridge, ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pysindy.optimizers.stlsq import STLSQ
#from pysindy.optimizers.base import EnsembleOptimizer
from tqdm import tqdm

class Res:
  def __init__(self, coef: np.array, data_deriv: np.array, design_deriv: np.array, fixed_coef: np.array = []):
    self.coef = coef
    self.fixed_coef = fixed_coef
    self.r2_score = r2_score(data_deriv, design_deriv @ coef)
    self.norm = np.linalg.norm(data_deriv - design_deriv @ coef)

  def __repr__(self):
    return f"Res({self.coef}, {self.r2_score}, {self.norm})"


class CoupledSINDy:
  def __init__(self, library: ReactionLibrary, verbose: bool = False, consider_rate_bounds: bool = False):
    data, times = library.ref_dataset.get_sindy_format()
    self.data = data
    self.data_deriv = np.transpose(library.ref_dataset.get_numeric_gradients()).reshape(-1)
    self.times = times
    self.library = library
    self.verbose = verbose
    self.consider_rate_bounds = consider_rate_bounds
    if self.verbose:
      self.log("length of library:", len(self.library.reactions), "reactions")

  def log(self, *msg):
    if self.verbose: print("[CoupledSINDy]", *msg)

  def loss(self, coeffs, theta):
    losses = []
    for coeff in coeffs:
      losses.append(np.linalg.norm(self.data_deriv - theta @ coeff))
    return losses

  def optimize(self, method="nnls"):
    """
    method: The method to use in [nnls, stlsq, lasso,  ensemble]
    Returns: Res(rate constants, r2 score, norm)
    """
    self.log("applying function.")
    progress_bar = tqdm if self.verbose else lambda x: x # only show progress bar if verbose
    theta = np.array([])
    if self.library.fixated_reactions is not None and len(self.library.fixated_reactions) > 0:
      theta_fix = np.transpose(np.stack([r.apply_to(self.data) for r in progress_bar(self.library.fixated_reactions)]))
      theta_fix = theta_fix.reshape(-1, theta_fix.shape[-1])
      coeff_fix = np.array([r.rate for r in self.library.fixated_reactions])
      theta = np.transpose(np.stack([r.apply_to(self.data) for r in progress_bar(self.library.non_fixated_reactions)]))
      self.data_deriv = self.data_deriv - theta_fix @ coeff_fix
    else:
      theta = np.transpose(np.stack([r.apply_to(self.data) for r in progress_bar(self.library.reactions)])) # library of reactions
    theta = theta.reshape(-1, theta.shape[-1])
    # bounds on the rate constants (default 0, inf)
    bounds = (0, np.inf)
    if self.consider_rate_bounds:
      method = "constrained_lsq"
      lb = np.zeros(len(self.library.non_fixated_reactions))
      ub = np.ones(len(self.library.non_fixated_reactions)) * float("inf")
      for idx, reac in enumerate(self.library.non_fixated_reactions):
        lb[idx] = reac.rate_bounds[0]
        ub[idx] = reac.rate_bounds[1]
      bounds = (lb, ub)
      self.log("rate constants are bounded to:", bounds)
    self.log("starting regression.")
    self.log("shape of design matrix (theta):", theta.shape)
    self.log("shape of data matrix:", self.data_deriv.shape)
    if method == "nnls":
      res = nnls(theta, self.data_deriv, maxiter=1e9)#, atol=1e-20)
      res = Res(res[0], self.data_deriv, theta)
    elif method == "pso":
      import pyswarms as ps
      options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
      min_bound = np.zeros(len(self.library.non_fixated_reactions))
      max_bound = np.ones_like(min_bound) * 3.0
      optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(self.library.non_fixated_reactions), options=options, bounds=(min_bound, max_bound))
      cost, pos = optimizer.optimize(self.loss, iters=5000, n_processes=12, theta=theta)
      print(cost, pos)
      res = Res(pos, self.data_deriv, theta)
    elif method == "stlsq":
      res = STLSQ(threshold=1e-4, max_iter=1000).fit(theta, self.data_deriv)
      res = Res(res.coef_[0], self.data_deriv, theta)
    elif method == "lasso":
      #lasso = Lasso(alpha=10.3, max_iter=1000000, tol=1e-12, fit_intercept=False)
      #lasso = LassoCV(n_alphas=10000, max_iter=1000000, tol=1e-12, fit_intercept=False, positive=True)
      lasso = ElasticNetCV(l1_ratio=0.75, n_alphas=10000, fit_intercept=False, positive=True)
      res = lasso.fit(theta, self.data_deriv)
      res = Res(res.coef_, self.data_deriv, theta)
    elif method == "constrained_lsq":
      res = lsq_linear(theta, self.data_deriv, bounds=bounds)
      res = Res(res.x, self.data_deriv, theta)
    elif method == "lib_ensemble":
      res = EnsembleOptimizer(WrappedOptimizer(LinearRegression(positive=True, fit_intercept=False)), library_ensemble=True, n_models=100, n_candidates_to_drop=2).fit(theta, self.data_deriv)
      res = Res(res.coef_[0], self.data_deriv, theta)
    elif method == "time_ensemble":
      res = EnsembleOptimizer(WrappedOptimizer(LinearRegression(positive=True, fit_intercept=False)), bagging=True, n_models=100, n_subset=10).fit(theta, self.data_deriv)
      res = Res(res.coef_[0], self.data_deriv, theta)
    else:
      raise Exception(f"Method {method} not supported. Choose one of nnls, lasso, least_squares or lsq_linear!")
    self.log("done.\n", res)
    return res

  def get_sparse_reaction_library(self, res: Res, thresh=None, sparsify=True):
    """Get the model determined via regression."""
    library = deepcopy(self.library)
    reactions = []
    if thresh is not None:
      res.coef[res.coef < thresh] = 0.0
    active_reaction_idxs = list(range(len(res.coef)))
    if sparsify:
      active_reaction_idxs = [idx for idx, x in enumerate(res.coef) if x != 0.0]
    for active_reaction_idx in active_reaction_idxs:
      library.non_fixated_reactions[active_reaction_idx].rate = res.coef[active_reaction_idx]
      reactions += [library.non_fixated_reactions[active_reaction_idx]]
    library.non_fixated_reactions = reactions + library.fixated_reactions
    library.fixated_reactions = []
    return library


def optimize_coupled_sindy(library, method="nnls", thresh=None, sparsify=True, verbose=False):
  """Convenience function to apply coupled sindy to library and return resulting model."""
  csindy = CoupledSINDy(library, verbose=verbose)
  res = csindy.optimize(method=method)
  return res, csindy.get_sparse_reaction_library(res, thresh=thresh, sparsify=sparsify)

