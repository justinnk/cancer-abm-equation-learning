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


import pandas as pd
import numpy as np
from kalmangrad import grad


class Dataset:
  def __init__(
    self,
    path: str,
    *,
    num_trajectories: int = 1,
    resample: int = 1,
    gradient_method: str = "central" # or: filtered
  ):
    """
    path: Path to the csv data file. First column must be time.
    num_trajectories: Number of trajectories stored in the file (one after the other).
    resample: Only take every "resample"-th time point into account.
    """
    self.path = path
    self.num_trajectories = num_trajectories
    # load measurement times and data points from csv to numpy arrays
    self._dataframe = pd.read_csv(path).iloc[::resample]
    _times = self._dataframe.time.to_numpy()
    _data = self._dataframe.drop("time", axis=1).to_numpy()
    # data may contain multiple trajectories -> reshape
    self._times = _times.reshape(self.num_trajectories, -1)
    self._data = _data.reshape(self.num_trajectories, self._times.shape[1], -1)
    if gradient_method not in ["central", "filtered"]:
      raise ValueError("Unknown gradient method '" + gradient_method + "'!")
    self._gradient_method = gradient_method

  def get_sindy_format(self):
    """Get the time and data points as a 1d array."""
    return self._data.reshape(-1, self._data.shape[-1]), self._times.reshape(-1)

  def get_numeric_gradients(self):
    if self._gradient_method == "central":
      gradient_func = self._central_gradient
    else:
      gradient_func = self._filtered_gradient
    grad = np.stack([
      gradient_func(idx)
      for idx in range(self.num_trajectories)
    ])
    grad = grad.reshape(-1, self._data.shape[-1])
    return grad

  @property
  def data(self):
    """The measured data as a numpy array of shape (num_traj, num_measurements, num_species)."""
    return self._data

  @property
  def times(self):
    """The measurement times as a numpy array of shape (num_traj, num_measurements)."""
    return self._times

  @property
  def ref_trajectory(self):
    # reference trajectory is the first in the set
    return self._dataframe.iloc[:self._data.shape[1]]

  @property
  def ref_gradient(self):
    # reference trajectory is the first in the set
    if self._gradient_method == "central":
      return self._central_gradient(0)
    else:
      return self._filtered_gradient(0)

  def _central_gradient(self, idx):
    return np.gradient(self._data[idx], self._times[idx], axis=0, edge_order=2)

  def _filtered_gradient(self, idx):
    derivative_rows = []
    for data_row, times_row in zip(self._data[idx].transpose(), self._times[idx]):
      #TODO: Note: assumes constant time step and equal time step over all trajectories
      smoother_states, filter_times = grad(data_row, self._times[0], delta_t=(self._times[idx][1]-self._times[idx][0]))
      derivative_rows.append([state.mean()[1] for state in smoother_states])
    return np.array(derivative_rows).transpose()

  def __repr__(self):
    return f"Dataset({self.path}, {self.num_trajectories})"



