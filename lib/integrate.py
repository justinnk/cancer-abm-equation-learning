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

from lib.reaction import Reaction
from lib.reaction_library import ReactionLibrary
from lib.dataset import Dataset

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from tqdm import tqdm


def gen_data_for(rs: ReactionLibrary, init_val: np.array, t_0=0.0, t_end=1.0, n_points=100, verbose=False):
  if len(rs.reactions) > 0 and len(rs.reactions[0].species_names) != 0:
    sim_data = pd.DataFrame(columns=["time"] + rs.reactions[0].species_names)
  else:
    sim_data = pd.DataFrame(columns=["time"] + list(map(lambda x: f"S{x}", range(0, rs.num_species))))
  solver = "LSODA"
  try:
    if n_points > 0:
      if verbose:
        print("Integrating from", t_0, "to", t_end, "with", solver, "measuring at", n_points, "points...")
        with tqdm(total=n_points, unit="%") as pbar:
          res = solve_ivp(rs.apply_odes, (t_0, t_end), init_val, solver, np.linspace(t_0, t_end, n_points), args=[pbar, [t_0, (t_end-t_0)/n_points]])#, rtol=1e-3, atol=1e-6)
      else:
        res = solve_ivp(rs.apply_odes, (t_0, t_end), init_val, solver, np.linspace(t_0, t_end, n_points))#, rtol=1e-3, atol=1e-6)
    else:
      res = solve_ivp(rs.apply_odes, (t_0, t_end), init_val, solver, None)#, rtol=1e-3, atol=1e-6)
  except:
    print("Exception during integration!")
  if not res.success:
    print("ERROR")
    print(res)
    exit(-1)
  sim_data.time = res.t
  for idx, col in enumerate(sim_data.columns[1:]):
    sim_data[col] = res.y[idx]
  return sim_data

def hyperparams_from_data(data: pd.DataFrame):
  init_val = data[[col for col in data.columns if col != "time"]].iloc[0]
  t_0 = data.time.iloc[0]
  t_end = data.time.iloc[-1]
  npoints = len(data.time) - 1
  return init_val, t_0, t_end, npoints

def plot_sim_trace_of(rs: ReactionLibrary, n_points=100, t_end=None, ax=None):
  ref_data = rs.ref_dataset.ref_trajectory
  init_val, t_0, _t_end, _ = hyperparams_from_data(ref_data)
  if t_end is None:
    t_end = _t_end
  sim_data = gen_data_for(rs, init_val, t_0, t_end, n_points)
  if ax is None:
    fig, ax = plt.subplots()
  ref_data.add_prefix("ref_").plot(ax=ax, x="ref_time", marker="o", lw=0)
  #plt.gca().set_prop_cycle(None)
  ax.set_prop_cycle(None)
  sim_data.add_prefix("sim_").plot(ax=ax, x="sim_time", ls="--")
  ax.set_xlabel("time")
  ax.set_ylabel("amount")
  #plt.show()

def plot_system_comparison(rs1: ReactionLibrary, rs2: ReactionLibrary, ref_dataset: Dataset, n_points=100, rs1_label="sim1", rs2_label="sim2"):
  init_val_1, t_0_1, t_end_1, _ = hyperparams_from_data(ref_dataset.ref_trajectory)
  init_val_2, t_0_2, t_end_2, _ = hyperparams_from_data(ref_dataset.ref_trajectory)
  if rs1.ref_dataset != None:
    init_val_1, t_0_1, t_end_1, _ = hyperparams_from_data(rs1.ref_dataset.ref_trajectory)
  if rs2.ref_dataset != None:
    init_val_2, t_0_2, t_end_2, _ = hyperparams_from_data(rs2.ref_dataset.ref_trajectory)
  ref_data = ref_dataset.ref_trajectory
  sim1_data = gen_data_for(rs1, init_val_1, t_0_1, t_end_1, n_points)
  sim2_data = gen_data_for(rs2, init_val_2, t_0_2, t_end_2, n_points)
  fig, ax = plt.subplots()
  ref_data.add_prefix("ref_").plot(ax=ax, x="ref_time", marker="o", lw=0)
  plt.gca().set_prop_cycle(None)
  sim1_data.add_prefix(rs1_label+"_").plot(ax=ax, x=f"{rs1_label}_time", ls="--")
  plt.gca().set_prop_cycle(None)
  sim2_data.add_prefix(rs2_label+"_").plot(ax=ax, x=f"{rs2_label}_time", ls="-")
  return fig, ax

def plot_der_trace_of(rs: ReactionLibrary, n_points=100, t_end=None):
  data, _ = rs.ref_dataset.get_sindy_format()
  ref_data = rs.ref_dataset.ref_trajectory
  columns_except_time = np.setdiff1d(ref_data.columns, ["time"])
  ref_data[columns_except_time] = rs.ref_dataset.ref_gradient
  sim_deriv = np.sum([r.apply_to(rs.ref_dataset._data[0]) * r.rate for r in rs.reactions], axis=0)
  #sim_deriv = [r.apply_to(rs.ref_dataset._data[0][45:70]) * r.rate for r in rs.reactions]
  #sim_deriv = np.sum(sim_deriv[::2], axis=0)
  #plt.plot(sim_deriv)
  #plt.show()
  sim_data = ref_data.copy()
  sim_data[columns_except_time] = sim_deriv
  fig, ax = plt.subplots()
  ref_data.add_prefix("ref_").plot(ax=ax, x="ref_time", marker="o", lw=0)
  plt.gca().set_prop_cycle(None)
  sim_data.add_prefix("sim_").plot(ax=ax, x="sim_time", ls="--")
  plt.xlabel("time")
  plt.ylabel(r"$\frac{d amount}{d t}$")
  #plt.show()

