"""
MIT License

Copyright (c) 2025 Justin Kreikemeyer, Hasitha N. Weerasinghe, Kevin Burrage, Pamela Burrage, Adelinde M. Uhrmacher

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

from lib.integrate import plot_sim_trace_of, plot_der_trace_of, gen_data_for, hyperparams_from_data
from lib.dataset import Dataset
from lib.reaction_library import library_for
from lib.coupled_sindy import optimize_coupled_sindy

import pickle
from itertools import product
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,extarrows}"
})

def plot(ax, model):
  ref_data = model.ref_dataset.ref_trajectory
  init_val, t_0, t_end, _ = hyperparams_from_data(ref_data)
  sim_data = gen_data_for(model, init_val, t_0, t_end, 100)
  sim_data = sim_data.rename({"I": "T-Cell (immune cell, I), EQL", "H": "Healthy Cell (H), EQL", "C": "Cancer Cell (C), EQL"}, axis=1)
  sim_data = sim_data.set_index("time")
  sim_data.plot(ax=ax, legend=False, cmap=ListedColormap(["blue","red","magenta"]))
  ax.set_prop_cycle(None)
  ref_data = ref_data.rename({"I": "T-Cell (immune cell, I), Data", "H": "Healthy Cell (H), Data", "C": "Cancer Cell (C), Data"}, axis=1)
  ref_data = ref_data.set_index("time")
  ref_data.plot(ax=ax, legend=False, marker="x", lw=0, markersize=4, cmap=ListedColormap(["blue","red","magenta"]), alpha=0.5)


if __name__ == "__main__":
  I0s = [0.2, 0.1, 0.05]                     # initial number of T-/immune-cells
  C_Is = [0.5, 0.75]                         # competition factor

  fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 10))

  union_accuracy_data = pd.read_csv("04_results/union_accuracy_data.csv")
  accuracy_measure = "freq_norm"
  union_accuracy_data = union_accuracy_data[union_accuracy_data["constrained"] == True]
  # the smaller the RMSE, the better -> take min()
  highest_accuracy = union_accuracy_data[union_accuracy_data[accuracy_measure] == union_accuracy_data[accuracy_measure].min()]
  print(f"{highest_accuracy=}")
  constrained, resamples, gradient_method = (
    highest_accuracy.constrained.values[0],
    highest_accuracy.resamples.values[0],
    highest_accuracy.gradient_method.values[0]
  )
  print(f"{constrained=}\n{resamples=}\n{gradient_method}\n{union_accuracy_data[accuracy_measure].min()=}")

  parameter_data = defaultdict(lambda: [])

  for idx, (C_I, I0) in enumerate(product(C_Is, I0s)):
    ax = axs[idx//3,idx%3]
    filename = f"02_data_processed/eql_{I0}_{C_I}.csv"
    ref_dataset = Dataset(filename, num_trajectories=1, gradient_method=gradient_method, resample=resamples)
    library = library_for(["C", "H", "I"], ref_dataset, 2, 2, verbose=True)
    with open(f"04_results/union_model_csindy_{constrained}_{resamples:02d}_{gradient_method}.pickle", "rb") as file:
      union_model = pickle.load(file)
    library.non_fixated_reactions = union_model.non_fixated_reactions
    method = "nnls"
    res, model = optimize_coupled_sindy(library, method, verbose=False, thresh=None, sparsify=False)
    for reac in model.reactions:
      parameter_data[reac.structure_str()].append(reac.rate)
    parameter_data["C_I"].append(C_I)
    parameter_data["I0"].append(I0)

    plot(ax, model)
    #print(f"{C_I=}, {I0=}")
    #print(model.to_ode_str())
    #print(model.to_ode_str(reduce=True, latex=True))
    model_str = model.to_latex(inline=True)
    ax.text(15.0, 0.37, f"${model_str}$", ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white"), fontsize=9)
    ax.set_ylabel("percent")
    ax.set_xlabel("time")
    if idx % 3 == 2:
      ax.text(22, 0.4, f"${C_I=}$")
    if idx // 3 == 0:
      ax.set_title("$\\tilde{I}(0)="+f"{I0}$")
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)
    ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    handles, labels = ax.get_legend_handles_labels()
  fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.06, left=0.08, top=0.9)
  # sort handles and labels to be aligned better in legend
  handles = handles[::3] + handles[1::3] + handles[2::3]
  labels = labels[::3] + labels[1::3] + labels[2::3]
  fig.legend(handles, labels, loc="upper center", ncols=3)
  fig.supxlabel("Initial percentage of immune (T-)cells $\\tilde{I}(0)$", y=0)
  fig.supylabel("Competition rate $C_I$", x=0.01)
  print(f"{constrained=}\n{resamples=}\n{gradient_method}")
  fig.savefig("06_union_fits.pdf", dpi=300)
  #plt.show()

  # print final union model
  for reac in model.reactions:
    reac.rate = 0.0
  print(model.to_latex())

