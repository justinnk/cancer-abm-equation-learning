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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import pandas as pd

def plot_matrix(ax, coeff_matrix, reaction_names, parameter_config_names, path, sampling_interval, gradient_method):
  im = ax.imshow(coeff_matrix.transpose(), aspect=1.7, cmap="Greys")#, norm=matplotlib.colors.SymLogNorm(linthresh=1e-6)
  ax.set_xticks(range(len(reaction_names)))
  ax.set_xticklabels(reaction_names, rotation=90)
  ax.set_yticks(range(len(parameter_config_names)))
  ax.set_yticklabels(parameter_config_names)
  if gradient_method == "central":
    gradient_method = "unfiltered"
  ax.set_title("using every " + sampling_interval + "th datapoint; " + gradient_method + " gradients")
 #for i in range(coeff_matrix.shape[0]):
 #  for j in range(coeff_matrix.shape[1]):
 #    text = ax.text(i, j, f"{coeff_matrix[i, j]:.1}", ha="center", va="center", color="w", size=5)
  return im


def plot(restricted_library: bool):
  fig = plt.figure(figsize=(20, 10))
  axs = fig.subplots(nrows=4, ncols=3, sharey=True, sharex=True, gridspec_kw={"width_ratios": [1, 1, 0.05], "hspace": 0.0, "wspace": 0.05} )

  data_paths = [
    f"04_results/model_database_csindy_{restricted_library}_10_central.csv",
    f"04_results/model_database_csindy_{restricted_library}_20_central.csv",
    f"04_results/model_database_csindy_{restricted_library}_40_central.csv",
    f"04_results/model_database_csindy_{restricted_library}_50_central.csv",
  ]
  datasets = [pd.read_csv(d) for d in data_paths]
  for data, path, ax in zip(datasets, data_paths, axs[:,0]):
    coeff_matrix = data.iloc[:,1:].to_numpy()
    reaction_names = list(map(lambda x: "$"+x.replace('->', '\\rightarrow')+"$", data.iloc[:,0]))
    #parameter_config_names = list(map(lambda x: x.replace("_", " "), data.columns[1:]))
    parameter_config_names = list(map(lambda x: "$\\tilde{I}(0)="+x.replace("_", ", C_I=")+"$", data.columns[1:]))
    im = plot_matrix(ax, coeff_matrix, reaction_names, parameter_config_names, path, path[-14:-12], path[path.rfind("_")+1:-4])
    ax.set_ylabel("ABM params.")
  axs[3,0].set_xlabel("Reactions")

  data_paths = [
    f"04_results/model_database_csindy_{restricted_library}_10_filtered.csv",
    f"04_results/model_database_csindy_{restricted_library}_20_filtered.csv",
    f"04_results/model_database_csindy_{restricted_library}_40_filtered.csv",
    f"04_results/model_database_csindy_{restricted_library}_50_filtered.csv",
  ]
  datasets = [pd.read_csv(d) for d in data_paths]
  for data, path, ax in zip(datasets, data_paths, axs[:,1]):
    coeff_matrix = data.iloc[:,1:].to_numpy()
    reaction_names = list(map(lambda x: "$"+x.replace('->', '\\rightarrow')+"$", data.iloc[:,0]))
    parameter_config_names = list(map(lambda x: "$\\tilde{I}(0)="+x.replace("_", ", C_I=")+"$", data.columns[1:]))
    im = plot_matrix(ax, coeff_matrix, reaction_names, parameter_config_names, path, path[-15:-13], path[path.rfind("_")+1:-4])
  axs[3,1].set_xlabel("Reactions")
  
  gs = axs[0,0].get_gridspec()
  cax = fig.add_subplot(gs[:,2])
  cbar = fig.colorbar(im, cax=cax, location="right", shrink=0.2)
  cax.set_ylabel("coefficient value", rotation=-90, va="bottom")
  [axs[i,2].remove() for i in range(4)]

  fig.supxlabel("Gradient Method", y=0)
  fig.supylabel("Sampling Frequency", x=0.01)

  if restricted_library:
    fig.suptitle("Coefficient values derived with constrained library")
    fig.savefig("06_aggregated_results_restricted_lib.pdf")
  else:
    fig.suptitle("Coefficient values derived with full library")
    fig.savefig("06_aggregated_results_full_lib.pdf")
  #plt.show()

if __name__ == "__main__":
  plot(False)
  plot(True)
