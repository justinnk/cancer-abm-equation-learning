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
from matplotlib.colors import ListedColormap
import pandas as pd

from itertools import product

# blue: healthy
# red: cancer
# purple: immune


def frequency_space_plot(ax, data):
  ldata = data.rename({"I": "T-Cell (immune cell, I)", "H": "Healthy Cell (H)", "C": "Cancer Cell (C)"}, axis=1)
  ldata = ldata.set_index("time")
  ldata.plot(ax=ax, legend=False, cmap=ListedColormap(["blue","red","magenta"]))


def phase_space_plot(all_data):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  
  # Plot the phase portrait
  ax.scatter(all_data["I"], all_data["H"], all_data["C"], label="Cancer ABM")
  ax.set_xlabel("I")
  ax.set_ylabel("H")
  ax.set_zlabel("C")
  ax.set_title("Phase Portrait")
  ax.grid(True)
  ax.legend()
  ax.view_init(31, 10, 0)
  fig.savefig("02_analysis_results/phase_portrait.pdf")

  
if __name__ == "__main__":
  I0s = [0.2, 0.1, 0.05] # initial number of T-cells (imune cells)
  C_Is = [0.5, 0.75] # competition factor
  all_data = []
  fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8, 5))
  for idx, (C_I, I0) in enumerate(product(C_Is, I0s)):
    data = pd.read_csv(f"02_data_processed/eql_{I0}_{C_I}.csv")
    ax = axs[idx//3,idx%3]
    frequency_space_plot(ax, data)
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
    all_data.append(data)
  all_data = pd.concat(all_data)
  #fig.tight_layout()
  fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.15)
  fig.legend(handles, labels, loc="upper center", ncols=3)
  fig.supxlabel("Initial percentage of immune (T-)cells $\\tilde{I}(0)$", y=0)
  fig.supylabel("Competition rate $C_I$", x=0.01)
  fig.savefig("02_analysis_results/timeseries.pdf")
  phase_space_plot(all_data)
