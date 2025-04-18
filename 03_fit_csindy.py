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

from lib.coupled_sindy import CoupledSINDy, optimize_coupled_sindy
from lib.dataset import Dataset
from lib.integrate import plot_sim_trace_of, plot_der_trace_of, gen_data_for
from lib.reaction_library import get_model_intersection, get_model_union, library_for

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from itertools import product
from copy import deepcopy
from collections import defaultdict
import pickle


if __name__ == "__main__":

  save_results = True

  ## determine a model for each time-series and calculate intersection

  restrict_interactions_s = [False, True]     # let immune cells only interact with cancer cells
  resamples = [10, 20, 40, 50]               # only consider every n-th time point
  gradient_methods = ["central", "filtered"] # method used to determine numerical gradients
  I0s = [0.2, 0.1, 0.05]                     # initial number of T-/imunte-cells
  C_Is = [0.5, 0.75]                         # competition factor

  fine_accuracy_data = dict(
    constrained=[],
    resamples=[],
    gradient_method=[],
    I0=[],
    C_I=[],
    lib_length=[],
    r2_score=[],
    norm=[]
  )
  union_accuracy_data = dict(
    constrained=[],
    resamples=[],
    gradient_method=[],
    model_length=[],
    r2_score=[],
    freq_norm=[],
    phase_norm=[]
  )

  for restrict_interactions, resample, gradient_method  in product(restrict_interactions_s, resamples, gradient_methods):
    models = []
    model_database = defaultdict(lambda: [])
    for I0, C_I, in product(I0s, C_Is):
      filename = f"02_data_processed/eql_{I0}_{C_I}.csv"
      ref_dataset = Dataset(filename, num_trajectories=1, resample=resample, gradient_method=gradient_method)
      library = library_for(["C", "H", "I"], ref_dataset, 2, 2, verbose=True)
      def filter_interactions(reac):
        # 0 C, 1 H, 2 I
        if (1 in reac.reactands) and (2 in reac.reactands): # H + I
          return False
        if reac.reactands == [2, 2]: # I + I
          return False
        return True
      if restrict_interactions:
        library.non_fixated_reactions = list(filter(filter_interactions, library.reactions))
      model_database["reaction"] = list(map(lambda x: x.structure_str(), library.reactions))
      res, model = optimize_coupled_sindy(library, method="nnls", verbose=False, thresh=None)
      print("coeff. of determination:", res.r2_score)
      print("model learned for", filename)
      print(model)
      fine_accuracy_data["constrained"].append(restrict_interactions)
      fine_accuracy_data["resamples"].append(resample)
      fine_accuracy_data["gradient_method"].append(gradient_method)
      fine_accuracy_data["I0"].append(I0)
      fine_accuracy_data["C_I"].append(C_I)
      fine_accuracy_data["lib_length"].append(len(library.non_fixated_reactions))
      fine_accuracy_data["r2_score"].append(res.r2_score)
      fine_accuracy_data["norm"].append(res.norm)

      models.append(model)
      for reaction in library.reactions:
        for model_reaction in model.reactions:
          if model_reaction.equal_structure(reaction):
            model_database[f"{I0}_{C_I}"].append(model_reaction.rate)
            break
        else:
          model_database[f"{I0}_{C_I}"].append(0.0)
    if save_results:
      model_database_frame = pd.DataFrame(model_database)
      model_database_frame.to_csv(f"04_results/model_database_csindy_{restrict_interactions}_{resample:02d}_{gradient_method}.csv", index=False)

    intersection_model = models[0]
    for idx in range(1, len(models)):
      intersection_model = get_model_intersection(intersection_model, models[idx])
    print("Intersection")
    print(intersection_model)
    if save_results:
      with open(f"04_results/intersection_model_csindy_{restrict_interactions}_{resample:02d}_{gradient_method}.pickle", "wb+") as file:
        pickle.dump(intersection_model, file)

    union_model = models[0]
    for idx in range(1, len(models)):
      union_model = get_model_union(union_model, models[idx])
    print("Union")
    print(union_model)
    if save_results:
      with open(f"04_results/union_model_csindy_{restrict_interactions}_{resample:02d}_{gradient_method}.pickle", "wb+") as file:
        pickle.dump(union_model, file)

    ### try to fit the union model to the different time-series
    print("Fitting union model to different parametrizations...")

    mean_r2 = 0
    mean_phase_norm = 0
    mean_freq_norm = 0
    count = 0
    for I0, C_I in product(I0s, C_Is):

      filename = f"02_data_processed/eql_{I0}_{C_I}.csv"
      ref_dataset = Dataset(filename, num_trajectories=1, gradient_method=gradient_method, resample=resample)
      library = library_for(["C", "H", "I"], ref_dataset, 2, 2, verbose=True)
      library.non_fixated_reactions = union_model.non_fixated_reactions

      method = "nnls"
      res, model = optimize_coupled_sindy(library, method, verbose=False, thresh=None)
      print("coeff. of determination:", res.r2_score)
      print("model learned for", filename)
      print(model)
      mean_r2 += res.r2_score
      mean_phase_norm += res.norm
      data = gen_data_for(model, ref_dataset.ref_trajectory.values[0][1:], t_0=0, t_end=19.99, n_points=len(ref_dataset.ref_trajectory))
      mean_freq_norm = np.sqrt(mean_squared_error(ref_dataset.ref_trajectory.set_index("time"), data.set_index("time")))
      count += 1
    union_accuracy_data["constrained"].append(restrict_interactions)
    union_accuracy_data["resamples"].append(resample)
    union_accuracy_data["gradient_method"].append(gradient_method)
    union_accuracy_data["model_length"].append(len(library.non_fixated_reactions))
    union_accuracy_data["r2_score"].append(mean_r2 / count)
    union_accuracy_data["phase_norm"].append(mean_phase_norm / count)
    union_accuracy_data["freq_norm"].append(mean_freq_norm / count)
  if save_results:
    pd.DataFrame(fine_accuracy_data).to_csv(f"04_results/fine_accuracy_data.csv", index=False)
  if save_results:
    pd.DataFrame(union_accuracy_data).to_csv(f"04_results/union_accuracy_data.csv", index=False)

