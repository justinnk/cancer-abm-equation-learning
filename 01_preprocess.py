#!/usr/bin/env python3

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

import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from collections import defaultdict


if __name__ == "__main__":
  I0s = [0.2, 0.1, 0.05] # initial number of T-cells (imune cells)
  C_Is = [0.5, 0.75] # competition factor
  time_step = 0.01
  all_data = []
  for I0, C_I in product(I0s, C_Is):
    data = pd.read_csv(f"01_data/eql_{I0}_{C_I}.csv")
    # (possibly) subselect a range from the 2000 original samples
    data = data.iloc[:2000]
    # rename columns to make life easier
    # Cancer, Healthy, Immune (T)
    data = data.rename({"X": "C", "Y": "H", "I": "I"}, axis=1)
    # add proper time
    old_columns = data.columns
    data["time"] = data.index * time_step
    # move time column to front
    data = data[["time", *old_columns]]
    data.to_csv(f"02_data_processed/eql_{I0}_{C_I}.csv", index=False)
    all_data.append(data)
  all_data = pd.concat(all_data)
  all_data.to_csv(f"02_data_processed/eql_all.csv", index=False)
