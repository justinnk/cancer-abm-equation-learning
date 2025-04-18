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

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from random import randint, shuffle, choice
from collections import Counter

import numpy as np

def power(val, ma):
  """Calculate val_i ** ma_i / ma_i for each element i of val."""
  p = np.power(val, ma)
  return np.prod(np.divide(p, ma, out=np.ones_like(p), where=ma!=0))

def full_hist(l: np.array, num: int):
  """Generate a histogram of the numbers in l over the domain 0 to num."""
  return np.histogram(l, num, (0, num))[0]

class Reaction:
  """Representation of a single reaction."""

  def __init__(self, reactands: List[int], products: List[int], rate: float, num_species: int, species_names=None, rate_bounds: tuple=(0, np.inf)):
    """
    reactands: list of reactands (left side)
    products:  list of products (right side)
    rate:      rate of the reaction
    """
    self._reactands = reactands
    self._products = products
    self.rate = rate
    self.num_species = num_species
    self.species_names = species_names
    self.rate_bounds = rate_bounds
    #self.species_to_labels = {}
    # cache internal state
    self._update_ma()
    self._update_change()

  @property
  def reactands(self):
    return self._reactands

  @reactands.setter
  def reactands(self, reactands):
    self._update_ma()
    self._update_change()
    self._reactands = reactands

  @property
  def products(self):
    return self._products

  @products.setter
  def products(self, products):
    self._update_change()
    self._products = products

  @property
  def change(self):
    return self._change

  def _update_ma(self):
    self._ma = full_hist(self._reactands, self.num_species)

  def _update_change(self):
    self._change = full_hist(self._products, self.num_species) - full_hist(self._reactands, self.num_species)

  def func(self, val):
    return power(val, self._ma) * self._change

  def apply_to(self, data):
    return np.apply_along_axis(self.func, 1, data)

  def get_deriv_terms(self, state):
    return np.apply_along_axis(self.func, 0, state)

  def structure_str(self) -> str:
    out = ""
    reactands =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._reactands).items()]
    out += " + ".join(reactands)
    out += " -> "
    products =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._products).items()]
    out += " + ".join(products)
    if self.species_names is not None:
      replacements = [(f"S{k}", v) for k, v in enumerate(self.species_names)]
      for k, v in reversed(replacements):
        out = out.replace(k, v)
    return out

  def __str__(self) -> str:
    out = ""
    reactands =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._reactands).items()]
    out += " + ".join(reactands)
    out += " -> "
    products =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._products).items()]
    out += " + ".join(products)
    #out += f" @@ {self.rate}[Hz] * "
    out += f" @ {self.rate:.2g}[Hz]"
    #out += f" @ {self.rate}[Hz]"
    #out += " * ".join([" * ".join([f"##S{idx}" for _ in range(ma)]) for idx, ma in enumerate(self._ma) if ma > 0])
    out += ";"#\t\t\t"
    #labels = "// labels: " + ", ".join(map(str, self.labels))
    if self.species_names is not None:
      replacements = [(f"S{k}", v) for k, v in enumerate(self.species_names)]
      for k, v in reversed(replacements):
        out = out.replace(k, v)
      #replacements = [(f"{k}", v) for k, v in enumerate(self.species_names)]
      #for k, v in reversed(replacements):
      #  labels = labels.replace(k, v)
    #out += labels
    return out

  def __repr__(self) -> str:
    return f"Reaction({self._reactands}, {self._products}, {self.rate}, {self.num_species})"

  def to_latex(self, gt_model=None) -> str:
    is_in_gt = False
    out = ""
    if gt_model is not None and any(self.equal_structure(r) for r in gt_model.reactions):
      is_in_gt = True
      out = r"\mathbf{"
    reactands =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._reactands).items()]
    out += " + ".join(reactands)
    if is_in_gt:
      out += "}"
    out += f"&\\xrightarrow{{{self.rate:.2g}}}&"
    if is_in_gt:
      out += "\mathbf{"
    products =  [f"{v}S{r}" if v > 1 else f"S{r}" if r > -1 else "" for r, v in Counter(self._products).items()]
    out += " + ".join(products)
    if is_in_gt:
      out += "}"
    #out += f" @ {self.rate:.2g}"
    if self.species_names is not None:
      replacements = [(f"S{k}", v.replace("_", "")) for k, v in enumerate(self.species_names)]
      for k, v in reversed(replacements):
        out = out.replace(k, v)
    return out

  def __hash__(self) -> int:
    return hash((frozenset(self._reactands), frozenset(self._products), self.rate))

  def __eq__(self, other: "Reaction") -> bool:
    """Two reactions are equal if they share reactands, products and rate, and have the same species max."""
    return (
      self.equal_structure(other) and
      self.rate == other.rate and 
      self.num_species == other.num_species
    )

  def equal_structure(self, other: "Reaction") -> bool:
    """Two reactions share an equal structure if they share reactands and products."""
    return (
      self._reactands == other._reactands and 
      self._products == other.products
    )
