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

from lib.reaction import Reaction, full_hist
from lib.reaction_enumerator import ReactionEnumerator
from lib.dataset import Dataset

from random import randint, choice
from typing import List, Union
from copy import deepcopy
from math import gcd
from collections import defaultdict
from ast import literal_eval

import numpy as np
import pandas as pd

def library_for(
    species_names: list[str],
    ref_dataset: Dataset,
    max_num_left: int,
    max_num_right: int,
    *,
    subgroups: list = [],
    shuffle: bool=False,
    avoid_colinear: bool=False,
    verbose: bool=False
  ) -> "ReactionLibrary":
  """Create a complete reaction library for the given number of species."""
  reac_enum = ReactionEnumerator(
    num_species=len(species_names),
    max_num_left=max_num_left,
    max_num_right=max_num_right,
    max_stoichiometry=1,
    constraints=[],
    subgroups=subgroups,
    shuffle=shuffle,
    species_names=species_names
  )
  reactions = list(reac_enum.generator())
  if verbose: print("[library_for]", "Length of library:", len(reactions), "(filtered by stoichiometry, constraints), ", reac_enum.get_number(), "(all possible)")
  library = ReactionLibrary(
    reactions,
    len(species_names),
    ref_dataset=ref_dataset,
    species_names=reac_enum.species_names,
    fixated_reactions=[]
  )
  if verbose: print("[library_for]", "Final length of library:", len(library.reactions), "(", len(library.fixated_reactions), "of them fixated)")
  return library

def species_from_dataframe(df: pd.DataFrame):
  """Return (number of species, list of names) from the given dataframe."""
  species_names = list(df.columns[1:])
  return len(species_names), species_names

def get_model_intersection(model1, model2):
  """Return a ReactionLibrary with all common reactions of model1 and model2, disregarding rates and fixated reactions."""
  inter_model = deepcopy(model1)
  inter_model.fixated_reactions = []
  inter_model.non_fixated_reactions = []
  for reac1 in model1.non_fixated_reactions:
    for reac2 in model2.reactions:
      if reac1.equal_structure(reac2):
        inter_model.non_fixated_reactions.append(deepcopy(reac1))
  return inter_model


def get_model_union(model1, model2):
  """Return a ReactionLibrary with all common reactions of model1 and model2, disregarding rates and fixated reactions."""
  union_model = deepcopy(model1)
  union_model.fixated_reactions = []
  union_model.non_fixated_reactions = []
  for reac2 in model2.non_fixated_reactions:
    union_model.non_fixated_reactions.append(deepcopy(reac2))
  union_model.clean_duplicate_reactions()
  return union_model


class ReactionLibrary:
  """A library of biochemical reactions, some of which can be marked as immutable."""
  def __init__(self, non_fixated_reactions: List[Reaction], num_species, *, ref_dataset: Dataset = None, species_names=[], enumerator=None, fixated_reactions=[]): 
    self.fixated_reactions = fixated_reactions
    self.non_fixated_reactions = non_fixated_reactions
    self.ref_dataset = ref_dataset
    self.enumerator = enumerator
    self.num_species = num_species
    self.species_names = species_names
    for reaction in self.reactions:
      reaction.species_names = species_names
    self.clean_duplicate_reactions()

  @classmethod
  def from_ref_data(cls, reactions: List[Reaction], ref_dataset: Dataset, enumerator=None):
    if isinstance(ref_dataset, Dataset) and ref_dataset is not None:
      num_species, species_names = species_from_dataframe(ref_dataset.ref_trajectory)
      return cls(reactions, num_species, ref_dataset=ref_dataset, species_names=species_names, enumerator=enumerator)
    else:
      print("Error: ref_dataset should be a non-empty string!")
      exit(-1)

  @property
  def reactions(self):
    """All reactions within the library (fixated or not)."""
    if self.fixated_reactions is None:
      return self.non_fixated_reactions
    return self.fixated_reactions + self.non_fixated_reactions

  def __str__(self) -> str:
    out = ""
    if self.fixated_reactions is not None:
      for reaction in self.fixated_reactions:
        out += str(reaction) + " (fix)\n"
    for reaction in self.non_fixated_reactions:
      out += str(reaction) + "\n"
    return out

  def __repr__(self) -> str:
    return f"ReactionLibrary({self.non_fixated_reactions}, {self.num_species}, ref_dataset={self.ref_dataset}, species_names={self.species_names}, fixated_reactions={self.fixated_reactions})"

  def to_latex(self, gt_model=None, inline=False) -> str:
    out = "$\\begin{aligned}"
    if not inline:
      out += "\n"
    for reaction in self.fixated_reactions:
      out += reaction.to_latex(gt_model) + r" \\[-0.2cm]"
      if not inline:
        out += "\n"
    if len(self.fixated_reactions) > 0:
      out += r"\cline{1-3}"
    for reaction in self.non_fixated_reactions:
      out += reaction.to_latex(gt_model) + r" \\[-0.2cm]"
      if not inline:
        out += "\n"
    out += "\\end{aligned}$"
    return out

  def to_ode(self) -> dict:
    terms = {self.species_names[species]: [] for species in range(self.num_species)}
    for reaction in self.reactions:
      for species in range(self.num_species):
        if reaction.change[species] != 0 and reaction.rate != 0:
          term = f"{reaction.change[species] * reaction.rate:.2g}"
          term += "*" + "*".join(map(lambda x: self.species_names[x], reaction.reactands))
          for m in reaction._ma:
            if m >= 2:
              term += f"*1/{m}"
          terms[self.species_names[species]].append(term)
    return terms

  def to_reduced_ode(self) -> dict:
    terms = {self.species_names[species]: defaultdict(lambda: []) for species in range(self.num_species)}
    for reaction in self.reactions:
      for species in range(self.num_species):
        if reaction.change[species] != 0 and reaction.rate != 0:
          term = reaction.change[species] * reaction.rate
          term_vars = "*".join(map(lambda x: self.species_names[x], reaction.reactands))
          for m in reaction._ma:
            if m >= 2:
              term_vars += f"*1/{m}"
          terms[self.species_names[species]][term_vars].append(term)
    odes = {self.species_names[species]: [] for species in range(self.num_species)}
    for species_deriv in terms:
      for term in terms[species_deriv]:
        term_text = f"{sum(terms[species_deriv][term]):.2g}*{term}"
        odes[species_deriv].append(term_text)
    return odes

  def to_ode_str(self, reduce: bool = False, latex: bool = False) -> str:
    if reduce:
      odes = self.to_reduced_ode()
    else:
      odes = self.to_ode()
    odestr = ""
    if latex:
      odestr += r"$\begin{aligned}"
    for species in odes:
      if latex:
        odestr += f"\\frac{{d{species}}}{{dt}} = & "
        odestr += " + ".join(odes[species])
        odestr += r"\\"
      else:
        odestr += f"d{species}/dt = "
        odestr += " + ".join(odes[species])
        odestr += "\n"
    if latex: 
      odestr = odestr.replace("1/2", r"\frac{1}{2}")
      odestr = odestr.replace("*", "")
      odestr += r"\end{aligned}$"
    return odestr

  def print_model(self) -> str:
    # same as __str__ but skipping 0-rate reactions
    out = ""
    if self.fixated_reactions is not None:
      for reaction in self.fixated_reactions:
        if reaction.rate > 0.0:
          out += str(reaction) + " (fix)\n"
    for reaction in self.non_fixated_reactions:
      if reaction.rate > 0.0:
        out += str(reaction) + "\n"
    return out

  def clean_duplicate_reactions(self):
    self.non_fixated_reactions = list(sorted(set(self.non_fixated_reactions), key=self.non_fixated_reactions.index))

  def remove_sorted_out_reactions(self, thresh: float):
    non_fixated_reactions = []
    for r in self.non_fixated_reactions:
      if r.rate > thresh:
        non_fixated_reactions.append(r)
      #else:
      #  non_fixated_reactions.append(self.enumerator.get_random())
    self.non_fixated_reactions = non_fixated_reactions

  def clean_slow_reactions(self, thresh: float):
    non_fixated_reactions = []
    for r in self.non_fixated_reactions:
      if r.rate > thresh:
        non_fixated_reactions.append(r)
    self.non_fixated_reactions = non_fixated_reactions

  def apply_odes(self, t, state, pbar=None, pbar_state=None):
    if pbar is not None and pbar_state is not None:
      last_t, dt = pbar_state
      #n = int((t - last_t)/dt)
      n = (t - last_t)/dt
      pbar.update(n)
      pbar_state[0] = last_t + dt * n
    #else:
    #  print(t)

    deriv = np.zeros_like(state)
    for reaction in self.reactions:
      deriv += reaction.get_deriv_terms(state) * reaction.rate
    return deriv


