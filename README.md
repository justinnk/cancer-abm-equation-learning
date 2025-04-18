[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15240283.svg)](https://doi.org/10.5281/zenodo.15240283)
# Learning Surrogate Equations for the Analysis of an Agent-Based Cancer Model

Code artifacts for the paper "Learning Surrogate Equations for the Analysis of an Agent-Based Cancer Model" submitted to Frontiers in Applied Mathematics and Statistics. 

## Authors and Contacts

- Kevin Burrage, Queensland University of Technology
- Pamela Burrage, Queensland University of Technology
- Justin Kreikemeyer, University of Rostock ([contact](https://mosi.informatik.uni-rostock.de/team/staff/justin-kreikemeyer/))
- Adelinde Uhrmacher, University of Rostock
- Hasitha Weerasinghe, Queensland University of Technology

## :rocket: Quickstart

1. Clone the repository and move to the repository folder:
```shell
git clone https://github.com/justinnk/cancer-abm-equation-learning
cd cancer-abm-equation-learning
```
2. Run the script `./setup_and_run.sh` to check requirements and install the dependencies. Note: you possibly have to edit the `PYTHON` variable inside this script to point to a Python binary of version 3.10. If this is the case, the script will take around 5 minutes to reproduce all results (cf. sections below for more information and where to find the figures).

## :cd: Setup

### Required Software

- Linux operating system; tested on Fedora 40
  - Should generally also work on Windows but adaptations may be required
- Python 3.10 (newer versions may not work!)
- The dependencies listed in `requirements.txt`
  - can be installed automatically, see the next section

### Required Hardware

- no special hardware requirements, an average Laptop suffices
- tested on a laptop with
  - 12th Gen Intel i7-1260P (16) @ 4.700GHz
  - 32 GB Ram

### Installation Guide
(for Linux)

This is the manual installation guide. If you used the quickstart script without problems, skip this section.

1. Clone the repository and move to the repository folder:
```shell
git clone https://github.com/justinnk/cancer-abm-equation-learning
cd cancer-abm-equation-learning
```
2. Create a virtual environment and install the dependencies
```shell
# depending on your linux distribution, you may have
# to use either "python3" or "python" in this step.
# if you typical python version `python --version` is
# too new, you may have to use "python3.10" instead!
# Afterwards, "python" suffices inside the venv.
python3 -m venv .venv      
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
You should now have a virtual environment stored in the folder `.venv` with all the dependencies installed.
 
## :file_folder: Overview of Contents

The following table provides an overview of the contents of the repository root.

| Folder/file                                               | Content/Purpose                                                                                                                                |
| ------:                                                   | :--------                                                                                                                                      |
| `01_data/`                                                | Contains the raw data from the ABM simulations with different parameters.                                                                      |
| `02_analysis_results/`                                    | After running the experiments, contains the plots resulting from the input data analysis.                                                      |
| `02_data_processed/`                                      | After running the experiments, contains the preprocessed data.                                                                                 |
| `04_results/`                                             | After running the experiments, containts all results of the equation learning, that from the basis for the figures and tables.                 |
| `01_preprocess.py`                                        | Preprocess the raw data (rename columns, correct timestep, ...).                                                                               |
| `02_analyze.py`                                           | Produce the data analysis plots.                                                                                                               |
| `03_fit_csindy.py`                                        | Perform the main equation learning (hyperparameter scan) using a copled version of SINDy.                                                      |
| `05_plot_matrix.py`                                       | Plot all hyperparameter scan results as a matrix.                                                                                              |
| `05_plot_union_fits.py`                                   | Plot analysis of the best model identified.                                                                                                    |
| `lib/`                                                    | Contains helper classes.                                                                                                                       |
| `lib/integrate.py`                                        | Numerical simulation of ODE reaction models.                                                                                                   |
| `lib/coupled_sindy.py`                                    | Equation learning of reaction models.                                                                                                          |
| `lib/reaction.py`                                         | Data structure and methods for single reactions.                                                                                               |
| `lib/reaction_library.py`                                 | Data structure and methods for libraries of reactions and reaction models.                                                                     |
| `lib/reaction_enumerator.py`                              | Systematically build "complete" libraries of reactions.                                                                                        |
| `lib/dataset.py`                                          | Methods to handle equation lerning data and numerical differentiation.                                                                         |
| `.gitignore`                                              | Contains list of paths that should not be included in the Git version control.                                                                 |
| `requirements.txt`                                        | Contains Python dependencies required by the Python scripts (can be instlled with `pip install -r requirements.txt`).                          |
| `setup_and_run.sh`                                        | Quick script to check system requirements, install dependencies, and run all experiments.                                                      |

## :balance_scale: License

This project is licensed under the MIT License contained in `LICENSE`, unless indicated otherwise in a file.

## :bar_chart: Reproduce Results

These are the steps to reproduce all the figures and tables in the paper.
The steps assume that you are inside the virtual environment created before with the installation guide above (using `source .venv/bin/activate`, if not already done).
Running all experiments may take around 5 to 10 minutes, depending on how fast your computer is. 

To reproduce all results, execute the following steps. You can alternatively use the script `setup_and_run.sh` which will automatically install dependencies and run everything in sequence.

1. `python 01_preprocess.py`
2. `python 02_analyze.py`
3. `python 03_fit_csindy.py`
4. `python 05_plot_matrix.py`
5. `python 05_plot_union_fits.py`

The equation learning result plots and table from the paper can now be found in the following places:

| Figure                | Path                                                                                                   |
| ---:                  | :---                                                                                                   |
| Figure 3              | `02_analysis_results/timeseries.pdf`                                                                   |
| Figure 4              | `06_aggregated_results_full_lib.pdf`                                                                   |
| Figure 5              | `06_aggregated_results_restricted_lib.pdf`                                                             |
| Figure 6              | `06_union_fits.pdf`                                                                                    |
| Table 1               | `04_results/union_accuracy_data.csv`                                                                   |


## :page_facing_up: Cite

Cite the preprint:

```
Burrage, K., Burrage, P., Kreikemeyer, J. N., Uhrmacher, A. M., & Weerasinghe, H. N.
"Learning Surrogate Equations for the Analysis of an Agent-Based Cancer Model."
arXiv preprint arXiv:2503.01718 (2025).
```

Bibtex:

```bib
@misc{burrage2025learning,
      title={Learning Surrogate Equations for the Analysis of an Agent-Based Cancer Model}, 
      author={Kevin Burrage and Pamela Burrage and Justin N. Kreikemeyer and Adelinde M. Uhrmacher and Hasitha N. Weerasinghe},
      year={2025},
      eprint={2503.01718},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01718}, 
}
```

