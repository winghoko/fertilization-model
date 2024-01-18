# fertilization-model

Implements the broadcast spawning fertilization model described in [REF] in python. Provides both python module(s) and example jupyter notebook(s).

The core of this repository is the `fertilization.py` python module. This module defines a `FertilizationModel` class that encapsulate the broadcast spawning fertilization model. To use this module, the third-party packages `numpy` and `scipy` are required. 

All jupyter notebooks (`.ipynb` files) in this repository amonut to examples of using this core module, and corresponds to the case studies presented in [REF]. To run these notebooks and all the codes therein, the third-party packages `jupyter` (optionally `jupyterlab`), `matplotlib`, and `pandas` are required.

The case studies that corresponds to each jupyter notebook, in roughly increasing order of sophistication, are:

+ `fertilization_speed_var`: one group of sperms, one group of eggs, constant sperm % motile, time-varying sperm swimming speed.

+ `fertilization_chi_var`: one group of sperms, one group of eggs, constant sperm swimming speed, time-varying % motile.

+ `fertilization_worst_var`: one group of sperms, one group of eggs, time varying sperm % motile _and_ swimming speed.

+ `fertilization_mixed_eggs`: one group of sperms, two groups of eggs.

+ `fertilization_mixed_sperms`: two groups of sperms, one group of eggs.

+ `fertilization_hybrid`: two groups (species) of sperms, two groups (species) of eggs; asymmetric fertilizability matrix that models hybridization.