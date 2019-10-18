# Content

This folder contains scripts for checking LM 3d-1d coupled solvers. The
components are checked by running `./test.sh` (having assigned execute
rights to the script). A simple benchmark problem is included to check
convergence properties of the two formulations. Note that the geometries
and setups are particularly simple. The way to interpret the results is
that the implementation is not entirely wrong. That is, if the results are
positive.

## Dependencies

 - FeniCS 2017.2.0
 - FEniCS_ii incomplete-trace branch 
 - weak_bcs master branch
 - cbc.block

## Running benchmark problems

The scripts are meant to be run using `weak_bcs` module and therefore
each script(module) must conform to the API requested by `weak_bcs`. This
folder it self is a module (`src_lm3d1d`) and should be put on python path,
e.g. by running from this folder

```bash
cd ..
export PYTHONPATH=`pwd`:"$PYTHONPATH"
``

Each benchmark is then defined in a submodule of `src_lm3d1d`. For example
`src_lm3d1d.poisson3d` is a uncoupled Poisson problem. To check solver defined
in the module (i) navigate to the source repo of `weak_bcs` and (ii) from there
launch convergence test by

```bash
python check_sanity.py src_lm3d1d.poisson3d -ncases 4 -plot 1
```

This will refine mesh 4 times, dump the rates in the `results` folder and
plots in `results/plots`.
