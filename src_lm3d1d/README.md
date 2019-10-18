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
```

Each benchmark is then defined in a submodule of `src_lm3d1d`. For example
`src_lm3d1d.poisson3d` is a uncoupled Poisson problem. To check solver defined
in the module (i) navigate to the source repo of `weak_bcs` and (ii) from there
launch convergence test by

```bash
python check_sanity.py src_lm3d1d.poisson3d -ncases 4 -plot 1
```

This will refine mesh 4 times, dump the rates in the `results` folder and
plots in `results/plots`.

## Results for formulation with multiplier on curve

```
l ndofs h e[|u3|_1] r[|u3|_1] e[|u1|_1] r[|u1|_1] e[|p|_{-1/2}] r[|p|_{-1/2}] niters dt |r|_2
1 135 4.33E-01 5.3358E-01 nan 5.3358E-01 nan 4.3197E-02 nan 9 0.01 7.49E-12 125 5 5
2 747 2.17E-01 2.5580E-01 1.06 2.5580E-01 1.06 1.0851E-02 1.99 21 0.02 8.13E-12 729 9 9
3 4947 1.08E-01 1.2641E-01 1.02 1.2641E-01 1.02 2.6942E-03 2.01 36 0.11 1.23E-11 4913 17 17
4 36003 5.41E-02 6.3019E-02 1.00 6.3019E-02 1.00 6.7138E-04 2.00 42 1.16 3.31E-11 35937 33 33
5 274755 2.71E-02 3.1486E-02 1.00 3.1486E-02 1.00 1.6670E-04 2.01 36 8.47 1.07E-10 274625 65 65
6 2146947 1.35E-02 1.9593E-04 7.33 1.5740E-02 1.00 4.1425E-05 2.01 31 64.62 3.77E-10 2146689 129 129
```
