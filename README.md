# Tchakaloff singular-Hermite paper results

This is the code used to generate the three singular-Hermite CSV files and 
figures accompanying the paper. 

## Setup

From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
or, within Julia,
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Reproduce the Figures

To generate the data in the folder `output_data/`:
```bash
julia --project=. scripts/generate_csvs.jl output_data/
```
or, within Julia,
```julia
include("scripts/generate_csvs.jl")
generate_csvs("output_data/")
```
Without providing `output_data/` the results will by default overwrite
contents in `data/`.
This uses the paper settings: `seed=123`, `repeats=1000`, even basis dimensions
`Ns=4:2:60`, and alpha values `alphas=1:0.375:10`.

To then generate the figures and save them to `output_figures/`, run
```bash
julia --project=. scripts/generate_figures.jl output_data/ output_figures/
```
or, within Julia,
```julia
include("scripts/generate_figures.jl")
generate_figures("output_data/", "output_figures/")
```
Without providing arguments, it will assume the data is stored in `data/` and
will save the figures to `figures/`.


It is a substantial computation (~8hrs). To run a smaller experiment
interactively, for example, within Julia:
```julia
using TchakaloffResults
trial_run(repeats=50, Ns=4:2:10, alphas=2:4:10, save_to="output_data/")
include("scripts/generate_figures.jl")
generate_figures("output_data/", "output_figures/")
```
## Additional Notes
The files `ortho_basis.pdf` and `pdfs.pdf` are Figure 1 in the paper.
The files `conc_succ_prob.pdf`, `kappa_med.pdf`, and `max_w_95per.pdf`
are Figure 2 in the paper. Note that the associated code computes and
plots additional statistics which are not provided in the paper. 
They are as follows:

- `nan_prob.pdf` - Number of rules for which the Gramian was not invertible
- `sigma_rel_med.pdf` - Median relative error between the weights and their reference values
- `sigma_abs_med.pdf` - Median absolute error between the weights and their reference values
- `pos_prob.pdf` - Probability the intermediate rule has strictly positive weights

The files `standard_rule.pdf`, `induced_rule.pdf`, and `mixture_rule.pdf`
are Figure 3 in the paper.