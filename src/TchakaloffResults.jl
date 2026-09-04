module TchakaloffResults

using CSV
import CairoMakie
using CaratheodoryPruning
using DataFrames
using LaTeXStrings
using LinearAlgebra
using NaNStatistics
using PolyChaos
using ProgressBars
using QuadGK: quadgk, quadgk_count
using Random
using Statistics

include("singular_hermite.jl")
include("least_squares_quadrature.jl")
include("paper_results.jl")

export SingularMixedHermiteInducedDistribution,
       LeastSquaresQuadrature,
       least_squares_quadrature,
       plot_results,
       set_theta!,
       trial_run,
       basis_and_pdfs_plots,
       quad_rule_plots
end
