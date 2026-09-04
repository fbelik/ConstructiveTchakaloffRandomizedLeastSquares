using TchakaloffResults

function generate_figures(results_directory="data/", figures_directory="figures/")
    # Figure 1 from the paper
    basis_and_pdfs_plots(figures_directory)
    # Figure 2 from the paper (along with additional heatmaps)
    plot_results(results_directory; save_to=figures_directory)
    # Figure 3 from the paper
    quad_rule_plots(figures_directory)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_figures(ARGS...)
end