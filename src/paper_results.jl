CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
CairoMakie.update_theme!(CairoMakie.Theme(fontsize=18))

# (per-rule quantity, across-repeat statistic, CSV/PDF stem, plot title, scale)
const QOIS = [
    (concentration, xs -> Statistics.mean(xs .<= 0.5), "conc_succ_prob",
     L"\mathbb{P}\left[||G^{(M)}-I||\leq\frac{1}{2}\right]", identity),
    (condition_number, nanmedian, "kappa_med",
     L"\mathrm{median}\left(\kappa(\mathcal{Q}_M)\right)", log10),
    (max_rel_error, nanmedian, "sigma_rel_med",
     L"\mathrm{median}\left(\sigma_{\mathrm{rel}}(\mathcal{Q}_M)\right)", log10),
    (max_abs_error, nanmedian, "sigma_abs_med",
     L"\mathrm{median}\left(\sigma_{\mathrm{abs}}(\mathcal{Q}_M)\right)", log10),
    (q -> q.M * maximum(abs, q.weights), xs -> nanpctile(xs, 95), "max_w_95per",
     L"Q_{0.95}\left(\max |M w_m| \right)", log10),
    (q -> minimum(q.weights), xs -> nanmean(xs .>= 0.0), "pos_prob",
     L"\mathbb{P}\left[w_m\geq 0 \;\forall\; m \right]", identity),
    (q -> any(isnan, q.weights), sum, "nan_prob",
     L"\#\left[G^{(M)} \;\;\mathrm{singular}\right]", identity),
]

"""Generate `standard.csv`, `induced.csv`, and `mixture05.csv`."""
function trial_run(; repeats=1000, Ns=4:2:60, alphas=1:0.375:10.0,
        save_to="data/", seed=123, cdf_grid_size=200_000)
    mkpath(save_to)
    Random.seed!(seed)
    storages = [Matrix{Float64}(undef, repeats, length(QOIS)) for _ in 1:3]
    names = [:N, :alpha, :M, Symbol.([qoi[3] for qoi in QOIS])...]
    types = [Int, Float64, Int, fill(Float64, length(QOIS))...]
    template = DataFrame([T[] for T in types], names)
    frames = [copy(template) for _ in 1:3]
    progress = ProgressBar(total=length(Ns) * length(alphas))

    for N in Ns
        d = SingularMixedHermiteInducedDistribution(N; cdf_grid_size)
        for alpha in alphas
            M = ceil(Int, N * log(N) * alpha)
            storages[1][:, end] .= 0
            storages[2][:, end] .= 0
            storages[3][:, end] .= 0
            for repeat in 1:repeats
                standard = least_squares_quadrature(d, M; standard=true)
                set_theta!(d, 1.0)
                induced = least_squares_quadrature(d, M)
                set_theta!(d, 0.5)
                mixture = least_squares_quadrature(d, M)
                rules = (standard, induced, mixture)
                for (iqoi, qoi) in enumerate(QOIS), j in 1:3
                    storages[j][repeat, iqoi] = qoi[1](rules[j])
                end
            end
            rows = [Any[N, alpha, M] for _ in 1:3]
            for (iqoi, qoi) in enumerate(QOIS), j in 1:3
                push!(rows[j], qoi[2](view(storages[j], :, iqoi)))
            end
            foreach(j -> push!(frames[j], rows[j]), 1:3)
            set_description(progress, "On N=$N/$(last(Ns))")
            update(progress)
        end
        for (filename, frame) in zip(("standard.csv", "induced.csv", "mixture05.csv"), frames)
            CSV.write(joinpath(save_to, filename), frame)
        end
    end
    return frames
end

function _heatmap_matrix(df::DataFrame, column::Symbol, Ns, alphas)
    N_indices = Dict(N => i for (i, N) in enumerate(Ns))
    alpha_indices = Dict(alpha => i for (i, alpha) in enumerate(alphas))
    values = fill(NaN, length(Ns), length(alphas))
    for row in eachrow(df)
        values[N_indices[row.N], alpha_indices[row.alpha]] = row[column]
    end
    return values
end

function _shared_colorrange(matrices)
    values = filter(isfinite, reduce(vcat, vec.(matrices)))
    isempty(values) && throw(ArgumentError("cannot plot heatmaps without finite values"))
    lo, hi = extrema(values)
    lo > 0 && (lo = 10.0^floor(Int, log10(lo)) * (1 - 1e-4))
    hi > 0 && (hi = 10.0^ceil(Int, log10(hi)) * (1 + 1e-4))
    lo == hi == 0 && return (0, 1)
    lo == hi && return (lo - abs(lo) * 0.05, hi + abs(hi) * 0.05)
    return (lo, hi)
end

"""Read the three experiment CSVs and regenerate the paper PDFs."""
function plot_results(path="data/"; save_to="figures/", extension="pdf",
        colormap=:gist_ncar, figure_size=(800, 1050))
    frames = [CSV.read(joinpath(path, filename), DataFrame) for filename in
              ("standard.csv", "induced.csv", "mixture05.csv")]
    sampling_names = [L"\mathrm{Standard}", L"\mathrm{Induced}",
                      L"Mixture ($\theta = 0.5$)"]
    Ns = sort(unique(frames[1].N))
    alphas = sort(unique(frames[1].alpha))
    for frame in frames[2:end]
        sort(unique(frame.N)) == Ns || error("CSV files have different N values")
        sort(unique(frame.alpha)) == alphas || error("CSV files have different alpha values")
    end

    mkpath(save_to)
    file_extension = startswith(extension, ".") ? extension[2:end] : extension
    figures = Dict{String,Any}()
    for qoi in QOIS
        matrices = [_heatmap_matrix(frame, Symbol(qoi[3]), Ns, alphas) for frame in frames]
        colorrange = _shared_colorrange(matrices)
        fig = CairoMakie.Figure(size=figure_size)
        CairoMakie.Label(fig[0, 1:2], qoi[4], fontsize=22)
        axes = CairoMakie.Axis[]
        heatmaps = Any[]
        for row in 1:3
            axis = CairoMakie.Axis(fig[row, 1]; title=sampling_names[row],
                xlabel=row == 3 ? "N" : "",
                ylabel=L"\alpha\approx M/(N\log(N))",
                xticks=Ns[1]:4:Ns[end], yticks=floor(Int, alphas[1]):ceil(Int, alphas[end]),
                xticklabelsvisible=row == 3, xticksvisible=row == 3)
            heatmap = CairoMakie.heatmap!(axis, Ns, alphas, matrices[row];
                                         colormap, colorrange, colorscale=qoi[5])
            push!(axes, axis)
            push!(heatmaps, heatmap)
        end
        CairoMakie.linkxaxes!(axes...)
        CairoMakie.linkyaxes!(axes...)
        colorbar = CairoMakie.Colorbar(fig[1:3, 2], first(heatmaps))
        if qoi[5] == log10
            lo, hi = colorrange
            colorbar.ticks = CairoMakie.LogTicks(ceil(Int, log10(lo)):floor(Int, log10(hi)))
        end
        CairoMakie.save(joinpath(save_to, "$(qoi[3]).$file_extension"), fig)
        figures[qoi[3]] = fig
    end
    return figures
end

function basis_and_pdfs_plots(path="figures/"; N=5)
    d = SingularMixedHermiteInducedDistribution(N)
    mkpath(path)

    # Basis and representer plot
    xs = collect(range(-3,3,1000))
    append!(xs, range(-1-1e-3,-1+1e-3,1000))
    append!(xs, range(-.5-1e-3,-.5+1e-3,1000))
    append!(xs, range(0-1e-3,0+1e-3,1000))
    append!(xs, range(.5-1e-3,.5+1e-3,1000))
    append!(xs, range(1-1e-3,1+1e-3,1000))
    sort!(xs)
    raw_basis = singular_raw_basis(d, xs)
    ortho_basis = singular_basis(d, xs)

    ortho_basis_figure = CairoMakie.Figure(size=(800, 500))
    ortho_basis_axis = CairoMakie.Axis(
        ortho_basis_figure[1, 1];
        xlabel=L"x",
        ylabel=L"y",
        title="Modified Hermite Basis",
    )
    CairoMakie.xlims!(ortho_basis_axis, (xs[1],xs[end]))
    CairoMakie.ylims!(ortho_basis_axis, (-3,5))
    for j in axes(ortho_basis, 2)
        CairoMakie.lines!(
            ortho_basis_axis, xs, view(ortho_basis, :, j);
            color=(:gray45, 0.65),
            linewidth=1.5,
            label=j == 1 ? "Orthonormal basis" : nothing,
        )
    end
    CairoMakie.lines!(
        ortho_basis_axis, xs, view(raw_basis, :, 2);
        color=:red,
        linewidth=3,
        label="Riesz representer",
    )
    CairoMakie.axislegend(ortho_basis_axis; position=:rb)
    CairoMakie.save(joinpath(path, "ortho_basis.pdf"), ortho_basis_figure)

    # PDF plot
    xs = collect(range(-5,5,1000))
    append!(xs, range(-1-1e-3,-1+1e-3,1000))
    append!(xs, range(-.5-1e-3,-.5+1e-3,1000))
    append!(xs, range(0-1e-3,0+1e-3,1000))
    append!(xs, range(.5-1e-3,.5+1e-3,1000))
    append!(xs, range(1-1e-3,1+1e-3,1000))
    sort!(xs)
    raw_basis = singular_raw_basis(d, xs)
    ortho_basis = singular_basis(d, xs)
    standard_pdf = standard_normal_density.(xs)
    induced_pdf = standard_pdf .* induced_density_from_basis(d, ortho_basis)
    representer_pdf = standard_pdf .* representer_density_from_basis(d, ortho_basis)

    pdf_figure = CairoMakie.Figure(size=(800, 500))
    pdf_axis = CairoMakie.Axis(
        pdf_figure[1, 1];
        xticks=-5:5,
        xlabel=L"x",
        ylabel="Density",
        title="Sampling densities",
    )
    CairoMakie.lines!(
        pdf_axis, xs, standard_pdf;
        color=:black,
        linestyle=:dash,
        linewidth=2,
        label="Standard Gaussian",
    )
    CairoMakie.lines!(
        pdf_axis, xs, induced_pdf;
        linewidth=2,
        label="Induced",
    )
    CairoMakie.lines!(
        pdf_axis, xs, 0.5 .* (induced_pdf .+ representer_pdf);
        linewidth=2,
        label=L"\theta = 0.5",
    )
    CairoMakie.lines!(
        pdf_axis, xs, representer_pdf;
        linewidth=2,
        label=L"\theta = 0",
    )
    CairoMakie.xlims!(pdf_axis, (xs[1], xs[end]))
    CairoMakie.ylims!(pdf_axis, (0,1.0))
    CairoMakie.axislegend(pdf_axis; position=:rt)
    CairoMakie.save(joinpath(path, "pdfs.pdf"), pdf_figure)
end

function quad_rule_plots(path="figures/"; N=10, M=500, atts=100, seed=12345)
    d = SingularMixedHermiteInducedDistribution(N)
    xs_omega = range(-6,6,10000)
    V = singular_basis(d, xs_omega)
    L = V * d.moments

    Random.seed!(seed)
    qs = [least_squares_quadrature(d, M) for _ in 1:atts];
    q = qs[findfirst(q -> minimum(q.weights) > 0, qs)]
    w,inds = caratheodory_pruning(q.V, q.weights)
    ys_omega = L ./ induced_density_from_basis(d, V)
    fig1 = CairoMakie.Figure(size=(800, 500))
    ax1 = CairoMakie.Axis(
        fig1[1, 1];
        xticks=-6:6,
        limits=(-6,6,-0.25,3.0),
        xlabel=L"x",
        ylabel=L"y",
        title=L"Induced Sampling Positive Rules with $N=%$N$, $M=%$M$",
    )
    CairoMakie.scatter!(ax1, q.nodes, M .* q.weights, label=L"w_m\cdot M")
    CairoMakie.lines!(ax1, xs_omega, ys_omega,
                           color = :green,
                           label=L"\omega(x)")
    CairoMakie.scatter!(ax1, q.nodes[inds], w[inds] .* N, label=L"w_N \cdot N", 
                        color=:red, markersize=15)
    CairoMakie.axislegend(ax1, position=:lt)
    CairoMakie.save(joinpath(path, "induced_rule.pdf"), fig1)

    theta = 0.5
    set_theta!(d, theta)
    qs = [least_squares_quadrature(d, M) for _ in 1:atts];
    q = qs[findfirst(q->minimum(q.weights)>0,qs)]
    w,inds = caratheodory_pruning(q.V, q.weights)
    ys_omega = L ./ mixture_density_from_basis(d, V)
    fig2 = CairoMakie.Figure(size=(800, 500))
    ax2 = CairoMakie.Axis(
        fig2[1, 1];
        xticks=-6:6,
        limits=(-6,6,-0.25,3.0),
        xlabel=L"x",
        ylabel=L"y",
        title=L"Mixture Sampling ($\theta$=0.5) Positive Rules with $N=%$N$, $M=%$M$",
    )
    CairoMakie.scatter!(ax2, q.nodes, M .* q.weights, label=L"w_m\cdot M")
    CairoMakie.lines!(ax2, xs_omega, ys_omega,
                           color = :green,
                           label=L"\omega(x)")
    CairoMakie.scatter!(ax2, q.nodes[inds], w[inds] .* N, label=L"w_N \cdot N", 
                        color=:red, markersize=15)
    CairoMakie.axislegend(ax2, position=:lt)
    CairoMakie.save(joinpath(path, "mixture_rule.pdf"), fig2)

    qs = [least_squares_quadrature(d, M, standard=true) for _ in 1:atts];
    q = qs[findfirst(q->minimum(q.weights)>0,qs)]
    w,inds = caratheodory_pruning(q.V, q.weights)
    ys_omega = L
    fig3 = CairoMakie.Figure(size=(800, 500))
    ax3 = CairoMakie.Axis(
        fig3[1, 1];
        xticks=-6:6,
        limits=(-6,6,-0.25,3.0),
        xlabel=L"x",
        ylabel=L"y",
        title=L"Standard Sampling Positive Rules with $N=%$N$, $M=%$M$",
    )
    CairoMakie.scatter!(ax3, q.nodes, M .* q.weights, label=L"w_m\cdot M")
    CairoMakie.lines!(ax3, xs_omega, ys_omega,
                           color = :green,
                           label=L"\omega(x)")
    CairoMakie.scatter!(ax3, q.nodes[inds], w[inds] .* N, label=L"w_N \cdot N", 
                        color=:red, markersize=15)
    CairoMakie.axislegend(ax3, position=:lt)
    CairoMakie.save(joinpath(path, "standard_rule.pdf"), fig3)

end
