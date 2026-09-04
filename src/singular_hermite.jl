mutable struct SingularMixedHermiteInducedDistribution
    N::Int
    theta::Float64
    polynomial::AbstractOrthoPoly
    singularities::Vector{Float64}
    singular_orders::Vector{Float64}
    roots::Vector{Float64}
    root_orders::Vector{Float64}
    transform::Matrix{Float64}
    moments::Vector{Float64}
    x_grid::Vector{Float64}
    induced_cdf::Vector{Float64}
    representer_cdf::Vector{Float64}
    representer_normalization::Float64
    mixture_cdf::Vector{Float64}
end

standard_normal_density(x::Real) = exp(-x^2 / 2) / sqrt(2pi)

function singular_raw_basis(d::SingularMixedHermiteInducedDistribution, x::Real)
    v = ones(Float64, d.N)
    for j in eachindex(d.roots)
        v[2] *= abs(x - d.roots[j])^d.root_orders[j]
    end
    for j in eachindex(d.singularities)
        v[3] *= abs(x - d.singularities[j])^(-d.singular_orders[j])
    end
    d.N > 3 && (v[4:end] .= vec(evaluate(1:(d.N - 3), x, d.polynomial)))
    return v
end

function singular_raw_basis(d::SingularMixedHermiteInducedDistribution,
        xs::AbstractVector{<:Real})
    V = ones(Float64, length(xs), d.N)
    for j in eachindex(d.roots)
        @views V[:, 2] .*= abs.(xs .- d.roots[j]) .^ d.root_orders[j]
    end
    for j in eachindex(d.singularities)
        @views V[:, 3] .*= abs.(xs .- d.singularities[j]) .^ (-d.singular_orders[j])
    end
    d.N > 3 && (V[:, 4:end] .= evaluate(1:(d.N - 3), xs, d.polynomial))
    return V
end

singular_basis(d::SingularMixedHermiteInducedDistribution, x::Real) =
    transpose(d.transform) * singular_raw_basis(d, x)
singular_basis(d::SingularMixedHermiteInducedDistribution,
    xs::AbstractVector{<:Real}) = singular_raw_basis(d, xs) * d.transform

induced_density_from_basis(d::SingularMixedHermiteInducedDistribution,
    v::AbstractVector) = sum(abs2, v) / d.N
induced_density_from_basis(d::SingularMixedHermiteInducedDistribution,
    V::AbstractMatrix) = vec(sum(abs2, V; dims=2)) ./ d.N

function representer_density_from_basis(d::SingularMixedHermiteInducedDistribution,
        v::AbstractVector)
    Labs = abs(dot(d.moments, v))
    Winv = sqrt(induced_density_from_basis(d, v))
    return Labs * Winv / d.representer_normalization
end

function representer_density_from_basis(d::SingularMixedHermiteInducedDistribution,
        V::AbstractMatrix)
    Labs = abs.(V * d.moments)
    Winv = sqrt.(induced_density_from_basis(d, V))
    return Labs .* Winv ./ d.representer_normalization
end

mixture_density_from_basis(d::SingularMixedHermiteInducedDistribution,
    v::Union{AbstractVector,AbstractMatrix}) =
        d.theta .* induced_density_from_basis(d, v) .+
        (1 - d.theta) .* representer_density_from_basis(d, v)

function quantile_from_grid(d::SingularMixedHermiteInducedDistribution, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "quantile probability must lie in [0, 1]"))
    iszero(q) && return -Inf
    isone(q) && return Inf
    i = searchsortedfirst(d.mixture_cdf, q)
    i <= 1 && return first(d.x_grid)
    i > length(d.x_grid) && return last(d.x_grid)
    c0, c1 = d.mixture_cdf[i - 1], d.mixture_cdf[i]
    x0, x1 = d.x_grid[i - 1], d.x_grid[i]
    c1 <= c0 && return x1
    return x0 + (q - c0) * (x1 - x0) / (c1 - c0)
end

Base.rand(rng::AbstractRNG, d::SingularMixedHermiteInducedDistribution) =
    quantile_from_grid(d, rand(rng))
Base.rand(rng::AbstractRNG, d::SingularMixedHermiteInducedDistribution, M::Int) =
    [rand(rng, d) for _ in 1:M]
Base.rand(d::SingularMixedHermiteInducedDistribution, M::Int) =
    rand(Random.default_rng(), d, M)

rand_standard(rng::AbstractRNG, ::SingularMixedHermiteInducedDistribution, M::Int) =
    randn(rng, M)
rand_standard(d::SingularMixedHermiteInducedDistribution, M::Int) =
    rand_standard(Random.default_rng(), d, M)

function raw_gram_quadgk(d::SingularMixedHermiteInducedDistribution;
        rtol=1e-10, atol=1e-12, maxevals=10^7)
    pairs = [(i, j) for j in 1:d.N for i in 1:j]
    breakpoints = sort(unique(vcat(-Inf, d.roots, d.singularities, Inf)))
    function integrand(x)
        density = standard_normal_density(x)
        iszero(density) && return zeros(Float64, length(pairs))
        v = singular_raw_basis(d, x)
        return [density * v[i] * v[j] for (i, j) in pairs]
    end
    packed_gram, error, count = quadgk_count(
        integrand, breakpoints...; rtol, atol, maxevals, norm=v -> norm(v, Inf))
    G = Matrix{Float64}(undef, d.N, d.N)
    for (value, (i, j)) in zip(packed_gram, pairs)
        G[i, j] = value
        G[j, i] = value
    end
    return Hermitian(G), error, count
end

function representer_normalization_quadgk(d::SingularMixedHermiteInducedDistribution;
        rtol=1e-10, atol=1e-12, maxevals=10^7)
    breakpoints = sort(unique(vcat(-Inf, d.roots, d.singularities, Inf)))
    function integrand(x)
        density = standard_normal_density(x)
        iszero(density) && return 0.0
        v = singular_basis(d, x)
        return density * abs(dot(d.moments, v)) *
               sqrt(induced_density_from_basis(d, v))
    end
    return quadgk(integrand, breakpoints...; rtol, atol, maxevals)
end

"""Construct the singular/zero-enriched Hermite basis used in the paper."""
function SingularMixedHermiteInducedDistribution(N=5;
        theta::Real=1.0,
        cdf_grid_size::Int=200_000,
        cdf_grid_radius=nothing,
        singularities=[-1, 0, 1],
        singular_orders=[0.24, 0.24, 0.24],
        roots=[-0.5, 0.5],
        root_orders=[0.49, 0.49])
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    0 <= theta <= 1 || throw(ArgumentError("theta must lie in [0, 1]"))
    cdf_grid_size >= N || throw(ArgumentError("cdf_grid_size must be at least N"))
    length(singularities) == length(singular_orders) ||
        throw(ArgumentError("singularities and singular_orders must have equal length"))
    length(roots) == length(root_orders) ||
        throw(ArgumentError("roots and root_orders must have equal length"))

    singularities = Float64.(singularities)
    singular_orders = Float64.(singular_orders)
    roots = Float64.(roots)
    root_orders = Float64.(root_orders)
    theta = Float64(theta)
    grid_radius = isnothing(cdf_grid_radius) ?
        max(17.0, sqrt(2 * (N - 1)) + 6) : Float64(cdf_grid_radius)
    polynomial = OrthoPoly(GaussMeasure(), N - 1, Nrec=max(N + 1, 20))
    d = SingularMixedHermiteInducedDistribution(
        N, theta, polynomial, singularities, singular_orders, roots, root_orders,
        Matrix{Float64}(I, N, N), zeros(N), Float64[], Float64[], Float64[],
        1.0, Float64[])

    gram, _, _ = raw_gram_quadgk(d)
    factor = cholesky(gram)
    d.transform .= factor.U \ Matrix{Float64}(I, N, N)
    d.moments .= factor.U[:, 2]
    d.representer_normalization = first(representer_normalization_quadgk(d))

    dx = 2 * grid_radius / cdf_grid_size
    d.x_grid = collect(range(-grid_radius + dx / 2, grid_radius - dx / 2;
                             length=cdf_grid_size))
    for location in singularities
        i = searchsortedfirst(d.x_grid, location)
        i <= length(d.x_grid) && d.x_grid[i] == location &&
            (d.x_grid[i] = nextfloat(d.x_grid[i]))
    end
    V = singular_basis(d, d.x_grid)
    normal = standard_normal_density.(d.x_grid)
    d.induced_cdf = cumsum(normal .* induced_density_from_basis(d, V))
    d.induced_cdf ./= last(d.induced_cdf)
    d.representer_cdf = cumsum(normal .* representer_density_from_basis(d, V))
    d.representer_cdf ./= last(d.representer_cdf)
    d.mixture_cdf = @. d.theta * d.induced_cdf +
                        (1 - d.theta) * d.representer_cdf
    return d
end

function set_theta!(d::SingularMixedHermiteInducedDistribution, theta::Real)
    0 <= theta <= 1 || throw(ArgumentError("theta must lie in [0, 1]"))
    d.theta = Float64(theta)
    @. d.mixture_cdf = d.theta * d.induced_cdf +
                       (1 - d.theta) * d.representer_cdf
    return d
end

basis_dimension(d::SingularMixedHermiteInducedDistribution) = d.N
default_moments(d::SingularMixedHermiteInducedDistribution) = copy(d.moments)

function basis_matrix_and_weighting(d::SingularMixedHermiteInducedDistribution,
        nodes::AbstractVector{Float64}, standard=false)
    V = singular_basis(d, nodes)
    tau2 = standard ? ones(length(nodes)) : mixture_density_from_basis(d, V)
    return V, tau2
end
