"""
The fixed-node least-squares rule needed by the paper experiment.

This deliberately has no square-space matrix or insertion/removal state.
"""
mutable struct LeastSquaresQuadrature{D}
    M::Int
    distribution::D
    V::Matrix{Float64}
    tau2::Vector{Float64}
    gram::Union{Matrix{Float64},Cholesky{Float64,Matrix{Float64}}}
    moments::Vector{Float64}
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function set_weights!(q::LeastSquaresQuadrature)
    if q.M > size(q.gram, 1)
        try
            q.weights .= vec(transpose((transpose(q.moments) / q.gram) *
                transpose(q.V) * Diagonal(1 ./ q.tau2))) ./ q.M
            return q
        catch error
            error isa SingularException || rethrow()
        end
    end
    q.weights .= NaN
    return q
end

function least_squares_quadrature(d, M::Int;
        moments=default_moments(d), standard=false)
    nodes = standard ? rand_standard(d, M) : rand(d, M)
    V, tau2 = basis_matrix_and_weighting(d, nodes, standard)
    gram = (transpose(V) * Diagonal(1 ./ tau2) * V) ./ M
    try
        gram = cholesky(Hermitian(gram))
    catch error
        error isa PosDefException || rethrow()
    end
    q = LeastSquaresQuadrature(M, d, V, tau2, gram, Float64.(moments),
                               nodes, zeros(M))
    return set_weights!(q)
end

concentration(q::LeastSquaresQuadrature) = q.gram isa Cholesky ?
    opnorm(Matrix(q.gram) - I) : opnorm(q.gram - I)

# This is the paper experiment's historical `kappa` quantity.
condition_number(q::LeastSquaresQuadrature) = norm(q.weights, 1)

reference_weights(q::LeastSquaresQuadrature) =
    vec(transpose(transpose(q.moments) * transpose(q.V))) ./ q.tau2

function max_rel_error(q::LeastSquaresQuadrature)
    reference = reference_weights(q)
    result = 0.0
    for i in eachindex(reference)
        absolute_error = abs(q.M * q.weights[i] - reference[i])
        if abs(reference[i]) > 0
            result = max(result, absolute_error / abs(reference[i]))
        elseif absolute_error > 0
            return Inf
        end
    end
    return result
end

max_abs_error(q::LeastSquaresQuadrature) =
    maximum(abs.(q.M .* q.weights .- reference_weights(q)))
