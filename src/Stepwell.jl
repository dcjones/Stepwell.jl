module Stepwell

using Muon
using Deldir: deldir
using LinearSolve: LinearProblem, solve
using LinearSolvePardiso: MKLPardisoFactorize
using LinearAlgebra: I
using SparseArrays

export CellularNeighborhoodGraph, expected_absorption_time

struct CellularNeighborhoodGraph
    adata::AnnData
    edges::Vector{Tuple{Int,Int}}
end


function CellularNeighborhoodGraph(adata::AnnData)
    if !haskey(adata.obsm, "spatial")
        error("AnnData has no 'spatial' key in obsm. Spatial coordinates needed.")
    end

    xys = adata.obsm["spatial"]

    xs = Vector{Float64}(xys[:,1])
    xmin, xmax = extrema(xs)
    xs .-= xmin
    xs ./= xmax - xmin

    ys = Vector{Float64}(xys[:,2])
    ymin, ymax = extrema(ys)
    ys .-= ymin
    ys ./= ymax - ymin

    del, vor, summ = deldir(xs, ys)

    return CellularNeighborhoodGraph(
        adata, collect(Tuple{Int, Int}, zip(del.ind1, del.ind2)))
end


function expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool})

    ncells = size(G.adata, 1)
    sink_count = sum(absorbing_states)

    if sink_count == 0
        return fill(Inf32, ncells)
    end

    transient_states = .!absorbing_states
    transient = (1:ncells)[transient_states]

    P = zeros(Float32, (ncells, ncells)) # transition probabilities

    # count used edges
    nedges = 0
    degree = zeros(Int, ncells)
    for (i, j) in G.edges
        if !absorbing_states[i]
            nedges += 1
            degree[i] += 1
        end

        if !absorbing_states[j]
            nedges += 1
            degree[j] += 1
        end
    end

    from = Vector{Int32}(undef, nedges)
    to = Vector{Int32}(undef, nedges)
    weight = ones(Float32, nedges)

    # neighbor edges
    k = 0
    for (i, j) in G.edges
        if !absorbing_states[i]
            k += 1
            from[k] = i
            to[k] = j
            weight[k] = 1/degree[i]
        end

        if !absorbing_states[j]
            k += 1
            from[k] = j
            to[k] = i
            weight[k] = 1/degree[j]
        end
    end
    @assert k == nedges
    P = sparse(from, to, weight, ncells, ncells)
    Q = P[transient,transient]

    linprob = LinearProblem(I - Q, ones(Float32, size(Q, 1)))
    E = solve(linprob)

    Efull = zeros(Float32, ncells)
    Efull[transient] .= E

    return Efull
end

end # module Stepwell
