module Stepwell

using Muon
using Deldir: deldir
using LinearSolve
using LinearSolvePardiso
using LinearAlgebra: I
using SparseArrays
using Random: shuffle!
using Statistics: mean

export CellularNeighborhoodGraph, expected_absorption_time, shuffled_expected_absorption_time


const default_solver = MKLPardisoFactorize

struct CellularNeighborhoodGraph
    adata::AnnData
    senders::Vector{Int}
    receivers::Vector{Int}
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

    return CellularNeighborhoodGraph(adata, del.ind1, del.ind2)
end


# Randomize a graph preserving each node's in- and out-degree.
function rewire_graph!(senders::Vector{Int}, receivers::Vector{Int}; nswaps_scale::Int=1)
    m = length(senders)
    @assert length(receivers) == m

    nswaps = nswaps_scale * m
    while nswaps > 0
        # choose two edges
        i = rand(1:m)
        j = rand(1:m)

        # the must be different edges
        if i == j
            continue
        end

        # and have different senders and receivers
        if senders[i] == senders[j] || receivers[i] == receivers[j]
            continue
        end

        # swap edges
        receivers[i], receivers[j] = receivers[j], receivers[i]

        nswaps -= 1
    end
end


"""
Compute expected absorption time for each node in a null model produced by
shuffling edges, preserving in- and out-degree.
"""
function edge_shuffled_expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool};
        niter::Int=500)

    shuffled_senders = copy(G.senders)
    shuffled_receivers = copy(G.receivers)
    ncells = size(G.adata, 1)

    E = zeros(Float32, ncells)
    for i in 1:niter
        rewire_graph!(shuffled_senders, shuffled_receivers)
        E .+= expected_absorption_time(ncells, shuffled_senders, shuffled_receivers, absorbing_states)
    end
    E ./= niter

    return E
end


"""
Compute the overall expected absorption time for transcient nodes in a null
model formed by shuffling node labels.
"""
function shuffled_expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool};
        niter::Int=500)

    ncells = size(G.adata, 1)
    shuffled_absorbing_states = copy(absorbing_states)
    E = 0.0
    for i in 1:niter
        shuffle!(shuffled_absorbing_states)
        E += mean(expected_absorption_time(ncells, G.senders, G.receivers, shuffled_absorbing_states)[.!shuffled_absorbing_states])
    end

    return E / niter
end


"""
Compute the expected absorption time for every node.
"""
function expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool})

    return expected_absorption_time(size(G.adata, 1), G.senders, G.receivers, absorbing_states)
end


function expected_absorption_time(
        ncells::Int, senders::Vector{Int}, receivers::Vector{Int}, absorbing_states::AbstractVector{Bool};
        solver=default_solver())
    sink_count = sum(absorbing_states)

    if sink_count == 0
        return fill(Inf32, ncells)
    end

    transient_states = .!absorbing_states
    transient = (1:ncells)[transient_states]

    # Some solvers work only with Float64
    # T = Float32
    T = Float64

    P = zeros(T, (ncells, ncells)) # transition probabilities

    # count used edges
    nedges = 0
    degree = zeros(Int, ncells)
    for (i, j) in zip(senders, receivers)
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
    weight = ones(T, nedges)

    # neighbor edges
    k = 0
    for (i, j) in zip(senders, receivers)
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

    linprob = LinearProblem(I - Q, ones(T, size(Q, 1)))
    if solver === nothing
        E = solve(linprob)
    else
        E = solve(linprob, solver)
    end

    Efull = zeros(Float32, ncells)
    Efull[transient] .= E

    return Efull
end

end # module Stepwell
