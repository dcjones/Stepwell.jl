module Stepwell

#using Deldir: deldir
using DelaunayTriangulation
using LinearAlgebra: I
using LinearSolve
using LinearSolvePardiso
using Muon
using ProgressMeter
using Random: shuffle!
using SparseArrays
using Statistics: mean

export CellularNeighborhoodGraph, expected_absorption_time, shuffled_expected_absorption_time, local_shuffled_expected_absorption_time, normalized_expected_absorption_time


const default_solver = LinearSolve.UMFPACKFactorization

struct CellularNeighborhoodGraph
    ncells::Int
    senders::Vector{Int}
    receivers::Vector{Int}
    A::SparseMatrixCSC{UInt8,Int}
end

function Base.show(io::IO, G::CellularNeighborhoodGraph)
    println(io, typeof(G), "(ncells=", G.ncells, ", senders=…, receivers=…, A=…)")
end


function CellularNeighborhoodGraph(adata::AnnData)
    if !haskey(adata.obsm, "spatial")
        error("AnnData has no 'spatial' key in obsm. Spatial coordinates needed.")
    end

    xys = adata.obsm["spatial"]

    xs = Vector{Float64}(xys[:,1])
    ys = Vector{Float64}(xys[:,2])

    return CellularNeighborhoodGraph(xs, ys)
end


function CellularNeighborhoodGraph(xs::Vector{Float64}, ys::Vector{Float64})
    @assert length(xs) == length(ys)
    n = length(xs)

    xmin, xmax = extrema(xs)
    xs .-= xmin
    xs ./= xmax - xmin

    ymin, ymax = extrema(ys)
    ys .-= ymin
    ys ./= ymax - ymin

    println("Computing Delaunay triangulation...")
    tri = triangulate(collect(zip(xs, ys)))
    delgraph = get_graph(tri)
    ind1 = Int[]
    ind2 = Int[]
    for (i, j) in DelaunayTriangulation.get_edges(delgraph)
        if i != j && i > 0 && j > 0
            push!(ind1, i)
            push!(ind2, j)
        end
    end
    println("Done.")

    A = adjacency_matrix(n, ind1, ind2)

    return CellularNeighborhoodGraph(n, ind1, ind2, A)
end


"""
Build sparse adjacency matrix from edges.
"""
function adjacency_matrix(ncells::Int, senders::Vector{T}, receivers::Vector{T}) where {T<:Integer}
    @assert length(senders) == length(receivers)
    nedges = length(senders)
    A = sparse(senders, receivers, fill(0x1, nedges), ncells, ncells)
    A += A'
    return A
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
        niter::Int=100)

    shuffled_senders = copy(G.senders)
    shuffled_receivers = copy(G.receivers)

    E = zeros(Float32, G.ncells)
    for i in 1:niter
        rewire_graph!(shuffled_senders, shuffled_receivers)
        E .+= expected_absorption_time(G.ncells, shuffled_senders, shuffled_receivers, absorbing_states)
    end
    E ./= niter

    return E
end


"""
Send every node on a `k`-step random walk with transition probailities `P`.
"""
function random_walk!(A::SparseMatrixCSC, destination::Vector{Int}, k::Int)
    @assert size(A, 1) == size(A, 2)
    n = size(A, 1)
    # Threads.@threads for i in 1:n
    for i in 1:n
        destination[i] = i
        for step in 1:k
            A.colptr[destination[i]] == A.colptr[destination[i]+1] && continue
            j = rand(A.colptr[destination[i]]:A.colptr[destination[i]+1]-1)
            destination[i] = A.rowval[j]
        end
    end
end



"""
Do a local shuffle by sending nodes on a k-step random walk.
"""
function local_shuffled_expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool};
        niter::Int=200, k::Int=100, solver=nothing)

    destinations = zeros(Int, G.ncells)

    eat = expected_absorption_time(G, absorbing_states, solver=solver)
    shuffled_eat = zeros(Float32, G.ncells)

    for iter in 1:niter
        random_walk!(G.A, destinations, k+1)
        shuffled_eat .+= eat[destinations] .+ 1
    end
    shuffled_eat ./= niter
    println("Done.")

    return shuffled_eat
end



"""
Compute the overall expected absorption time for transcient nodes in a null
model formed by shuffling node labels.
"""
function shuffled_expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool};
        niter::Int=100)

    shuffled_absorbing_states = copy(absorbing_states)
    E = 0.0
    for i in 1:niter
        shuffle!(shuffled_absorbing_states)
        E += mean(expected_absorption_time(G.ncells, G.senders, G.receivers, shuffled_absorbing_states)[.!shuffled_absorbing_states])
    end

    return E / niter
end


"""
Compute expected absorption time relative to absorption time after a local shuffle, thus
normalizing for the local cell type composition.
"""
function normalized_expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool};
        niter::Int=200, k::Int=100, solver=nothing)

    eat = expected_absorption_time(G, absorbing_states, solver=solver)
    shuffled_eat = local_shuffled_expected_absorption_time(G, absorbing_states, niter=niter, k=k, solver=solver)
    normalized_eat = eat ./ shuffled_eat

    # essentially defininig 0/0 = 1 for the purposes of normalization
    normalized_eat[eat .== 0] .= 1.0

    return normalized_eat
end

"""
Compute the expected absorption time for every node.
"""
function expected_absorption_time(
        G::CellularNeighborhoodGraph, absorbing_states::AbstractVector{Bool}; solver=nothing)

    return expected_absorption_time(G.ncells, G.senders, G.receivers, absorbing_states, solver=solver)
end


function expected_absorption_time(
        ncells::Int, senders::Vector{Int}, receivers::Vector{Int}, absorbing_states::AbstractVector{Bool};
        solver=nothing)
    sink_count = sum(absorbing_states)

    if sink_count == 0
        return fill(Inf32, ncells)
    end

    transient_states = .!absorbing_states
    transient = (1:ncells)[transient_states]

    # Some solvers work only with Float64
    #T = Float32
    T = Float64

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
        E = solve(linprob, default_solver())
    else
        E = solve(linprob, solver())
    end

    Efull = zeros(Float32, ncells)
    Efull[transient] .= E

    return Efull
end

end # module Stepwell
