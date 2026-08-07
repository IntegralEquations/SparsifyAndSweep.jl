# Assembly of the sparse system H ũ = f of eq. (19) that approximates the dense
# Lippmann–Schwinger system (I + ω² K M) u = g.  Dimension-generic.

"Medium sample, zero-padded outside `I` (`m` is supported in Ω by assumption)."
@inline mval(g::Grid{D}, m::AbstractArray{Float64,D}, i::NTuple{D,Int}) where {D} =
    in_I(g, i) ? @inbounds(m[i...]) : 0.0
mval(g::Grid{2}, m::AbstractMatrix, i1::Integer, i2::Integer) =
    mval(g, m, (Int(i1), Int(i2)))

"""
    build_H(g, ω, m, astar, bstar; C = 12.0)

Assemble the sparsified system matrix `H` on `Ihη` (eq. 19):

* rows `i ∈ Ih`      : `α* ũ_{μ_i} + ω² β* [m ũ]_{μ_i}`
* rows `i ∈ Ihη\\Ih`  : `γ_i* ũ_{μ_i}`  (global PML, local frequency ω)
* zero Dirichlet data outside `Ihη`, by dropping out-of-range entries.

The PML stencils depend on the point only through the per-dimension increments
of `σ`, which vanish on `0 ≤ i_d ≤ n+1`; that whole range collapses onto one
cache key, so only `O(b^D)` distinct stencils are ever built.
"""
function build_H(
    g::Grid{D},
    ω::Real,
    m::AbstractArray{Float64,D},
    astar::Vector{ComplexF64},
    bstar::Vector{ComplexF64};
    C::Real = 12.0,
) where {D}
    prof = global_profile(g, ω, C)
    N = nunk(g)
    ω2 = ω^2
    off = offsets(Val(D))
    np = length(off)

    pmlkey(i) = (0 <= i <= g.n + 1) ? typemin(Int) : i
    cache = Dict{NTuple{D,Int},Vector{ComplexF64}}()
    dσ_cache = Dict{Int,NTuple{3,Float64}}()
    dσ(i) = get!(() -> dsigma(prof, g.h, i), dσ_cache, i)

    Ir = Int[]; Jc = Int[]; Vv = ComplexF64[]
    sizehint!(Ir, np * N); sizehint!(Jc, np * N); sizehint!(Vv, np * N)

    for i in boxindices(g)
        row = lin(g, i)
        if in_Ih(g, i)
            for p in 1:np
                j = i .+ off[p]
                inbox(g, j) || continue
                v = astar[p] + ω2 * bstar[p] * mval(g, m, j)
                iszero(v) && continue
                push!(Ir, row); push!(Jc, lin(g, j)); push!(Vv, v)
            end
        else
            key = ntuple(d -> pmlkey(i[d]), D)
            γ = get!(() -> pml_stencil(ω, g.h, ntuple(d -> dσ(i[d]), D)), cache, key)
            for p in 1:np
                j = i .+ off[p]
                inbox(g, j) || continue
                push!(Ir, row); push!(Jc, lin(g, j)); push!(Vv, γ[p])
            end
        end
    end
    return sparse(Ir, Jc, Vv, N, N)
end

"""
    rhs_operator(g, astar)

Sparse `|Ihη| x N` matrix `S` mapping a right-hand side `r` on `I` to the
right-hand side of the sparse system: `f_i = α* r_{μ_i}` for `i ∈ Ih` (with `r`
zero-padded outside `I`) and `f_i = 0` otherwise.
"""
function rhs_operator(g::Grid{D}, astar::Vector{ComplexF64}) where {D}
    off = offsets(Val(D))
    Ir = Int[]; Jc = Int[]; Vv = ComplexF64[]
    for cid in CartesianIndices(ntuple(_ -> 0:(g.n+1), D))
        i = Tuple(cid)
        row = lin(g, i)
        for p in eachindex(off)
            j = i .+ off[p]
            in_I(g, j) || continue
            push!(Ir, row); push!(Jc, plin(g, j)); push!(Vv, astar[p])
        end
    end
    return sparse(Ir, Jc, Vv, nunk(g), nphys(g))
end

# --------------------------------------------------------------------------- #
# The earlier scheme of Ying (2015): no PML extension, one-sided rows on ∂Ω
# --------------------------------------------------------------------------- #

"""
    build_H_ying(g, ω, m, astar, bstar, bnd)

Assemble Ying's eq. (15), `[A + Cω²m ; B]`: the same `α`/`β` stencils at points
whose full `3^D` neighbourhood lies in `I`, and the one-sided stencils `B` on the
boundary shell, which carry the radiation condition in place of a PML.  Lives on
`I` itself — `N = n^D` unknowns, with no extension.
"""
function build_H_ying(
    g::Grid{D},
    ω::Real,
    m::AbstractArray{Float64,D},
    astar::Vector{ComplexF64},
    bstar::Vector{ComplexF64},
    bnd::BoundaryStencils{D},
) where {D}
    n = g.n
    N = nphys(g)
    ω2 = ω^2
    off = offsets(Val(D))
    zs = ntuple(_ -> 0, D)
    Ir = Int[]; Jc = Int[]; Vv = ComplexF64[]
    sizehint!(Ir, length(off) * N); sizehint!(Jc, length(off) * N)
    sizehint!(Vv, length(off) * N)

    for i in physindices(g)
        row = plin(g, i)
        s = clip_signature(n, i)
        if s == zs
            for p in eachindex(off)
                j = i .+ off[p]
                v = astar[p] + ω2 * bstar[p] * @inbounds(m[j...])
                push!(Ir, row); push!(Jc, plin(g, j)); push!(Vv, v)
            end
        else
            o = bnd.offsets[s]; c = bnd.coef[s]
            for p in eachindex(o)
                push!(Ir, row); push!(Jc, plin(g, i .+ o[p])); push!(Vv, c[p])
            end
        end
    end
    return sparse(Ir, Jc, Vv, N, N)
end

"""
    rhs_operator_ying(g, astar, bnd)

The operator `[A ; B]` of Ying's eq. (9): same sparsity as `H` but without the
`ω²β⊙m` term.
"""
function rhs_operator_ying(g::Grid{D}, astar::Vector{ComplexF64},
                           bnd::BoundaryStencils{D}) where {D}
    n = g.n
    N = nphys(g)
    off = offsets(Val(D))
    zs = ntuple(_ -> 0, D)
    Ir = Int[]; Jc = Int[]; Vv = ComplexF64[]
    for i in physindices(g)
        row = plin(g, i)
        s = clip_signature(n, i)
        if s == zs
            for p in eachindex(off)
                push!(Ir, row); push!(Jc, plin(g, i .+ off[p])); push!(Vv, astar[p])
            end
        else
            o = bnd.offsets[s]; c = bnd.coef[s]
            for p in eachindex(o)
                push!(Ir, row); push!(Jc, plin(g, i .+ o[p])); push!(Vv, c[p])
            end
        end
    end
    return sparse(Ir, Jc, Vv, N, N)
end
