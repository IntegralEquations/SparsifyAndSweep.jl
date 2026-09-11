# Sweeping factorisation of the sparsified system (Liu & Ying, §2.3 and §3.1).
#
# The grid is cut into slices D₁,…,D_ℓ along x₁ so that H is block tridiagonal.
# Block Gaussian elimination needs T_[i] = S_[i]^{-1}, the (i,i) block of
# H_{[1:i,1:i]}^{-1}.  The moving-PML approximation replaces the half-domain
# D_{1:i} by the two-slice problem (21): D_i itself plus an auxiliary PML placed
# on D_{i-1}, whose rows are modified-plane-wave stencils built with the local
# frequency ω√(1-m(x)).
#
# Since the radiation condition holds on all sides, the elimination runs from
# both ends towards a middle slice (remark 1 after Algorithm 2, and the
# configuration used for the paper's tables).  That is the twisted block
# factorisation: exact when the Schur complements are, and it halves the number
# of approximate solution operators composed along any path.  `fronts = 1`
# recovers the plain left-to-right sweep of Algorithms 1–2.
#
# ---------------------------------------------------------------------------
# Sweep axis, and the bi-directional alternation of Zepeda-Núñez & Zhao (SIAM J.
# Sci. Comput. 38 (2016) B866–B888, Alg. 2).
#
# Nothing below singles out x₁: `sweep_setup(...; axis = 2)` cuts the grid into
# slices along x₂ instead, and `axis = -1` runs the one-front sweep towards -x₁.
# A sweep along one axis reproduces every wave that never turns back along that
# axis — the moving PML discards what leaves a slice and returns — so in a medium
# that reflects (a disc, a lattice of discs) a single sweep is one reflection
# short, and each GMRES iteration buys one more.  Zepeda-Núñez & Zhao sweep once
# in horizontal layers, form the residual, sweep the residual in vertical layers
# and add:  u = M₁ f + M₂ (f − H M₁ f).  Waves that turn back along x₁ but not
# along x₂ live inside the x₂ slices and are exact there, so the second pass
# recovers what the first discarded, up to the waves that turn back along both.
# `AlternatingSweep` below is that construction for any list of factorisations
# applied in turn to the running residual: `axes = (1, 2)` is their algorithm;
# `axes = (1, -1)` with one front alternates the direction on one axis instead.
# The same wrapper chains L-sweeps (lsweep.jl): a second L-sweep on a
# checkerboard shifted by half a box, applied to the residual of the first.
# ---------------------------------------------------------------------------
# Recursion (§3.1, "the recursive approach", ref. [20]).
#
# Each slice subproblem is itself a sparse system on a slab: quasi-1D in 2D,
# quasi-2D in 3D.  The nonrecursive method factorises it directly, which in 3D
# costs O(b³n³) per slice and O(b²N^{4/3}) overall.  The recursive method sweeps
# that slab along the next axis instead, cutting it into quasi-1D subproblems
# and bringing the total to O(b⁴N) setup and O(b²N) application — linear in N,
# but more sensitive to b.
#
# Both are the same algorithm applied at successive depths, so what follows is a
# recursion over a list of sweep axes: `levels = 1` sweeps x₁ only and solves
# each slab directly; `levels = 2` sweeps x₁ and then x₂.  A slab whose axis list
# is exhausted, or which is too thin to cut, is factorised exactly.
# ---------------------------------------------------------------------------

"""
    slice_ranges(r, b)

Partition the index range `r` into slices.  The first and last carry `2b` layers
(the `b` PML layers plus `b` interior ones); the rest are cut into slices of
width `b`, the last of which absorbs any remainder.
"""
function slice_ranges(r::UnitRange{Int}, b::Integer)
    lo, hi = first(r), last(r)
    total = hi - lo + 1
    total >= 4b || return [lo:hi]           # too small to sweep: single slice
    rs = UnitRange{Int}[lo:(lo+2b-1)]
    laststart = hi - 2b + 1
    cur = lo + 2b
    while cur <= laststart - 1
        rem = laststart - cur
        w = (rem >= 2b) ? b : rem           # fold a short tail into one slice
        push!(rs, cur:(cur+w-1))
        cur += w
    end
    push!(rs, laststart:hi)
    return rs
end
slice_ranges(g::Grid) = slice_ranges(ilo(g):ihi(g), g.b)

"Local linear index of the point `i` inside the box `box`, first axis fastest."
@inline function boxlin(box::NTuple{D,UnitRange{Int}}, i::NTuple{D,Int}) where {D}
    idx = 0
    @inbounds for d in D:-1:1
        idx = idx * length(box[d]) + (i[d] - first(box[d]))
    end
    return idx + 1
end
boxlen(box::NTuple{D,UnitRange{Int}}) where {D} = prod(length, box)
boxpoints(box::NTuple{D,UnitRange{Int}}) where {D} =
    (Tuple(c) for c in CartesianIndices(box))
"Local indices of every point of `sub` inside `box`, first axis fastest."
boxidx(box::NTuple{D,UnitRange{Int}}, sub::NTuple{D,UnitRange{Int}}) where {D} =
    vec([boxlin(box, i) for i in boxpoints(sub)])

# --------------------------------------------------------------------------- #
# Slab solvers
# --------------------------------------------------------------------------- #

"""
    SlabSolver

Solves the sparse system on one box.  Either an exact sparse LU
([`DirectSlab`](@ref)) or a twisted moving-PML sweep along one axis whose slice
subproblems are themselves `SlabSolver`s ([`SweptSlab`](@ref)).
"""
abstract type SlabSolver end

struct DirectSlab <: SlabSolver
    fact::SparseArrays.UMFPACK.UmfpackLU{ComplexF64,Int}
end

struct SweptSlab <: SlabSolver
    dir::Int                                        # axis being swept
    mid::Int                                        # meeting slice of the fronts
    idx::Vector{Vector{Int}}                        # slice → indices in this box
    loc::Vector{Vector{Int}}                        # slice → indices in its subproblem
    naux::Vector{Int}                               # size of each subproblem
    sub::Vector{SlabSolver}
    Alow::Vector{SparseMatrixCSC{ComplexF64,Int}}   # A[D_s, D_{s-1}]
    Aup::Vector{SparseMatrixCSC{ComplexF64,Int}}    # A[D_s, D_{s+1}]
end

nsub(S::SweptSlab) = length(S.idx)

slab_solve(S::DirectSlab, b::AbstractVector{ComplexF64}) = S.fact \ b

"Apply the approximate solution operator of slice `s` to a vector on that slice."
function slab_apply_T(S::SweptSlab, s::Int, v::AbstractVector{ComplexF64})
    rhs = zeros(ComplexF64, S.naux[s])
    @inbounds rhs[S.loc[s]] = v
    return slab_solve(S.sub[s], rhs)[S.loc[s]]
end

"""
    slab_solve(S::SweptSlab, b)

Algorithm 2: both fronts advance independently towards the middle slice, the two
contributions meet there, and the solution is back-substituted outwards.
"""
function slab_solve(S::SweptSlab, b::AbstractVector{ComplexF64})
    ℓ = nsub(S)
    c = S.mid
    v = Vector{Vector{ComplexF64}}(undef, ℓ)

    if c > 1
        v[1] = slab_apply_T(S, 1, b[S.idx[1]])
        for s in 2:(c-1)
            v[s] = slab_apply_T(S, s, b[S.idx[s]] - S.Alow[s] * v[s-1])
        end
    end
    if c < ℓ
        v[ℓ] = slab_apply_T(S, ℓ, b[S.idx[ℓ]])
        for s in (ℓ-1):-1:(c+1)
            v[s] = slab_apply_T(S, s, b[S.idx[s]] - S.Aup[s] * v[s+1])
        end
    end

    rmid = b[S.idx[c]]
    c > 1 && (rmid -= S.Alow[c] * v[c-1])
    c < ℓ && (rmid -= S.Aup[c] * v[c+1])
    u = Vector{Vector{ComplexF64}}(undef, ℓ)
    u[c] = slab_apply_T(S, c, rmid)

    for s in (c-1):-1:1
        u[s] = v[s] - slab_apply_T(S, s, S.Aup[s] * u[s+1])
    end
    for s in (c+1):ℓ
        u[s] = v[s] - slab_apply_T(S, s, S.Alow[s] * u[s-1])
    end

    out = zeros(ComplexF64, length(b))
    for s in 1:ℓ
        @inbounds out[S.idx[s]] = u[s]
    end
    return out
end

# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

"""
    SweepContext

Everything the moving-PML rows need that does not depend on the recursion depth:
the medium, the local-frequency samples, and a cache of `γ` stencils keyed by
(frequency sample, per-axis σ increments).  The increments are exact function
outputs, so they compare equal whenever the geometry repeats, and the cache stays
concretely typed at any depth.
"""
struct SweepContext{D}
    g::Grid{D}
    ω::Float64
    m::Array{Float64,D}
    C::Float64
    κsamp::Vector{Float64}
    lo2::Float64
    step2::Float64
    fronts::Int
    dir::Int                             # one front: +1 sweeps towards +x_d, -1 towards -x_d
    cache::Dict{Tuple{Int,NTuple{D,NTuple{3,Float64}}},Vector{ComplexF64}}
end

"Slice where the fronts meet: the middle for two fronts, the far end of the sweep for one."
meeting_slice(ctx::SweepContext, ℓ::Int) =
    ctx.fronts == 2 ? cld(ℓ, 2) : (ctx.dir > 0 ? ℓ : 1)

@inline function snapfreq(ctx::SweepContext, κ2::Real)
    (length(ctx.κsamp) == 1 || ctx.step2 <= 0) && return 1     # constant medium: one sample
    return clamp(round(Int, (κ2 - ctx.lo2) / ctx.step2) + 1, 1, length(ctx.κsamp))
end

"The moving-PML row at `i`, given the σ increments in force on each axis."
function pml_row!(ctx::SweepContext{D}, i::NTuple{D,Int}, dσf) where {D}
    k = snapfreq(ctx, ctx.ω^2 * (1 - mval(ctx.g, ctx.m, i)))
    dσ = ntuple(d -> dσf[d](i[d])::NTuple{3,Float64}, D)
    return get!(ctx.cache, (k, dσ)) do
        pml_stencil(ctx.κsamp[k], ctx.g.h, dσ)
    end
end

"""
    build_slab(ctx, A, box, dirs, dσf)

Recursively build a solver for the sparse system `A` on `box`.  `dirs` lists the
axes still to be swept; when it runs out — or the box is too thin to cut — the
system is factorised exactly.  `dσf[d]` maps an index on axis `d` to the σ
increments in force there, and is overridden on each auxiliary PML slab as the
recursion descends, so a point inside two nested PMLs sees both stretchings.
"""
function build_slab(ctx::SweepContext{D}, A::SparseMatrixCSC{ComplexF64,Int},
                    box::NTuple{D,UnitRange{Int}}, dirs::Vector{Int}, dσf) where {D}
    isempty(dirs) && return DirectSlab(lu(A))
    d = dirs[1]
    rest = dirs[2:end]
    h = ctx.g.h
    rs = slice_ranges(box[d], ctx.g.b)
    ℓ = length(rs)
    mid = meeting_slice(ctx, ℓ)
    off = offsets(Val(D))

    slicebox(r) = ntuple(k -> k == d ? r : box[k], D)
    idx = [boxidx(box, slicebox(r)) for r in rs]

    sub = Vector{SlabSolver}(undef, ℓ)
    loc = Vector{Vector{Int}}(undef, ℓ)
    naux = Vector{Int}(undef, ℓ)
    Alow = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, ℓ)
    Aup = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, ℓ)

    for s in 1:ℓ
        padlo = s > 1 && s <= mid
        padhi = s < ℓ && s >= mid
        lo = padlo ? first(rs[s-1]) : first(rs[s])
        hi = padhi ? last(rs[s+1]) : last(rs[s])
        abox = ntuple(k -> k == d ? (lo:hi) : box[k], D)
        naux[s] = boxlen(abox)
        loc[s] = boxidx(abox, slicebox(rs[s]))

        # rows of the slice: inherited verbatim from A, columns outside abox dropped
        pcols = boxidx(box, abox)
        Acur = A[idx[s], pcols]
        ri, ci, vi = findnz(Acur)
        Ir = [loc[s][k] for k in ri]
        Jc = copy(ci)
        Vv = copy(vi)

        # σ on axis d is replaced by the auxiliary profile over the pad slabs
        base = dσf[d]
        rlo = padlo ? rs[s-1] : nothing
        rhi = padhi ? rs[s+1] : nothing
        plo = padlo ?
              left_profile(ctx.C / ctx.ω, length(rs[s-1]) * h, last(rs[s-1]) * h) :
              nothing
        phi = padhi ?
              right_profile(ctx.C / ctx.ω, length(rs[s+1]) * h, first(rs[s+1]) * h) :
              nothing
        newd = function (j::Int)
            rlo !== nothing && j in rlo && return dsigma(plo, h, j)
            rhi !== nothing && j in rhi && return dsigma(phi, h, j)
            return base(j)::NTuple{3,Float64}
        end
        newdσf = ntuple(k -> k == d ? newd : dσf[k], D)

        for r in (rlo, rhi)
            r === nothing && continue
            for i in boxpoints(slicebox(r))
                γ = pml_row!(ctx, i, newdσf)
                r0 = boxlin(abox, i)
                for p in eachindex(off)
                    j = i .+ off[p]
                    all(k -> first(abox[k]) <= j[k] <= last(abox[k]), 1:D) || continue
                    push!(Ir, r0); push!(Jc, boxlin(abox, j)); push!(Vv, γ[p])
                end
            end
        end

        As = sparse(Ir, Jc, Vv, naux[s], naux[s])
        sub[s] = build_slab(ctx, As, abox, rest, newdσf)

        s > 1 && (Alow[s] = A[idx[s], idx[s-1]])
        s < ℓ && (Aup[s] = A[idx[s], idx[s+1]])
    end
    return SweptSlab(d, mid, idx, loc, naux, sub, Alow, Aup)
end

"""
    SweepFactorization

Approximate twisted block factorisation of `H`.  `levels` is the number of axes
swept: `1` is the nonrecursive method of §2.3 (each slice factorised directly),
`2` the recursive method of §3.1 (each slice swept again along the next axis).
`axis` is the outermost swept axis, signed: `-1` is a one-front sweep towards
`-x₁`, `2` a sweep along `x₂`.
"""
struct SweepFactorization{D}
    g::Grid{D}
    ranges::Vector{UnitRange{Int}}
    gidx::Vector{Vector{Int}}
    mid::Int
    top::SweptSlab
    nstencil::Int
    levels::Int
    axis::Int
end

nslices(F::SweepFactorization) = length(F.ranges)
dim(::SweepFactorization{D}) where {D} = D

"Number of exact factorisations at the leaves, and their largest size."
function leafstats(S::SlabSolver)
    S isa DirectSlab && return (1, size(S.fact, 1))
    n, mx = 0, 0
    for c in (S::SweptSlab).sub
        a, b = leafstats(c)
        n += a
        mx = max(mx, b)
    end
    return (n, mx)
end
leafstats(F::SweepFactorization) = leafstats(F.top)

"""
    sweep_setup(g, ω, m, H; C = 12.0, nsamp = 200, fronts = 2, levels = 1, axis = 1)

Build the moving-PML sweeping factorisation of `H` (Algorithm 1, steps 3-8).

`fronts = 2` sweeps from both ends towards a middle slice; `fronts = 1` is the
plain left-to-right sweep.  `levels = 1` factorises each slice subproblem
directly; `levels = 2` sweeps it again along the next axis — the recursive
approach of §3.1, which only changes anything in 3D, where the subproblems are
quasi-2D.

`axis` selects the axis the slices are cut along, `1` (the paper's `x₁`) by
default; `axis = 2` sweeps along `x₂`.  Its sign fixes the direction of a
one-front sweep: `axis = -1` eliminates from the last slice towards the first.
Two fronts are symmetric and ignore the sign.  See [`alternating_setup`](@ref)
for the bi-directional combination of sweeps along different axes.

`nsamp` local-frequency samples are drawn uniformly from
`[ω²(1-max m), ω²(1-min m)]`; each auxiliary-PML point is assigned the stencil of
the closest sample, so the number of small SVDs stays independent of `N`.
"""
function sweep_setup(
    g::Grid{D},
    ω::Real,
    m::AbstractArray{Float64,D},
    H::SparseMatrixCSC{ComplexF64,Int};
    C::Real = 12.0,
    nsamp::Integer = 200,
    fronts::Integer = 2,
    levels::Integer = 1,
    axis::Integer = 1,
) where {D}
    fronts in (1, 2) || throw(ArgumentError("fronts must be 1 or 2"))
    1 <= levels <= D || throw(ArgumentError("levels must be between 1 and $D"))
    1 <= abs(axis) <= D || throw(ArgumentError("axis must be ±1, …, ±$D (got $axis)"))
    a = abs(Int(axis))

    mmin, mmax = extrema(m)
    lo2, hi2 = ω^2 * (1 - mmax), ω^2 * (1 - mmin)
    ns = max(1, Int(nsamp))
    dκ2 = ns > 1 ? (hi2 - lo2) / (ns - 1) : 0.0
    κsamp = [sqrt(lo2 + (k - 1) * dκ2) for k in 1:ns]
    ctx = SweepContext{D}(g, Float64(ω), Array(m), Float64(C), κsamp, lo2,
                          ns > 1 ? dκ2 : 1.0, Int(fronts), Int(sign(axis)),
                          Dict{Tuple{Int,NTuple{D,NTuple{3,Float64}}},
                               Vector{ComplexF64}}())

    prof = global_profile(g, ω, C)
    gcache = Dict{Int,NTuple{3,Float64}}()
    gdσ(i::Int) = get!(() -> dsigma(prof, g.h, i), gcache, i)
    dσf = ntuple(_ -> gdσ, D)

    box = ntuple(_ -> ilo(g):ihi(g), D)
    dirs = vcat(a, [d for d in 1:D if d != a])[1:Int(levels)]   # the chosen axis first
    top = build_slab(ctx, H, box, dirs, dσf)
    top isa SweptSlab || throw(ArgumentError("grid is too small to sweep"))
    return SweepFactorization{D}(g, slice_ranges(g), top.idx, top.mid, top,
                                 length(ctx.cache), Int(levels), Int(axis))
end

"Apply the approximate solution operator `T̃_[s]` of the outermost sweep."
apply_T(F::SweepFactorization, s::Int, v::AbstractVector{ComplexF64}) =
    slab_apply_T(F.top, s, v)

"Approximate solve `H ũ ≈ f` (Algorithm 2)."
sweep_solve(F::SweepFactorization, f::AbstractVector{ComplexF64}) =
    slab_solve(F.top, f)

"Distinct moving-PML stencils built for a factorisation."
nstencils(F::SweepFactorization) = F.nstencil

"One sweep, as a pass of an [`AlternatingSweep`](@ref)."
approx_solve(F::SweepFactorization, f::AbstractVector{ComplexF64}) = sweep_solve(F, f)

# --------------------------------------------------------------------------- #
# Bi-directional alternation
# --------------------------------------------------------------------------- #

"""
    AlternatingSweep

Approximate factorisations applied in turn to the running residual,

    u ← M₁ f,    u ← u + M_k (f − H u)   for k = 2, 3, …

With two layered factorisations along orthogonal axes this is the bi-directional
preconditioner of Zepeda-Núñez & Zhao (Alg. 2): one pass in horizontal layers,
one step of iterative refinement, one pass in vertical layers.  With two L-sweep
factorisations, the second on a checkerboard shifted by half a box, it is the
same idea for L-sweeps: the residual of an L-sweep lies on the layers either
side of its box interfaces, and the second board's boxes hold those layers in
their interiors.  Any mixture works; a pass is whatever `approx_solve` accepts.
It is a fixed linear operator, `M₁ + M₂ − M₂ H M₁`, so it can precondition plain
GMRES; each application costs one pass per factorisation plus one sparse matvec
with `H` per refinement.  Build with [`alternating_setup`](@ref) (layered) or
directly from factorisations, apply with [`alternating_solve`](@ref).
"""
struct AlternatingSweep{D,T}
    H::SparseMatrixCSC{ComplexF64,Int}
    F::Vector{T}
end
AlternatingSweep{D}(H::SparseMatrixCSC{ComplexF64,Int}, F::Vector{T}) where {D,T} =
    AlternatingSweep{D,T}(H, F)

dim(::AlternatingSweep{D}) where {D} = D
nstencils(A::AlternatingSweep) = sum(nstencils, A.F)
leafstats(A::AlternatingSweep) = (sum(F -> leafstats(F)[1], A.F), maximum(F -> leafstats(F)[2], A.F))

"""
    alternating_setup(g, ω, m, H; axes = (1, 2), C = 12.0, nsamp = 200, fronts = 2, levels = 1)

Build one [`sweep_setup`](@ref) factorisation per entry of `axes` (each a signed
axis, see `sweep_setup`) and chain them into an [`AlternatingSweep`](@ref).
`axes = (1, 2)` is the bi-directional preconditioner of Zepeda-Núñez & Zhao;
`axes = (1, -1)` with `fronts = 1` alternates the sweep direction on one axis;
`axes = (1,)` is a plain sweep with a residual-free wrapper.  Setup time and
memory grow with the number of factorisations.
"""
function alternating_setup(
    g::Grid{D},
    ω::Real,
    m::AbstractArray{Float64,D},
    H::SparseMatrixCSC{ComplexF64,Int};
    axes = (1, 2),
    kwargs...,
) where {D}
    ax = collect(Int, axes)
    isempty(ax) && throw(ArgumentError("axes must name at least one sweep axis"))
    F = [sweep_setup(g, ω, m, H; axis = a, kwargs...) for a in ax]
    return AlternatingSweep{D}(H, F)
end

"""
    alternating_solve(A, f)

`H ũ ≈ f` by the sweeps of `A` in turn, each applied to the residual the
previous ones left.
"""
function alternating_solve(A::AlternatingSweep, f::AbstractVector{ComplexF64})
    u = approx_solve(A.F[1], f)
    length(A.F) == 1 && return u
    e = similar(u)
    for k in 2:length(A.F)
        mul!(e, A.H, u)
        @. e = f - e
        u .+= approx_solve(A.F[k], e)
    end
    return u
end

"Global unknown indices of the slice `r` (an `x₁` range), ordered `x₁`-fast."
function collect_slice(g::Grid{D}, r::UnitRange{Int}) where {D}
    rest = ntuple(_ -> ilo(g):ihi(g), D - 1)
    idx = Vector{Int}(undef, length(r) * ntot(g)^(D - 1))
    c = 0
    for cid in CartesianIndices(rest), i1 in r
        idx[c += 1] = lin(g, (i1, Tuple(cid)...))
    end
    return idx
end
