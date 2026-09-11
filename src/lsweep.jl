# L-sweeps for the sparsified system (Taus, Zepeda-Núñez, Hewett & Demanet,
# J. Comput. Phys. 420 (2020) 109706), built from the moving-PML ingredients of
# Liu & Ying, and organised as a dataflow over boxes so that the same code runs
# on Julia threads, on MPI ranks, or on both at once.
#
# Why.  The layered sweep of sweep.jl has a critical path of one slab solve per
# slice, so a single right-hand side never parallelises beyond its two fronts.
# L-sweeps replace the layers by a Cartesian domain decomposition (CDD) into
# q x r boxes and move the wavefield box to box through *polarized* local solves:
# given the values of a field v on the two grid layers Γ₁ | Γ₂ either side of an
# interface, and a local matrix A that agrees with the global one near it,
#
#     A U = ( 0, A_{Γ₁Γ₂} v_{Γ₂}, -A_{Γ₂Γ₁} v_{Γ₁}, 0 )
#
# has U = 0 on the near side and U = v on the far side (their eq. 8-9).  This is
# an algebraic identity for any block-tridiagonal splitting, so it holds for the
# sparsified H verbatim: the 3^D-point stencil couples only adjacent layers.
#
# Local problems.  Each box is extended by two unstretched layers and then by
# `pad` layers of the fitted modified-plane-wave PML rows of stencils.jl, at the
# local frequency ω√(1-m) — the rows the moving PML plants inside the medium,
# here on all four sides.  The stretching starts *inside* the pad (σ = 0 on the
# first pad row), the convention the moving PML uses and the one the fitted
# stencils need.  Boxes on the outer boundary keep the global PML there.  Only
# the stencils are needed to assemble a box, never H, so every rank builds and
# factorises exactly the boxes it owns.
#
# Sweep (their §2.1.3, "scenario 3": sources anywhere, none on the skeleton —
# automatic on a grid whose interfaces run between grid lines):
#
#   stage 1  local solve in every box;
#   stage 2  four cardinal sweeps — right, left, up, down — along rows/columns;
#   stage 3  four diagonal sweeps corner to corner through L-shaped traces.
#
# The field of a box is kept as nine components: its own local solution, what
# arrived from each cardinal direction, and what arrived along each diagonal.
# The trace handed to a neighbour sums exactly the components that carry the
# sources of the relevant half-plane or quadrant (their Figs. 12-14), which is
# what prevents double counting.  The corner degree of freedom of an L-shaped
# trace belongs to the diagonal neighbour and takes its value from there.
#
# Parallel structure.  A *node* is (box, direction): the arrival of one
# component in one box.  A node fires when all its *contributions* — trace
# values extracted from neighbouring boxes — are present; firing it makes the
# contributions that depend on it extractable.  Ownership of boxes is a
# partition over ranks (row-based by default, as in the paper's Fig. 18); a
# contribution whose source and target live on different ranks travels as one
# small message.  Within a rank, nodes are executed by a pool of tasks, so the
# rows, columns and anti-diagonals of the paper overlap as far as the data
# dependencies allow.  Each node's arithmetic depends only on its inputs, in a
# fixed order, so sequential, threaded and distributed runs agree bitwise.
#
# Cost per application: 9 local solves per box against 2 slab solves per slice
# for the layered sweep, with concurrency q instead of 2 and O(1)-size local
# problems; O(N/p) on p ≤ q ranks.

using Base.Threads: Atomic, atomic_add!, atomic_sub!

const NCOMP = 9
const C_OWN = 1
"The eight receive directions `(si, sj)`: si acts on rows (axis 2), sj on columns (axis 1)."
const DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
"Index in `DIRS` of a direction; its component is `dirindex + 1`."
function dirindex(si::Int, sj::Int)
    for (k, d) in enumerate(DIRS)
        d == (si, sj) && return k
    end
    throw(ArgumentError("bad direction $((si, sj))"))
end
const NodeId = Tuple{Int,Int,Int}          # (i, j, k): box (i, j), k = 0 own, 1..8 direction

"Does box `(i, j)` of a `q x r` CDD receive along direction `k` (`k = 0` is its own solve)?"
function node_exists(i::Int, j::Int, k::Int, q::Int, r::Int)
    k == 0 && return true
    si, sj = DIRS[k]
    okj = sj == 0 || (sj > 0 ? j >= 2 : j <= r - 1)
    oki = si == 0 || (si > 0 ? i >= 2 : i <= q - 1)
    return okj && oki
end

# --------------------------------------------------------------------------- #
# Communication backends
# --------------------------------------------------------------------------- #

"""
    AbstractComm

How the ranks of an L-sweep talk.  [`LocalComm`](@ref) is one process (threads
only); the MPI extension provides `MPIComm`.  A backend implements `nranks`,
`myrank`, `post_recv!`, `post_send!`, `test_request!`, `wait_requests!` and
`allreduce_sum!`.
"""
abstract type AbstractComm end

"Single-process backend: every box is local, the worker tasks do all the work."
struct LocalComm <: AbstractComm end

nranks(::LocalComm) = 1
myrank(::LocalComm) = 0
post_recv!(::LocalComm, buf, src, tag) = error("LocalComm has no remote boxes")
post_send!(::LocalComm, buf, dst, tag) = error("LocalComm has no remote boxes")
test_request!(::LocalComm, req) = true
wait_requests!(::LocalComm, reqs) = nothing
allreduce_sum!(::LocalComm, u) = u

"""
    row_partition(q, r, p)

Owner rank (0-based) of every box: row `i` goes to rank `(i-1) mod p`, the
assignment of the paper's Fig. 18.  Rows stay on one rank, so the horizontal
sweeps never communicate; columns and diagonals pipeline across ranks.
"""
row_partition(q::Int, r::Int, p::Int) = [mod(i - 1, p) for i in 1:q, j in 1:r]

"Largest message tag the backend can carry."
max_tag(::AbstractComm) = typemax(Int)

"""
    MPIComm([comm])

Distributed backend over the ranks of an MPI communicator.  Available after
`using MPI` (the extension `SparsifyAndSweepMPIExt` defines the constructor);
`MPIComm()` uses `MPI.COMM_WORLD` and initialises MPI if needed.
"""
struct MPIComm{C} <: AbstractComm
    comm::C
    rank::Int
    size::Int
    lock::ReentrantLock
    tag_ub::Int
end

# --------------------------------------------------------------------------- #
# Geometry (replicated on every rank)
# --------------------------------------------------------------------------- #

"""
    cdd_ranges(r, k; shift = 0)

Cut the index range `r` into `k` contiguous pieces of (nearly) equal length.

With `shift > 0` (in points, less than a piece) the cuts are those of the
unshifted pieces moved up by `shift`, the one at `first(r)` included, so the
result has `k + 1` pieces: a part-piece of `shift` points at the front, `k - 1`
full pieces, and at the back the last unshifted piece less `shift` points.
With `shift` half a piece, both end pieces are half-pieces; e.g.
`cdd_ranges(1:256, 4; shift = 32)` is `[1:32, 33:96, 97:160, 161:224, 225:256]`.
That is the checkerboard of the second pass of a shifted L-sweep: its skeleton
runs through the middle of the first pass's boxes and meets the first skeleton
only at isolated crossings.
"""
function cdd_ranges(r::UnitRange{Int}, k::Integer; shift::Integer = 0)
    lo, hi = first(r), last(r)
    tot = hi - lo + 1
    k = clamp(Int(k), 1, tot)
    bnd = [lo + round(Int, tot * t / k) for t in 0:k]
    shift == 0 && return [bnd[t]:(bnd[t+1]-1) for t in 1:k]
    wmin = minimum(bnd[t+1] - bnd[t] for t in 1:k)
    0 < shift < wmin ||
        throw(ArgumentError("shift must lie strictly inside a piece: 0 < $shift < $wmin"))
    cuts = vcat(lo, bnd[1:k] .+ Int(shift), hi + 1)
    return [cuts[t]:(cuts[t+1]-1) for t in 1:(k+1)]
end

"""
Owned ranges, extended box, unstretched zone and neighbour flags of one box.  The
decomposition is always of axes 1 and 2; in three dimensions a box is a beam that
spans the whole grid along axis 3, global PML included, and has no neighbour there.
"""
struct BoxGeom{D}
    phys::NTuple{D,UnitRange{Int}}
    ext::NTuple{D,UnitRange{Int}}
    unstr::NTuple{D,UnitRange{Int}}
    n::Int
    hasnbr::NTuple{2,NTuple{2,Bool}}      # axes 1, 2: (neighbour below, neighbour above)
end

function box_geometry(g::Grid{D}, X, Y, i::Int, j::Int, q::Int, r::Int, pad::Int) where {D}
    lo, hi = ilo(g), ihi(g)
    phys = ntuple(d -> d == 1 ? X[j] : d == 2 ? Y[i] : (lo:hi), D)
    hasnbr = ((j > 1, j < r), (i > 1, i < q))
    ext = ntuple(d -> d > 2 ? (lo:hi) :
                      (hasnbr[d][1] ? first(phys[d]) - 2 - pad : lo):
                      (hasnbr[d][2] ? last(phys[d]) + 2 + pad : hi), D)
    unstr = ntuple(d -> d > 2 ? (lo:hi) :
                        max(first(phys[d]) - 2, lo):min(last(phys[d]) + 2, hi), D)
    for d in 1:2
        (first(ext[d]) >= lo && last(ext[d]) <= hi) ||
            throw(ArgumentError("box $phys extended by $pad leaves the grid"))
        ((hasnbr[d][1] && first(ext[d]) < 0) || (hasnbr[d][2] && last(ext[d]) > g.n + 1)) &&
            throw(ArgumentError("box $phys is too small: its pad reaches into the " *
                                "global PML; use fewer boxes or a smaller pad"))
    end
    return BoxGeom(phys, ext, unstr, boxlen(ext), hasnbr)
end

"Far-side predicate of the polarization step that receives from direction `(si, sj)`."
function far_predicate(phys::NTuple{D,UnitRange{Int}}, si::Int, sj::Int) where {D}
    x1lo, x1hi = first(phys[1]), last(phys[1])
    x2lo, x2hi = first(phys[2]), last(phys[2])
    return p -> (sj == 0 || (sj > 0 ? p[1] >= x1lo : p[1] <= x1hi)) &&
                (si == 0 || (si > 0 ? p[2] >= x2lo : p[2] <= x2hi))
end

"""
    trace_points(geom, si, sj)

The interface layers of the polarization step receiving from `(si, sj)`: `Γ₁`
(near side, adjacent to the far side) and `Γ₂` (far side, adjacent to the near
side), in global coordinates.  Purely geometric, so the source rank can compute
the receiver's point list without the receiver's matrix.
"""
function trace_points(geom::BoxGeom{D}, si::Int, sj::Int) where {D}
    far = far_predicate(geom.phys, si, sj)
    ext = geom.ext
    off = offsets(Val(D))
    inext(j) = all(d -> j[d] in ext[d], 1:D)
    Γ1 = NTuple{D,Int}[]
    Γ2 = NTuple{D,Int}[]
    for p in boxpoints(ext)
        fp = far(p)::Bool
        adj = false
        for t in off
            jj = p .+ t
            if inext(jj) && far(jj)::Bool != fp
                adj = true
                break
            end
        end
        adj || continue
        fp ? push!(Γ2, p) : push!(Γ1, p)
    end
    return Γ1, Γ2
end

"""
    Contribution

Part of the trace of one node: the values at `sel` (indices into the node's
`[Γ₁; Γ₂]` list) are the sum of components `comps` of box `src`, available once
the nodes `prereq` have fired on `src`'s rank.
"""
struct Contribution
    src::Tuple{Int,Int}
    comps::Vector{Int}
    sel::Vector{Int}
    prereq::Vector{NodeId}
end

"Contributions of node `(i, j, k)` given the receiver's trace point list."
function contributions(i::Int, j::Int, k::Int, q::Int, r::Int, geom::Matrix{BoxGeom{D}},
                       pts::Vector{NTuple{D,Int}}) where {D}
    si, sj = DIRS[k]
    c = k + 1
    exists(ii, jj, kk) = node_exists(ii, jj, kk, q, r)
    if si == 0 || sj == 0
        src = (i - si, j - sj)
        pre = NodeId[(src..., 0)]
        exists(src..., k) && push!(pre, (src..., k))
        return [Contribution(src, [C_OWN, c], collect(eachindex(pts)), pre)]
    end
    kh, kv = dirindex(0, sj), dirindex(si, 0)
    ch, cv = kh + 1, kv + 1
    phys = geom[i, j].phys
    x1lo, x1hi = first(phys[1]), last(phys[1])
    x2lo, x2hi = first(phys[2]), last(phys[2])
    corner = (sj > 0 ? x1lo - 1 : x1hi + 1, si > 0 ? x2lo - 1 : x2hi + 1)
    harm(p) = (si > 0 ? (p[2] == x2lo - 1 || p[2] == x2lo) : (p[2] == x2hi || p[2] == x2hi + 1)) &&
              (sj > 0 ? p[1] >= x1lo : p[1] <= x1hi)
    selc = Int[]; selh = Int[]; selv = Int[]
    for (n, p) in enumerate(pts)
        (p[1], p[2]) == corner ? push!(selc, n) : (harm(p) ? push!(selh, n) : push!(selv, n))
    end
    pre(src, ks) = NodeId[(src..., kk) for kk in ks if exists(src..., kk)]
    sd, sh, sv = (i - si, j - sj), (i - si, j), (i, j - sj)
    out = Contribution[]
    isempty(selc) || push!(out, Contribution(sd, [C_OWN, ch, cv, c], selc, pre(sd, (0, kh, kv, k))))
    isempty(selh) || push!(out, Contribution(sh, [ch, c], selh, pre(sh, (kh, k))))
    isempty(selv) || push!(out, Contribution(sv, [cv, c], selv, pre(sv, (kv, k))))
    return out
end

# --------------------------------------------------------------------------- #
# Owned boxes: local matrices, factors and coupling blocks
# --------------------------------------------------------------------------- #

"Cache of fitted PML rows keyed by (frequency sample, stretching increments)."
struct GammaCache{D}
    κsamp::Vector{Float64}
    lo2::Float64
    step2::Float64
    cache::Dict{Tuple{Int,NTuple{D,NTuple{3,Float64}}},Vector{ComplexF64}}
end
GammaCache{D}(κsamp, lo2, step2) where {D} =
    GammaCache{D}(κsamp, lo2, step2, Dict{Tuple{Int,NTuple{D,NTuple{3,Float64}}},Vector{ComplexF64}}())

function gamma_row!(gc::GammaCache{D}, h::Real, κ2::Real, dσ::NTuple{D,NTuple{3,Float64}}) where {D}
    k = (length(gc.κsamp) == 1 || gc.step2 <= 0) ? 1 :
        clamp(round(Int, (κ2 - gc.lo2) / gc.step2) + 1, 1, length(gc.κsamp))
    return get!(gc.cache, (k, dσ)) do
        pml_stencil(gc.κsamp[k], h, dσ)
    end
end

"A receive direction of an owned box: its trace layers and coupling blocks."
struct Trace{D}
    pts::Vector{NTuple{D,Int}}      # [Γ₁; Γ₂]
    n1::Int                         # length of Γ₁
    i1::Vector{Int}
    i2::Vector{Int}
    A12::SparseMatrixCSC{ComplexF64,Int}
    A21::SparseMatrixCSC{ComplexF64,Int}
    contrib::Vector{Contribution}
end

"""
    CompactLU

An UMFPACK factorisation re-stored for streaming: `L` and `U` as CSC in single
precision with 32-bit indices, 12 bytes per nonzero against UMFPACK's 24, with the
row scaling and permutations that reproduce its solve, `(Rs .* A)[p, q] == L * U`.
The unit diagonal of `L` is dropped; the diagonal of `U` is kept apart, inverted, in
double precision.  `solve!` runs the triangular solves in Julia, accumulates in
double precision and allocates nothing.  As a preconditioner the single-precision
factors cost nothing measurable: the sparsification error is four orders larger.
"""
struct CompactLU
    n::Int
    L::SparseMatrixCSC{ComplexF32,Int32}   # strictly lower part; unit diagonal implied
    U::SparseMatrixCSC{ComplexF32,Int32}   # strictly upper part
    uinv::Vector{ComplexF64}               # 1 ./ diag(U)
    p::Vector{Int32}
    q::Vector{Int32}
    Rs::Vector{Float64}
end

"Copy the off-diagonal entries of `A` into single precision with 32-bit indices."
function offdiag32(A::SparseMatrixCSC{ComplexF64,Int})
    n = size(A, 2)
    colptr = Vector{Int32}(undef, n + 1)
    colptr[1] = 1
    nz = 0
    @inbounds for j in 1:n
        for k in nzrange(A, j)
            A.rowval[k] != j && (nz += 1)
        end
        colptr[j+1] = nz + 1
    end
    rowval = Vector{Int32}(undef, nz)
    nzval = Vector{ComplexF32}(undef, nz)
    t = 0
    @inbounds for j in 1:n, k in nzrange(A, j)
        i = A.rowval[k]
        i == j && continue
        t += 1
        rowval[t] = i
        nzval[t] = A.nzval[k]
    end
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function CompactLU(F::SparseArrays.UMFPACK.UmfpackLU{ComplexF64,Int})
    # one umfpack_get_numeric call for all five parts; the property accessors
    # (F.L, F.U, ...) each extract everything again
    L, U, p, q, Rs = SparseArrays.UMFPACK.umfpack_extract(F)
    n = size(U, 2)
    uinv = Vector{ComplexF64}(undef, n)
    @inbounds for j in 1:n
        d = zero(ComplexF64)
        for k in nzrange(U, j)
            U.rowval[k] == j && (d = U.nzval[k])
        end
        uinv[j] = inv(d)
    end
    C = CompactLU(n, offdiag32(L), offdiag32(U), uinv, Int32.(p), Int32.(q), Rs)
    # release UMFPACK's own copy now rather than at the next collection, exactly as
    # the stdlib finalizer would: free the handle, then null it so the finalizer sees
    # nothing left to do
    SparseArrays.UMFPACK.umfpack_free_numeric(F.numeric, ComplexF64, Int)
    F.numeric.p = C_NULL
    SparseArrays.UMFPACK.umfpack_free_symbolic(F.symbolic, ComplexF64, Int)
    F.symbolic.p = C_NULL
    return C
end

"""
    solve!(x, F::CompactLU, b)

`x = A \\ b` from the compact factors.  `b` is used as scratch and destroyed.
Zero entries of the running vector are skipped in both triangular solves, which
is where a trace right-hand side -- nonzero on two grid layers -- saves most of
the forward pass; the skip does not change a single bit of the result.
"""
function solve!(x::Vector{ComplexF64}, F::CompactLU, b::Vector{ComplexF64})
    n = F.n
    p, q, Rs = F.p, F.q, F.Rs
    @inbounds for i in 1:n                              # x = (Rs .* b)[p]
        pi_ = p[i]
        x[i] = Rs[pi_] * b[pi_]
    end
    Lc, Lr, Lv = F.L.colptr, F.L.rowval, F.L.nzval
    @inbounds for j in 1:n                              # L x = x, unit lower
        xj = x[j]
        iszero(xj) && continue
        for k in Lc[j]:(Lc[j+1] - 1)
            x[Lr[k]] -= Lv[k] * xj
        end
    end
    Uc, Ur, Uv, uinv = F.U.colptr, F.U.rowval, F.U.nzval, F.uinv
    @inbounds for j in n:-1:1                           # U x = x
        xj = x[j] * uinv[j]
        x[j] = xj
        iszero(xj) && continue
        for k in Uc[j]:(Uc[j+1] - 1)
            x[Ur[k]] -= Uv[k] * xj
        end
    end
    @inbounds for i in 1:n                              # undo the column permutation
        b[q[i]] = x[i]
    end
    copyto!(x, b)
    return x
end
Base.:\(F::CompactLU, b::AbstractVector{ComplexF64}) =
    solve!(Vector{ComplexF64}(undef, F.n), F, Vector{ComplexF64}(b))

const UmfLU = SparseArrays.UMFPACK.UmfpackLU{ComplexF64,Int}

"An owned box: geometry, factors, owned-point index maps and its traces by direction."
struct LBox{D,Fa}
    geom::BoxGeom{D}
    fact::Fa                        # UmfLU, or CompactLU
    pidx::Vector{Int}
    gidx::Vector{Int}
    traces::Dict{Int,Trace{D}}      # by direction index k = 1..8
end

"""
    assemble_local(g, ω, m, astar, bstar, C, gc, geom)

The local matrix of one box from the stencils alone: `α`/`β` rows on the
unstretched zone (global PML rows there near the outer boundary), fitted `γ`
rows at the local frequency on the pad.
"""
function assemble_local(g::Grid{D}, ω::Float64, m::Array{Float64,D},
                        astar::Vector{ComplexF64}, bstar::Vector{ComplexF64},
                        C::Float64, gc::GammaCache{D}, geom::BoxGeom{D}, pad::Int) where {D}
    h = g.h
    ω2 = ω^2
    phys, ext, unstr = geom.phys, geom.ext, geom.unstr
    amp = C / ω
    prof = ntuple(d -> PMLProfile(amp, pad * h,
                                  geom.hasnbr[d][1] ? (first(phys[d]) - 3) * h : -Inf,
                                  geom.hasnbr[d][2] ? (last(phys[d]) + 3) * h : Inf), 2)
    gprof = global_profile(g, ω, C)
    dσ(d, c) = c in unstr[d] ? dsigma(gprof, h, c) : dsigma(prof[d], h, c)
    off = offsets(Val(D))
    np = length(off)
    n = geom.n
    inext(j) = all(d -> j[d] in ext[d], 1:D)

    # coef[q, i] is row i's coefficient at its neighbour i + off[q]; present[q, i]
    # marks entries the row carries.  An α/β row drops exact zeros, a γ row keeps
    # every neighbour inside the box, as the stencils define them.
    coef = zeros(ComplexF64, np, n)
    present = falses(np, n)
    for p in boxpoints(ext)
        row = boxlin(ext, p)
        dσs = ntuple(d -> dσ(d, p[d]), D)          # axis 3 only ever sees the global PML
        if all(==(DSIGMA0), dσs)
            for q in 1:np
                j = p .+ off[q]
                inext(j) || continue
                v = astar[q] + ω2 * bstar[q] * mval(g, m, j)
                iszero(v) && continue
                coef[q, row] = v
                present[q, row] = true
            end
        else
            γ = gamma_row!(gc, h, ω2 * (1 - mval(g, m, p)), dσs)
            for q in 1:np
                j = p .+ off[q]
                inext(j) || continue
                coef[q, row] = γ[q]
                present[q, row] = true
            end
        end
    end

    # Straight into CSC: column j holds row i = j - off[q] with value coef[q, i].
    # boxlin is affine with strides (1, n₁, n₁n₂, ...), so row - column is minus the
    # offset's shift; visiting the offsets in decreasing order of that shift makes
    # every column's rows increase.
    strides = cumprod((1, length.(ext)[1:end-1]...))
    order = sortperm([sum(t[d] * strides[d] for d in 1:D) for t in off]; rev = true)
    counts = zeros(Int, n)
    for pj in boxpoints(ext)
        col = boxlin(ext, pj)
        for q in order
            pi_ = pj .- off[q]
            inext(pi_) && present[q, boxlin(ext, pi_)] && (counts[col] += 1)
        end
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    for j in 1:n
        colptr[j+1] = colptr[j] + counts[j]
    end
    rowval = Vector{Int}(undef, colptr[end] - 1)
    nzval = Vector{ComplexF64}(undef, colptr[end] - 1)
    for pj in boxpoints(ext)
        col = boxlin(ext, pj)
        t = colptr[col]
        for q in order
            pi_ = pj .- off[q]
            inext(pi_) || continue
            row = boxlin(ext, pi_)
            present[q, row] || continue
            rowval[t] = row
            nzval[t] = coef[q, row]
            t += 1
        end
    end
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function build_lbox(g::Grid{D}, ω::Float64, m::Array{Float64,D}, astar, bstar, C::Float64,
                    gc::GammaCache{D}, geom::Matrix{BoxGeom{D}}, i::Int, j::Int, q::Int, r::Int,
                    pad::Int, factors::Symbol) where {D}
    G = geom[i, j]
    A = assemble_local(g, ω, m, astar, bstar, C, gc, G, pad)
    traces = Dict{Int,Trace{D}}()
    for k in 1:length(DIRS)
        node_exists(i, j, k, q, r) || continue
        Γ1, Γ2 = trace_points(G, DIRS[k]...)
        pts = vcat(Γ1, Γ2)
        i1 = [boxlin(G.ext, p) for p in Γ1]
        i2 = [boxlin(G.ext, p) for p in Γ2]
        traces[k] = Trace(pts, length(Γ1), i1, i2, A[i1, i2], A[i2, i1],
                          contributions(i, j, k, q, r, geom, pts))
    end
    pidx = boxidx(G.ext, G.phys)
    gidx = vec([lin(g, p) for p in boxpoints(G.phys)])
    F = lu(A)
    return LBox(G, factors === :compact ? CompactLU(F) : F, pidx, gidx, traces)
end

# --------------------------------------------------------------------------- #
# The factorization
# --------------------------------------------------------------------------- #

"""
    OutJob

A contribution this rank extracts from one of its boxes: the receiver's trace
points (global coordinates), the target node and its rank, and the message tag.
"""
struct OutJob{D}
    contrib::Contribution
    pts::Vector{NTuple{D,Int}}
    target::NodeId
    rank::Int
    tag::Int
end

"""
    LSweepFactorization

Approximate inverse of `H` by L-sweeps over a `q x r` CDD, distributed over the
ranks of `comm` by `owner`.  Apply with [`lsweep_solve`](@ref).
"""
struct LSweepFactorization{D,Cm<:AbstractComm,Fa}
    g::Grid{D}
    q::Int
    r::Int
    pad::Int
    geom::Matrix{BoxGeom{D}}
    owner::Matrix{Int}
    comm::Cm
    boxes::Dict{Tuple{Int,Int},LBox{D,Fa}}           # owned boxes
    nodes::Vector{NodeId}                             # owned nodes, all directions
    outjobs::Vector{OutJob{D}}                           # contributions I extract
    jobs_by_prereq::Dict{NodeId,Vector{Int}}          # node -> indices into outjobs
    nworkers::Int
    nstencil::Int
end

nboxes(F::LSweepFactorization) = F.q * F.r
nowned(F::LSweepFactorization) = length(F.boxes)
nstencils(F::LSweepFactorization) = F.nstencil
"Number of owned boxes, and the size of the largest."
leafstats(F::LSweepFactorization) =
    (nowned(F), maximum(b -> b.geom.n, values(F.boxes); init = 0))
"One L-sweep, as a pass of an [`AlternatingSweep`](@ref)."
approx_solve(F::LSweepFactorization, f::AbstractVector{ComplexF64}) = lsweep_solve(F, f)

"Run `f` over `items` on separate tasks when more than one worker is available."
function foreach_task(f, items; nworkers::Int, maxactive::Int = nworkers)
    if nworkers > 1 && Threads.nthreads() > 1
        sem = Base.Semaphore(max(1, maxactive))   # at most `maxactive` items in flight
        @sync for it in items
            Threads.@spawn Base.acquire(sem) do
                f(it)
            end
        end
    else
        foreach(f, items)
    end
    return nothing
end

"Message tag of contribution `m` of node `(i, j, k)`; unique per (target node, contribution)."
message_tag(i::Int, j::Int, k::Int, m::Int, q::Int, r::Int) =
    (((k - 1) * q + (i - 1)) * r + (j - 1)) * 3 + (m - 1)

"""
    lsweep_setup(g, ω, m, astar, bstar; C = 12.0, nsamp = 200, nbox = 64,
                 boxes = nothing, pad = g.b, shift = (0, 0), comm = LocalComm(),
                 owner = row_partition(q, r, nranks(comm)),
                 nworkers = Threads.nthreads())

Build the L-sweep approximate inverse of the sparsified system on a CDD with
boxes of about `nbox` points per side (or exactly `boxes = (q, r)` of them),
each padded by `pad` layers of fitted PML.  `shift = (s₁, s₂)`, fractions of a
box in `[0, 1)`, moves every cut of the checkerboard up by that much on each
axis and adds a cut, so a shifted axis carries one box more, with part-boxes at both ends;
`(0.5, 0.5)` is the board of the second pass of a shifted alternation (see
[`AlternatingSweep`](@ref)).  The interior boxes must still keep their pads
out of the global PML, so a half-box must exceed `pad + 2` points.  Every rank assembles and factorises
only the boxes it owns; the geometry and the dependency graph are replicated.
`nworkers` tasks execute the nodes of the sweep concurrently within a rank;
`nfact` (default `nworkers`) caps the factorisations in flight during setup, each of
which holds several times its final size in transients.

`factors = :compact` (the default) re-stores each box's LU as single-precision CSC
with 32-bit indices ([`CompactLU`](@ref)), half UMFPACK's bytes, which is what the
application streams; `:umfpack` keeps UMFPACK's own factors and solves through it.
"""
function lsweep_setup(g::Grid{D}, ω::Real, m::AbstractArray{Float64,D},
                      astar::Vector{ComplexF64}, bstar::Vector{ComplexF64};
                      C::Real = 12.0, nsamp::Integer = 200, nbox::Integer = 64,
                      boxes::Union{Nothing,Tuple{Int,Int}} = nothing,
                      pad::Integer = g.b, shift::NTuple{2,Real} = (0, 0),
                      comm::AbstractComm = LocalComm(),
                      owner::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
                      nworkers::Integer = Threads.nthreads(),
                      nfact::Integer = nworkers,
                      factors::Symbol = :compact) where {D}
    factors in (:umfpack, :compact) ||
        throw(ArgumentError("factors must be :umfpack or :compact"))
    D in (2, 3) || throw(ArgumentError("L-sweeps are implemented in two and three dimensions"))
    if boxes === nothing
        k = max(1, round(Int, ntot(g) / nbox))
        boxes = (k, k)
    end
    q, r = boxes
    q >= 1 && r >= 1 || throw(ArgumentError("boxes must be positive"))
    all(s -> 0 <= s < 1, shift) ||
        throw(ArgumentError("shift is a fraction of a box per axis, in [0, 1); got $shift"))
    function shiftpts(frac, k)
        s = round(Int, frac * ntot(g) / k)
        frac > 0 && s == 0 && throw(ArgumentError("boxes too small to shift by $frac of a box"))
        return s
    end
    X = cdd_ranges(ilo(g):ihi(g), r; shift = shiftpts(shift[1], r))   # columns, axis 1
    Y = cdd_ranges(ilo(g):ihi(g), q; shift = shiftpts(shift[2], q))   # rows, axis 2
    q, r = length(Y), length(X)                       # one more per shifted axis
    pad = Int(pad)
    p = nranks(comm)
    own = owner === nothing ? row_partition(q, r, p) : Matrix{Int}(owner)
    size(own) == (q, r) || throw(ArgumentError("owner must be $q x $r"))
    all(0 .<= own .< p) || throw(ArgumentError("owner ranks must lie in 0:$(p-1)"))
    me = myrank(comm)
    message_tag(q, r, length(DIRS), 3, q, r) <= max_tag(comm) ||
        throw(ArgumentError("too many boxes for the message tags of this MPI library"))

    geom = [box_geometry(g, X, Y, i, j, q, r, pad) for i in 1:q, j in 1:r]

    mmin, mmax = extrema(m)
    lo2, hi2 = ω^2 * (1 - mmax), ω^2 * (1 - mmin)
    ns = max(1, Int(nsamp))
    dκ2 = ns > 1 ? (hi2 - lo2) / (ns - 1) : 1.0
    κsamp = [sqrt(lo2 + (k - 1) * dκ2) for k in 1:ns]
    mm = Array{Float64,D}(m)

    mine = [(i, j) for j in 1:r for i in 1:q if own[i, j] == me]
    Fa = factors === :compact ? CompactLU : UmfLU
    built = Vector{LBox{D,Fa}}(undef, length(mine))
    nst = Atomic{Int}(0)
    # The boxes are the parallelism.  BLAS threads inside each UMFPACK call would
    # only spin at their barriers -- measured at 92 CPU-seconds of a 150-second
    # setup -- so they are pinned to one for the duration.  Unconditionally: the
    # BLAS thread count changes the factors in their last bits, and the result is
    # meant to agree bitwise across worker and rank counts.
    nblas = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        # A factorisation in flight holds several times its final size in UMFPACK
        # transients -- about four times the compact factors -- so `nfact` caps how
        # many run at once; it matters in three dimensions, where the beams are large.
        foreach_task(eachindex(mine); nworkers = Int(nworkers), maxactive = Int(nfact)) do t
            gc = GammaCache{D}(κsamp, lo2, dκ2)       # per box: no shared mutable state
            i, j = mine[t]
            built[t] = build_lbox(g, Float64(ω), mm, astar, bstar, Float64(C), gc, geom,
                                  i, j, q, r, pad, factors)
            atomic_add!(nst, length(gc.cache))
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(nblas)
    end
    boxesd = Dict{Tuple{Int,Int},LBox{D,Fa}}(mine[t] => built[t] for t in eachindex(mine))

    # owned nodes, and the contributions I must extract (for others or for myself)
    nodes = NodeId[]
    for (i, j) in mine, k in 0:length(DIRS)
        node_exists(i, j, k, q, r) && push!(nodes, (i, j, k))
    end
    outjobs = OutJob{D}[]
    jobs_by_prereq = Dict{NodeId,Vector{Int}}()
    for j in 1:r, i in 1:q, k in 1:length(DIRS)
        node_exists(i, j, k, q, r) || continue
        Γ1, Γ2 = trace_points(geom[i, j], DIRS[k]...)    # geometric: no matrix needed
        pts = vcat(Γ1, Γ2)
        for (mth, cb) in enumerate(contributions(i, j, k, q, r, geom, pts))
            own[cb.src...] == me || continue
            push!(outjobs, OutJob(cb, pts[cb.sel], (i, j, k), own[i, j],
                                  message_tag(i, j, k, mth, q, r)))
            for pre in cb.prereq
                push!(get!(jobs_by_prereq, pre, Int[]), length(outjobs))
            end
        end
    end
    return LSweepFactorization(g, q, r, pad, geom, own, comm, boxesd, nodes, outjobs,
                               jobs_by_prereq, max(1, Int(nworkers)), nst[])
end

# --------------------------------------------------------------------------- #
# Application: the dataflow
# --------------------------------------------------------------------------- #

"Per-application state of the dataflow on this rank."
struct SolveState
    f::Vector{ComplexF64}
    W::Dict{Tuple{Int,Int},Vector{Vector{ComplexF64}}}   # components of owned boxes
    inbox::Dict{NodeId,Vector{ComplexF64}}               # assembled trace values
    pending::Dict{NodeId,Atomic{Int}}                     # contributions still missing
    remaining::Vector{Atomic{Int}}                        # prerequisites left, per outjob
    queue::Channel{NodeId}
    ndone::Atomic{Int}
    ntotal::Int
    done::Threads.Condition
    recvs::Vector{Tuple{Any,NodeId,Vector{Int},Vector{ComplexF64}}}   # (request, node, sel, buffer)
    sends::Vector{Tuple{Any,Vector{ComplexF64}}}
    sendlock::ReentrantLock
end

"Trace values of an outgoing job, summed from the owned source box's components."
function extract(F::LSweepFactorization, S::SolveState, job::OutJob)
    src = F.boxes[job.contrib.src]
    Wsrc = S.W[job.contrib.src]
    out = Vector{ComplexF64}(undef, length(job.pts))
    @inbounds for (n, p) in enumerate(job.pts)
        k = boxlin(src.geom.ext, p)
        acc = zero(ComplexF64)
        for c in job.contrib.comps
            acc += Wsrc[c][k]
        end
        out[n] = acc
    end
    return out
end

"Place trace values in a node's inbox; enqueue the node once all its contributions are in."
function deliver!(S::SolveState, node::NodeId, sel::Vector{Int}, vals::AbstractVector{ComplexF64})
    buf = S.inbox[node]
    @inbounds for (n, s) in enumerate(sel)
        buf[s] = vals[n]
    end
    if atomic_sub!(S.pending[node], 1) == 1
        put!(S.queue, node)
    end
    return nothing
end

"Extract an outgoing contribution and hand it over, locally or by message."
function dispatch_job!(F::LSweepFactorization, S::SolveState, jb::Int)
    job = F.outjobs[jb]
    vals = extract(F, S, job)
    if job.rank == myrank(F.comm)
        deliver!(S, job.target, job.contrib.sel, vals)
    else
        req = post_send!(F.comm, vals, job.rank, job.tag)
        lock(S.sendlock) do
            push!(S.sends, (req, vals))
        end
    end
    return nothing
end

"Put a box's component `c` into `W[c]`: UMFPACK allocates its result, the compact solve writes in place."
solve_into!(W::Vector{Vector{ComplexF64}}, c::Int, fact::UmfLU, rhs::Vector{ComplexF64}) =
    (W[c] = fact \ rhs; nothing)
solve_into!(W::Vector{Vector{ComplexF64}}, c::Int, fact::CompactLU, rhs::Vector{ComplexF64}) =
    (solve!(W[c], fact, rhs); nothing)

"Fire node `(i, j, k)`: the box's own solve (`k = 0`) or a polarized solve; then release dependants."
function run_node!(F::LSweepFactorization, S::SolveState, node::NodeId)
    i, j, k = node
    b = F.boxes[(i, j)]
    if k == 0
        fl = zeros(ComplexF64, b.geom.n)
        @inbounds fl[b.pidx] = S.f[b.gidx]
        solve_into!(S.W[(i, j)], C_OWN, b.fact, fl)
    else
        tr = b.traces[k]
        v = S.inbox[node]
        v1 = view(v, 1:tr.n1)
        v2 = view(v, (tr.n1+1):length(v))
        rhs = zeros(ComplexF64, b.geom.n)
        rhs[tr.i1] = tr.A12 * v2
        rhs[tr.i2] = -(tr.A21 * v1)
        solve_into!(S.W[(i, j)], k + 1, b.fact, rhs)
    end
    for jb in get(F.jobs_by_prereq, node, Int[])
        atomic_sub!(S.remaining[jb], 1) == 1 && dispatch_job!(F, S, jb)
    end
    atomic_add!(S.ndone, 1)
    lock(S.done) do
        notify(S.done)
    end
    return nothing
end

function worker_loop!(F::LSweepFactorization, S::SolveState)
    for node in S.queue
        run_node!(F, S, node)
    end
    return nothing
end

"Wait for all owned nodes to fire, polling remote receives if there are any."
function progress_loop!(F::LSweepFactorization, S::SolveState)
    if isempty(S.recvs)
        lock(S.done) do
            while S.ndone[] < S.ntotal
                wait(S.done)
            end
        end
    else
        active = trues(length(S.recvs))
        while S.ndone[] < S.ntotal
            progressed = false
            for n in eachindex(S.recvs)
                active[n] || continue
                req, node, sel, buf = S.recvs[n]
                if test_request!(F.comm, req)
                    active[n] = false
                    deliver!(S, node, sel, buf)
                    progressed = true
                end
            end
            progressed || yield()
        end
    end
    return nothing
end

"""
    lsweep_solve(F, f)

Approximate solve `H ũ ≈ f` by L-sweeps.  `f` must be the full right-hand side
on every rank; the result is the full approximate solution on every rank.  All
nodes of the sweep run as one dataflow: a box's own solve first, then the
cardinal and diagonal receptions as their traces arrive.
"""
function lsweep_solve(F::LSweepFactorization, f::AbstractVector{ComplexF64})
    q, r = F.q, F.r
    me = myrank(F.comm)
    W = Dict{Tuple{Int,Int},Vector{Vector{ComplexF64}}}()
    for (ij, b) in F.boxes
        W[ij] = [zeros(ComplexF64, b.geom.n) for _ in 1:NCOMP]
    end
    inbox = Dict{NodeId,Vector{ComplexF64}}()
    pending = Dict{NodeId,Atomic{Int}}()
    for node in F.nodes
        node[3] == 0 && continue
        tr = F.boxes[(node[1], node[2])].traces[node[3]]
        inbox[node] = zeros(ComplexF64, length(tr.pts))
        pending[node] = Atomic{Int}(length(tr.contrib))
    end
    S = SolveState(Vector{ComplexF64}(f), W, inbox, pending,
                   [Atomic{Int}(length(job.contrib.prereq)) for job in F.outjobs],
                   Channel{NodeId}(Inf), Atomic{Int}(0), length(F.nodes),
                   Threads.Condition(), Tuple{Any,NodeId,Vector{Int},Vector{ComplexF64}}[],
                   Tuple{Any,Vector{ComplexF64}}[], ReentrantLock())

    # receives for contributions coming from other ranks
    for node in F.nodes
        node[3] == 0 && continue
        tr = F.boxes[(node[1], node[2])].traces[node[3]]
        for (mth, cb) in enumerate(tr.contrib)
            src_rank = F.owner[cb.src...]
            src_rank == me && continue
            buf = Vector{ComplexF64}(undef, length(cb.sel))
            req = post_recv!(F.comm, buf, src_rank, message_tag(node..., mth, q, r))
            push!(S.recvs, (req, node, cb.sel, buf))
        end
    end
    # jobs with nothing to wait for (all-zero components) go out immediately
    for jb in eachindex(F.outjobs)
        S.remaining[jb][] == 0 && dispatch_job!(F, S, jb)
    end
    # seed the dataflow with the boxes' own solves
    for node in F.nodes
        node[3] == 0 && put!(S.queue, node)
    end
    if S.ntotal > 0
        @sync begin
            for _ in 1:F.nworkers
                Threads.@spawn worker_loop!(F, S)
            end
            progress_loop!(F, S)
            close(S.queue)
        end
    end
    wait_requests!(F.comm, Any[s[1] for s in S.sends])

    u = zeros(ComplexF64, nunk(F.g))
    for (ij, b) in F.boxes
        Wb = S.W[ij]
        @inbounds for (n, gi) in enumerate(b.gidx)
            li = b.pidx[n]
            acc = zero(ComplexF64)
            for c in 1:NCOMP
                acc += Wb[c][li]
            end
            u[gi] = acc
        end
    end
    allreduce_sum!(F.comm, u)
    return u
end
