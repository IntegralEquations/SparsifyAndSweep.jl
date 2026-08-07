# Grid bookkeeping for the extended domain Ω^{h+η}, in any dimension.

"""
    Grid{D}(n, b)

Uniform Cartesian grid used throughout the solver, in `D = 2` or `3` dimensions.
Following Liu & Ying we take `Ω = (0,1)^D`, discretised with `n` interior points
per dimension and step size `h = 1/(n+1)`; grid points are `p_i = i*h`.

Three nested index sets are used:

| set                        | definition                | role                                   |
|:---------------------------|:--------------------------|:---------------------------------------|
| `I   = {1,…,n}^D`          | `Ω`                       | physical unknowns, `N = n^D`           |
| `Ih  = {0,…,n+1}^D`        | `Ω^h`  (`h`-extension)    | rows carrying the `α`/`β` stencils     |
| `Ihη = {-b,…,n+1+b}^D`     | `Ω^{h+η}`, `η = b*h`      | rows carrying the PML `γ` stencils     |

The unknown vector of the sparsified system lives on `Ihη`. Zero Dirichlet data
on the ring just outside `Ihη` is realized by dropping stencil entries that leave
the index range. Unknowns are numbered with `x₁` fastest.
"""
struct Grid{D}
    n::Int
    b::Int
    h::Float64
end
Grid{D}(n::Integer, b::Integer) where {D} = Grid{D}(Int(n), Int(b), 1 / (n + 1))
Grid(n::Integer, b::Integer) = Grid{2}(n, b)

dim(::Grid{D}) where {D} = D
ntot(g::Grid) = g.n + 2 + 2 * g.b
ilo(g::Grid) = -g.b
ihi(g::Grid) = g.n + 1 + g.b
nunk(g::Grid{D}) where {D} = ntot(g)^D
nphys(g::Grid{D}) where {D} = g.n^D

inrange(g::Grid, i::Integer) = ilo(g) <= i <= ihi(g)
xcoord(g::Grid, i::Integer) = i * g.h

"Linear index on `Ihη` of the point with integer coordinates `i`, `x₁` fastest."
@inline function lin(g::Grid{D}, i::NTuple{D,Int}) where {D}
    nt = ntot(g)
    idx = 0
    @inbounds for d in D:-1:1
        idx = idx * nt + (i[d] + g.b)
    end
    return idx + 1
end
lin(g::Grid{2}, i1::Integer, i2::Integer) = lin(g, (Int(i1), Int(i2)))
lin(g::Grid{3}, i1::Integer, i2::Integer, i3::Integer) = lin(g, (Int(i1), Int(i2), Int(i3)))

"Linear index on `I = {1,…,n}^D`, `x₁` fastest."
@inline function plin(g::Grid{D}, i::NTuple{D,Int}) where {D}
    idx = 0
    @inbounds for d in D:-1:1
        idx = idx * g.n + (i[d] - 1)
    end
    return idx + 1
end

@inline in_I(g::Grid{D}, i::NTuple{D,Int}) where {D} = all(d -> 1 <= i[d] <= g.n, 1:D)
@inline in_Ih(g::Grid{D}, i::NTuple{D,Int}) where {D} = all(d -> 0 <= i[d] <= g.n + 1, 1:D)
@inline inbox(g::Grid{D}, i::NTuple{D,Int}) where {D} = all(d -> inrange(g, i[d]), 1:D)
in_I(g::Grid{2}, i1::Integer, i2::Integer) = in_I(g, (Int(i1), Int(i2)))
in_Ih(g::Grid{2}, i1::Integer, i2::Integer) = in_Ih(g, (Int(i1), Int(i2)))

"All integer coordinates of `Ihη`, as an iterator of `NTuple{D,Int}`."
boxindices(g::Grid{D}) where {D} =
    (Tuple(c) for c in CartesianIndices(ntuple(_ -> ilo(g):ihi(g), D)))
"All integer coordinates of `I`."
physindices(g::Grid{D}) where {D} =
    (Tuple(c) for c in CartesianIndices(ntuple(_ -> 1:g.n, D)))

# --------------------------------------------------------------------------- #
# The 3^D neighbourhood μ = {t : ‖t‖_∞ ≤ 1}, ordered with t₁ fastest.
# --------------------------------------------------------------------------- #

_make_offsets(D::Int) =
    vec(NTuple{D,Int}[ntuple(d -> c[d] - 2, D) for c in CartesianIndices(ntuple(_ -> 1:3, D))])

const OFFSETS2 = _make_offsets(2)
const OFFSETS3 = _make_offsets(3)
offsets(::Val{2}) = OFFSETS2
offsets(::Val{3}) = OFFSETS3
offsets(g::Grid{D}) where {D} = offsets(Val(D))

"Backwards-compatible name for the 2D neighbourhood."
const OFFSETS = OFFSETS2
const ICENTER = 5   # index of (0,0) in OFFSETS2

"Number of points in μ: 9 in 2D, 27 in 3D."
nnbr(::Val{D}) where {D} = 3^D

"Linear indices on `Ihη` of the physical unknowns `I`, in column-major order."
phys_indices(g::Grid{D}) where {D} =
    vec([lin(g, i) for i in physindices(g)])
