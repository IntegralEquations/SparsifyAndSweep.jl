# The two families of compact 3^D stencils of Liu & Ying, §2.2 (2D) and §3.1 (3D).
#
#  * interior stencils (α, β): annihilate the far part of the Green's function,
#    turning the dense row  u_i + ω² Σ_j K_{ij} m_j u_j = g_i  into a 3^D-point row.
#  * PML stencils (γ): annihilate complex-stretched ("modified") plane waves.
#
# Throughout we store the conjugated stencils `astar = α*`, `bstar = β*`,
# `gstar = γ*`, i.e. exactly the coefficients that appear in a matrix row.
# Everything here is dimension-generic; only the sizes change (9 and 8 in 2D,
# 27 and 26 in 3D).

"""
    _annihilate(K, offsets, ranges, skip; chunk)

Smallest right singular vector of the tall matrix `W[j,q] = k_{offsets[q] - j}`,
with `j` ranging over the box `ranges` minus the points where `skip(j)` is true.

`W` is never formed: it is reduced by chunked Householder QR to a `pxp`
triangular factor with the same singular values.  Forming the `pxp` Gram matrix
instead would square the condition number and destroy the answer, since the
whole point is that `σ_min` is tiny.  Returns the unit vector and `σ_min/σ_max`.
"""
function _annihilate(K::Kernel{D}, offsets::Vector{NTuple{D,Int}},
                     ranges::NTuple{D,UnitRange{Int}}, skip;
                     chunk::Integer = 8192) where {D}
    p = length(offsets)
    Rfac = zeros(ComplexF64, 0, p)
    buf = Matrix{ComplexF64}(undef, chunk, p)
    cnt = 0
    @inbounds for cid in CartesianIndices(ranges)
        j = Tuple(cid)
        skip(j) && continue
        cnt += 1
        for q in 1:p
            buf[cnt, q] = kval(K, offsets[q] .- j)
        end
        if cnt == chunk
            Rfac = _qrR(vcat(Rfac, buf))
            cnt = 0
        end
    end
    cnt > 0 && (Rfac = _qrR(vcat(Rfac, @view buf[1:cnt, :])))
    F = svd(Rfac)
    return ComplexF64.(F.V[:, end]), F.S[end] / F.S[1]
end

_qrR(A::AbstractMatrix) = Matrix(qr(A).R)

"""
    sparsify_stencils(K, n; farfield = n, chunk = 8192)

Solve `min_{‖α‖=1} ‖α* K_{μ,μᶜ}‖₂` with `μ = {t : ‖t‖_∞ ≤ 1}` and
`μᶜ = {-R,…,R}^D \\ μ`, then set `β* = α* K_{μ,μ}` (eqs. 11-12).

Returns `(astar, bstar, ratio)`, both `3^D`-vectors ordered by `offsets(Val(D))`,
and `ratio = σ_min/σ_max`.

`farfield` is the half-width `R` of `μᶜ`.  The paper takes `R = n`, which costs
`O(n^D)` work; `R` well below `n` gives the same stencil to many digits, because
Green's functions centred far away are nearly linearly dependent on `μ`.  That
matters in 3D, where `(2n+1)³` is already `10^8` at `n = 255`.  Use
[`farfield_residual`](@ref) to check a truncation.
"""
function sparsify_stencils(K::Kernel{D}, n::Integer; farfield::Integer = n,
                           chunk::Integer = 8192) where {D}
    R = min(Int(farfield), Int(n))
    off = offsets(Val(D))
    astar, ratio = _annihilate(K, off, ntuple(_ -> (-R):R, D),
                               j -> all(t -> abs(t) <= 1, j); chunk = chunk)
    # β* = α* K_{μ,μ}:  bstar[t] = Σ_s astar[s] k_{s-t}
    p = length(off)
    bstar = zeros(ComplexF64, p)
    @inbounds for q in 1:p
        acc = zero(ComplexF64)
        for s in 1:p
            acc += astar[s] * kval(K, off[s] .- off[q])
        end
        bstar[q] = acc
    end
    return astar, bstar, ratio
end

# legacy 2D entry point
sparsify_stencils(n::Integer, ω::Real, h::Real, k0::Complex; chunk::Integer = 8192,
                  table = nothing, farfield::Integer = n) =
    sparsify_stencils(Kernel(Val(2), ω, h, k0, n + 1), n; farfield = farfield, chunk = chunk)

"""
    farfield_residual(K, astar, n; skipradius)

`‖α* K_{μ,J}‖ / ‖K_{μ,J}‖_F` over the *full* far field `J = {-n,…,n}^D \\ μ`.
Use it to confirm that a stencil computed with a truncated `farfield` still
annihilates the sources it was not shown.
"""
function farfield_residual(K::Kernel{D}, astar::Vector{ComplexF64}, n::Integer;
                           skipradius::Integer = 1) where {D}
    off = offsets(Val(D))
    num = 0.0
    den = 0.0
    @inbounds for cid in CartesianIndices(ntuple(_ -> (-Int(n)):Int(n), D))
        j = Tuple(cid)
        all(t -> abs(t) <= skipradius, j) && continue
        acc = zero(ComplexF64)
        for q in eachindex(off)
            v = kval(K, off[q] .- j)
            acc += astar[q] * v
            den += abs2(v)
        end
        num += abs2(acc)
    end
    return sqrt(num / den)
end

# --------------------------------------------------------------------------- #
# PML stencils from modified plane waves
# --------------------------------------------------------------------------- #

"""
    pmldirs(Val(D))

The `3^D - 1` directions from the centre of a neighbourhood to its boundary
neighbours, normalised: 8 in 2D (N, S, E, W and the diagonals), 26 in 3D.
"""
function _make_dirs(D::Int)
    d = NTuple{D,Float64}[]
    for c in CartesianIndices(ntuple(_ -> 1:3, D))
        r = ntuple(k -> float(c[k] - 2), D)
        all(iszero, r) && continue
        nrm = sqrt(sum(abs2, r))
        push!(d, ntuple(k -> r[k] / nrm, D))
    end
    return d
end
const PMLDIRS2 = _make_dirs(2)
const PMLDIRS3 = _make_dirs(3)
pmldirs(::Val{2}) = PMLDIRS2
pmldirs(::Val{3}) = PMLDIRS3
const PMLDIRS = PMLDIRS2

"""
    pml_stencil(κ, h, dσ)

Local `3^D` stencil `γ*` annihilating the `3^D - 1` modified plane waves
`F^σ(x) = exp(i κ r·x^σ)` on the neighbourhood `μ_i` (eq. 16, and its 3D
analogue in §3.1).

`dσ[d][t+2] = σ_d(x_d + t h) - σ_d(x_d)` for `t ∈ {-1,0,1}` are the increments
of the complex stretching across the neighbourhood; only increments matter,
because a common factor `F^σ(x_i)` does not change the null vector.  All-zero
increments recover the unstretched (free-space) case.

The system is `3^D x (3^D - 1)`, so the left null space is one-dimensional and
`γ` is unique up to a unimodular factor.
"""
function pml_stencil(κ::Real, h::Real, dσ::NTuple{D,NTuple{3,Float64}}) where {D}
    off = offsets(Val(D))
    dirs = pmldirs(Val(D))
    p = length(off)
    F = Matrix{ComplexF64}(undef, p, p - 1)
    @inbounds for q in 1:(p-1)
        r = dirs[q]
        for s in 1:p
            t = off[s]
            z = zero(ComplexF64)
            for d in 1:D
                z += r[d] * complex(t[d] * h, dσ[d][t[d]+2])
            end
            F[s, q] = exp(im * κ * z)
        end
    end
    U = svd(F; full = true).U
    return conj.(ComplexF64.(U[:, p]))     # γ* = conj(γ), γ ⟂ range(F)
end

pml_stencil(κ::Real, h::Real, d1::NTuple{3,Float64}, d2::NTuple{3,Float64}) =
    pml_stencil(κ, h, (d1, d2))

# --------------------------------------------------------------------------- #
# Complex-stretching profiles
# --------------------------------------------------------------------------- #

"""
    PMLProfile(amp, η, xL, xR)

The stretching `σ(x)` of eq. (14): quadratic ramps of amplitude `amp = C/ω` over
the `η = b h` wide layers outside `[xL, xR]`, zero in between.  The sign
convention (negative on the left, positive on the right) makes outgoing waves
decay.
"""
struct PMLProfile
    amp::Float64
    η::Float64
    xL::Float64
    xR::Float64
end

@inline function (p::PMLProfile)(x::Real)
    if x <= p.xL
        return -p.amp * ((x - p.xL) / p.η)^2
    elseif x >= p.xR
        return p.amp * ((x - p.xR) / p.η)^2
    else
        return 0.0
    end
end

"Global stretching for `Ω^{h+η}`: zero on `(-h, 1+h)`, ramping over `b` layers."
global_profile(g::Grid, ω::Real, C::Real) = PMLProfile(C / ω, g.b * g.h, -g.h, 1 + g.h)

"""
    left_profile(amp, η, xedge)

One-sided stretching used by the moving PML: zero for `x ≥ xedge`, ramping to
`-amp` over the width `η` to its left.  Absorbs waves travelling in `-x₁`.
"""
left_profile(amp::Real, η::Real, xedge::Real) = PMLProfile(amp, η, xedge, Inf)

"Mirror image of [`left_profile`](@ref), for the right-going front."
right_profile(amp::Real, η::Real, xedge::Real) = PMLProfile(amp, η, -Inf, xedge)

"σ increments across the neighbourhood of the grid point with index `i`."
@inline function dsigma(p::PMLProfile, h::Real, i::Integer)
    x = i * h
    s0 = p(x)
    return (p(x - h) - s0, 0.0, p(x + h) - s0)
end

const DSIGMA0 = (0.0, 0.0, 0.0)

# --------------------------------------------------------------------------- #
# Ying (2015) one-sided boundary stencils
# --------------------------------------------------------------------------- #

"""
Clipping signature of a grid point of `I = {1,…,n}^D`: `s_d = -1` on the low
face, `+1` on the high face, `0` in between.  All-zero marks an interior point;
in 3D one nonzero is a face point, two an edge point, three a corner point.
"""
@inline clip_signature(n::Integer, i::NTuple{D,Int}) where {D} =
    ntuple(d -> i[d] == 1 ? -1 : (i[d] == n ? 1 : 0), D)
clip_signature(n::Integer, i1::Integer, i2::Integer) = clip_signature(n, (Int(i1), Int(i2)))

"""
    clipped_offsets(s)

The neighbourhood `μ(i) = {j : ‖i-j‖_∞ ≤ 1, x_j ∈ Ω}` of Ying (2015), translated
to the origin: `3^D` points in the interior, and one factor of 3 replaced by 2
for each clipped direction (6 on a 2D edge, 4 at a 2D corner; 18, 12, 8 for a 3D
face, edge, corner).
"""
function clipped_offsets(s::NTuple{D,Int}) where {D}
    o = NTuple{D,Int}[]
    for t in offsets(Val(D))
        all(d -> s[d] * t[d] <= 0, 1:D) && push!(o, t)
    end
    return o
end
clipped_offsets(s1::Integer, s2::Integer) = clipped_offsets((Int(s1), Int(s2)))

"""
    far_range(s, n, b)

Per-dimension index range of the source set a boundary stencil must annihilate:
Ying's `Eₙ = {j : -n < j₁ < -b, -n < j₂ < n}` for a face clipped in `+x₁`, and
`Cₙ` for the corresponding corner.  Sources closer than `b` layers need no
constraint because `m` vanishes there.
"""
far_range(s::Integer, n::Integer, b::Integer) =
    s == 0 ? ((-n+1):(n-1)) : (s > 0 ? ((-n+1):(-b-1)) : ((b+1):(n-1)))

"""
    BoundaryStencils

Ying's one-sided stencils `B` for the `3^D - 1` boundary cases of a rectangular
grid, keyed by the clipping signature.  These rows *are* the discrete radiation
condition in that scheme.
"""
struct BoundaryStencils{D}
    offsets::Dict{NTuple{D,Int},Vector{NTuple{D,Int}}}
    coef::Dict{NTuple{D,Int},Vector{ComplexF64}}
    ratio::Dict{NTuple{D,Int},Float64}
    buffer::Int
end

"""
    boundary_stencils(K, n; buffer = 2)

Build all `3^D - 1` boundary stencils of Ying (2015) (eqs. 13–14, and the
face/edge/corner list of §3.1).  `buffer` is the paper's `b`: the number of grid
layers next to `∂Ω` on which `m` is assumed to vanish.
"""
function boundary_stencils(K::Kernel{D}, n::Integer; buffer::Integer = 2,
                           chunk::Integer = 8192) where {D}
    off = Dict{NTuple{D,Int},Vector{NTuple{D,Int}}}()
    coef = Dict{NTuple{D,Int},Vector{ComplexF64}}()
    rat = Dict{NTuple{D,Int},Float64}()
    never(_) = false
    for c in CartesianIndices(ntuple(_ -> 1:3, D))
        s = ntuple(d -> c[d] - 2, D)
        all(iszero, s) && continue
        o = clipped_offsets(s)
        v, r = _annihilate(K, o, ntuple(d -> far_range(s[d], n, buffer), D), never;
                           chunk = chunk)
        off[s] = o
        coef[s] = v
        rat[s] = r
    end
    return BoundaryStencils{D}(off, coef, rat, Int(buffer))
end

boundary_stencils(n::Integer, ω::Real, h::Real, k0::Complex; buffer::Integer = 2,
                  chunk::Integer = 8192, table = nothing) =
    boundary_stencils(Kernel(Val(2), ω, h, k0, n + 1), n; buffer = buffer, chunk = chunk)
