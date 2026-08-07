# Free-space Green's function, the Nyström weights k_i, and the FFT-based
# application of the (D-level Toeplitz) matrix K.  Dimension-generic.

const EULERGAMMA = 0.577215664901532860606512090082

"""
2D square-lattice constant for the logarithmic kernel:

    ∫_{R²} log|y| φ(y) dy  -  h² Σ_{j≠0} log|jh| φ(jh)  =  h²(log h + CLOG) φ(0) + O(h⁴)
"""
const CLOG = -1.3105329259115095

"""
3D simple-cubic lattice constant for the Coulomb kernel:

    ∫_{R³} φ(y)/|y| dy  -  h³ Σ_{j≠0} φ(jh)/|jh|  =  h² C3 φ(0) + O(h⁴)

(the analytic continuation `-ζ_{Z³}(1)` of the Epstein zeta function).  Both
constants are obtained by Richardson extrapolation of the punctured trapezoidal
defect for a Gaussian test function; see `test/test_kernel.jl`.
"""
const C3 = 2.8372986

"Free-space Helmholtz Green's function: `(i/4)H₀⁽¹⁾(ωr)` in 2D, `e^{iωr}/(4πr)` in 3D."
@inline green(::Val{2}, ω::Real, r::Real) = (im / 4) * complex(besselj0(ω * r), bessely0(ω * r))
@inline green(::Val{3}, ω::Real, r::Real) = cis(ω * r) / (4π * r)
green2d(ω::Real, r::Real) = green(Val(2), ω, r)

"""
    self_weight(Val(D), ω, h; scheme = :lattice)

Quadrature correction `k₀` replacing the singular diagonal weight `h^D G(0)` of
the punctured trapezoidal rule.

The Green's function is split into its singular part and an analytic remainder,

* `D = 2`: `G(r) = -(1/2π)log(r) J₀(ωr) + S(r)`, `S(0) = i/4 - (log(ω/2)+γ)/(2π)`,
  giving `k₀ = h²[S(0) - (log h + CLOG)/(2π)]`;
* `D = 3`: `G(r) = 1/(4πr) + S(r)`, `S(0) = iω/(4π)`,
  giving `k₀ = h³ S(0) + h² C3/(4π)`.

Both are fourth-order accurate for smooth densities.  `scheme = :disk` selects
the cruder equal-volume-ball weight, kept for comparison (second order, 2D only).
"""
function self_weight(::Val{2}, ω::Real, h::Real; scheme::Symbol = :lattice)
    if scheme === :lattice
        S0 = im / 4 - (log(ω / 2) + EULERGAMMA) / (2π)
        return h^2 * (S0 - (log(h) + CLOG) / (2π))
    elseif scheme === :disk
        r0 = h / sqrt(π)
        z = ω * r0
        H1 = complex(besselj1(z), bessely1(z))
        return (im * π * r0 / (2ω)) * H1 - 1 / ω^2
    else
        throw(ArgumentError("unknown quadrature scheme $scheme"))
    end
end

function self_weight(::Val{3}, ω::Real, h::Real; scheme::Symbol = :lattice)
    scheme === :lattice || throw(ArgumentError("only :lattice is available in 3D"))
    return h^3 * (im * ω / (4π)) + h^2 * C3 / (4π)
end

self_weight(ω::Real, h::Real; kwargs...) = self_weight(Val(2), ω, h; kwargs...)

# --------------------------------------------------------------------------- #
# Nyström weights
# --------------------------------------------------------------------------- #

"""
    Kernel{D,T}

The Nyström weights `k_i = h^D G(p_i)` (with the corrected `k₀` at the origin),
either tabulated (`T = Array{ComplexF64,D}`) or evaluated on the fly
(`T = Nothing`).

2D tabulates, because each entry costs a pair of Bessel evaluations; 3D does
not, because `e^{iωr}/(4πr)` is cheap and a table over `{-n-1,…,n+1}³` would be
the largest array in the program.
"""
struct Kernel{D,T}
    ω::Float64
    h::Float64
    k0::ComplexF64
    tab::T
    R::Int
end

function Kernel(::Val{2}, ω::Real, h::Real, k0::Complex, R::Integer)
    T = Matrix{ComplexF64}(undef, 2R + 1, 2R + 1)
    h2 = h^2
    @inbounds for i2 in -R:R, i1 in -R:R
        T[i1+R+1, i2+R+1] =
            (i1 == 0 && i2 == 0) ? k0 :
            h2 * green(Val(2), ω, h * sqrt(float(i1)^2 + float(i2)^2))
    end
    return Kernel{2,typeof(T)}(ω, h, k0, T, Int(R))
end

Kernel(::Val{3}, ω::Real, h::Real, k0::Complex, R::Integer) =
    Kernel{3,Nothing}(ω, h, k0, nothing, Int(R))

@inline function kval(K::Kernel{D,Nothing}, i::NTuple{D,Int}) where {D}
    s = 0
    @inbounds for d in 1:D
        s += i[d] * i[d]
    end
    s == 0 && return K.k0
    return K.h^D * green(Val(D), K.ω, K.h * sqrt(float(s)))
end

@inline function kval(K::Kernel{2,Matrix{ComplexF64}}, i::NTuple{2,Int})
    @inbounds return K.tab[i[1]+K.R+1, i[2]+K.R+1]
end

# legacy 2D helpers used by the tests
kernel_table(ω::Real, h::Real, R::Integer, k0::Complex) = Kernel(Val(2), ω, h, k0, R).tab
@inline ktab(T::AbstractMatrix, R::Integer, i1::Integer, i2::Integer) = T[i1+R+1, i2+R+1]

# --------------------------------------------------------------------------- #
# Fast application of K by circulant embedding
# --------------------------------------------------------------------------- #

"""
    ToeplitzConv(Val(D), ω, h, n, k0)

Applies the `n^D x n^D` `D`-level Toeplitz matrix `K` with `K[i,j] = k_{i-j}` to
vectors on `I = {1,…,n}^D` in `O(N log N)` via an `L^D` circulant embedding.
"""
struct ToeplitzConv{D,P1,P2}
    n::Int
    L::Int
    khat::Array{ComplexF64,D}
    buf::Array{ComplexF64,D}
    P::P1
    Pi::P2
end

function ToeplitzConv(::Val{D}, ω::Real, h::Real, n::Integer, k0::Complex) where {D}
    L = nextprod((2, 3, 5), 2n - 1)
    c = zeros(ComplexF64, ntuple(_ -> L, D))
    hD = h^D
    rng = ntuple(_ -> (-(n-1)):(n-1), D)
    @inbounds for cid in CartesianIndices(rng)
        i = Tuple(cid)
        s = 0
        for d in 1:D
            s += i[d] * i[d]
        end
        v = s == 0 ? k0 : hD * green(Val(D), ω, h * sqrt(float(s)))
        c[ntuple(d -> mod(i[d], L) + 1, D)...] = v
    end
    khat = fft(c)
    buf = zeros(ComplexF64, ntuple(_ -> L, D))
    P = plan_fft!(buf; flags = FFTW.MEASURE)
    Pi = plan_bfft!(buf; flags = FFTW.MEASURE)
    return ToeplitzConv{D,typeof(P),typeof(Pi)}(Int(n), L, khat, buf, P, Pi)
end

ToeplitzConv(ω::Real, h::Real, n::Integer, k0::Complex) = ToeplitzConv(Val(2), ω, h, n, k0)

"`y ← K x` for `x`, `y` of length `n^D` (column-major on `I`)."
function LinearAlgebra.mul!(y::AbstractVector, C::ToeplitzConv{D}, x::AbstractVector) where {D}
    n, L = C.n, C.L
    fill!(C.buf, 0)
    X = reshape(x, ntuple(_ -> n, D))
    sub = ntuple(_ -> 1:n, D)
    @inbounds copyto!(view(C.buf, sub...), X)
    C.P * C.buf
    @inbounds @. C.buf *= C.khat
    C.Pi * C.buf
    s = 1 / float(L)^D
    Y = reshape(y, ntuple(_ -> n, D))
    @inbounds copyto!(Y, view(C.buf, sub...))
    @inbounds @. Y *= s
    return y
end

Base.:*(C::ToeplitzConv, x::AbstractVector) = mul!(similar(x, ComplexF64), C, x)
