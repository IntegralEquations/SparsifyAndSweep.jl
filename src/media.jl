# Velocity fields c(x) and the perturbation m(x) = 1 - 1/c(x)².
#
# The four fields of Liu & Ying §2.5.  The paper describes them qualitatively
# ("a converging Gaussian centred at (0.5,0.5)", "32 randomly placed converging
# Gaussians with narrow width") and its colour bars show c ∈ [0.7, 1.3], but the
# Gaussian widths are not given, so `width` is a free parameter here; see
# `examples/field_width_study.jl` for its effect on the iteration count.
#
# All fields are tapered so that m is *exactly* supported inside Ω, which the
# Lippmann–Schwinger formulation and the global PML both assume.

"C^∞ bump: 1 at t = 0, identically 0 for t ≥ 1."
@inline bump(t::Real) = t >= 1 ? 0.0 : exp(1 - 1 / (1 - t^2))

"C^∞ ramp: 0 for t ≤ 0, 1 for t ≥ 1."
@inline function ramp(t::Real)
    t <= 0 && return 0.0
    t >= 1 && return 1.0
    a = exp(-1 / t)
    return a / (a + exp(-1 / (1 - t)))
end

"Window equal to 1 on `[d, 1-d]` and vanishing smoothly at 0 and 1."
@inline taper(x::Real, d::Real) = ramp(x / d) * ramp((1 - x) / d)

"""
    lens(r, width, Rmax; shape = :gaussian)

Radial lens profile, 1 at `r = 0` and exactly 0 for `r ≥ Rmax`.

`:gaussian` is `exp(-r²/2σ²)` (σ = `width`) multiplied by a `C^∞` taper acting
over the outermost sliver of `[0, Rmax]`, where the Gaussian has already decayed
to `O(10⁻⁵)`; `:bump` is the compactly supported `exp(1 - 1/(1-t²))` of radius
`width`.
"""
@inline function lens(r::Real, width::Real, Rmax::Real; shape::Symbol = :gaussian)
    if shape === :bump
        return bump(r / width)
    end
    Rc = min(6.5 * width, Rmax)
    Rt = min(5 * width, 0.9 * Rc)
    return exp(-r^2 / (2 * width^2)) * ramp((Rc - r) / (Rc - Rt))
end

"Grid coordinates of `I = {1,…,n}²`."
gridpoints(n::Integer) = ((1:n) ./ (n + 1))

"m(x) = 1 - 1/c(x)² from a sampled velocity field."
perturbation(c::AbstractArray) = @. 1 - 1 / c^2

"Default lens width per field (see the module docstring on `width`)."
default_width(kind::Symbol) = kind === :multi ? 0.02 : 0.1

"""
    velocity(kind, n; amp = 0.3, width = default_width(kind), shape = :gaussian, seed = 1)

Sample one of the paper's test velocity fields on `I = {1,…,n}²`:

* `:converging` - a single converging (slow, `c < 1`) lens at (0.5,0.5)
* `:diverging`  - the same lens with the opposite sign (fast, `c > 1`)
* `:multi`      - 32 randomly placed narrow converging lenses
* `:random`     - a smooth random field, equal to 1 near ∂Ω
* `:constant`   - `c ≡ 1` (m ≡ 0), useful for tests
"""
function velocity(::Val{D}, kind::Symbol, n::Integer; seed::Integer = 1,
                  amp::Real = 0.3, width::Real = default_width(kind),
                  shape::Symbol = :gaussian) where {D}
    x = gridpoints(n)
    dims = ntuple(_ -> n, D)
    c = ones(Float64, dims)
    kind === :constant && return c
    idx = CartesianIndices(dims)
    if kind === :converging || kind === :diverging
        sgn = kind === :converging ? -amp : amp
        @inbounds for cid in idx
            r = sqrt(sum(d -> (x[cid[d]] - 0.5)^2, 1:D))
            c[cid] = 1 + sgn * lens(r, width, 0.5; shape = shape)
        end
    elseif kind === :multi
        rng = MersenneTwister(seed)
        nb, pad = (D == 2 ? 32 : 256), 0.12
        ctr = [ntuple(_ -> pad + (1 - 2pad) * rand(rng), D) for _ in 1:nb]
        @inbounds for cid in idx
            v = 0.0
            for q in ctr
                r = sqrt(sum(d -> (x[cid[d]] - q[d])^2, 1:D))
                v = max(v, lens(r, width, pad; shape = shape))
            end
            c[cid] = 1 - amp * v
        end
    elseif kind === :random
        f = smooth_random(Val(D), n; seed = seed, corr = 0.08)
        @inbounds for cid in idx
            w = prod(d -> taper(x[cid[d]], 0.15), 1:D)
            c[cid] = 1 + amp * f[cid] * w
        end
    else
        throw(ArgumentError("unknown velocity field $kind"))
    end
    return c
end

velocity(kind::Symbol, n::Integer; kwargs...) = velocity(Val(2), kind, n; kwargs...)

"""
    smooth_random(n; seed, corr)

Band-limited random field on the `nxn` grid with correlation length `corr`,
scaled to `max|f| = 1`.
"""
function smooth_random(::Val{D}, n::Integer; seed::Integer = 1,
                       corr::Real = 0.08) where {D}
    rng = MersenneTwister(seed)
    ŵ = fft(randn(rng, ntuple(_ -> n, D)))
    freq = [(k <= n ÷ 2 ? k - 1 : k - 1 - n) for k in 1:n]
    @inbounds for cid in CartesianIndices(ŵ)
        q2 = sum(d -> freq[cid[d]]^2, 1:D) * (2π * corr)^2
        ŵ[cid] *= exp(-q2 / 2)
    end
    f = real(ifft(ŵ))
    return f ./ maximum(abs, f)
end
smooth_random(n::Integer; kwargs...) = smooth_random(Val(2), n; kwargs...)

"""
    plane_wave(Val(D), n, ω, dir)

Plane wave `exp(i ω dir·x)` sampled on `I`; the default `dir` points along the
last axis, matching the paper's "plane wave shooting downward".
"""
function plane_wave(::Val{D}, n::Integer, ω::Real,
                    dir::NTuple{D,Float64} = ntuple(d -> d == D ? 1.0 : 0.0, D)) where {D}
    x = gridpoints(n)
    u = Array{ComplexF64,D}(undef, ntuple(_ -> n, D))
    @inbounds for cid in CartesianIndices(u)
        u[cid] = cis(ω * sum(d -> dir[d] * x[cid[d]], 1:D))
    end
    return u
end
plane_wave(n::Integer, ω::Real) = plane_wave(Val(2), n, ω)
plane_wave(n::Integer, ω::Real, d::NTuple{2,Float64}) = plane_wave(Val(2), n, ω, d)
