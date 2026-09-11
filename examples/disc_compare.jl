# The layered sweep against L-sweeps on a constant-contrast disc, with the Mie
# series as the reference solution.  The disc has radius R centred at (1/2, 1/2)
# in the unit square -- the unit disc after rescaling lengths by 1/R, so that its
# size parameter is ωR -- and velocity c_in inside (index 1/c_in), 1 outside; the
# incident field is the package's default plane wave e^{iωx₂}; the package solves
# for the scattered field, and that is what the Mie error measures.  Every row solves
# the same discrete system, so the Mie error it reports is the discretisation
# error, the same for every row at a frequency; it validates the solution rather
# than the preconditioner.
#
#   julia --project=. -t 16 examples/disc_compare.jl 2 4 8 16 32 64 128 256
#   julia --project=. -t 16 examples/disc_compare.jl 256 --only=sweep
using SparsifyAndSweep, LinearAlgebra, SparseArrays, SpecialFunctions, Printf, Random

# ------------------------------------------------------------------ Mie series
"""
    mie_coefficients(k, R, nin, M)

Scattered (`a`) and interior (`b`) coefficients, orders 0..M, of a plane wave
scattered by a penetrable disc of radius `R` and refractive index `nin`, for the
scalar Helmholtz equation with `u` and `∂ᵣu` continuous.  With the Wronskian the
interior coefficient is `b_m = (2i/πR)/D_m`.
"""
function mie_coefficients(k::Real, R::Real, nin::Real, M::Int)
    kin = nin * k
    x, y = k * R, kin * R
    J(m, z) = besselj(m, z); H(m, z) = besselh(m, 1, z)
    dJ(m, z) = (J(m - 1, z) - J(m + 1, z)) / 2
    dH(m, z) = (H(m - 1, z) - H(m + 1, z)) / 2
    a = zeros(ComplexF64, M + 1); b = zeros(ComplexF64, M + 1)
    for m in 0:M
        D = k * dH(m, x) * J(m, y) - kin * dJ(m, y) * H(m, x)
        a[m+1] = (kin * dJ(m, y) * J(m, x) - k * dJ(m, x) * J(m, y)) / D
        b[m+1] = (2im / (π * R)) / D
    end
    return a, b
end

"Total field of the Mie solution at the points `(x1, x2)`; incident wave e^{ikx₂}."
function mie_field(x1::AbstractVector, x2::AbstractVector, k::Real, R::Real, nin::Real,
                   centre::NTuple{2,Float64})
    kin = nin * k
    M = ceil(Int, kin * R + 12 * cbrt(kin * R) + 10)
    a, b = mie_coefficients(k, R, nin, M)
    u = Vector{ComplexF64}(undef, length(x1))
    @inbounds for p in eachindex(x1)
        dx, dy = x1[p] - centre[1], x2[p] - centre[2]
        r = hypot(dx, dy)
        θ = atan(dy, dx)                    # e^{ikx₂} = Σ_m J_m(kr) e^{imθ}, θ from the x₁ axis
        inside = r < R
        acc = zero(ComplexF64)
        for m in 0:M
            cm = inside ? b[m+1] * besselj(m, kin * r) :
                          besselj(m, k * r) + a[m+1] * besselh(m, 1, k * r)
            e = cis(m * θ)
            acc += m == 0 ? cm : cm * (e + (isodd(m) ? -conj(e) : conj(e)))   # ±m terms
        end
        u[p] = acc
    end
    return u
end

# ------------------------------------------------------------------ problem
function disc_velocity(n::Int, R::Float64, cin::Float64, centre::NTuple{2,Float64})
    x = gridpoints(n)
    c = ones(Float64, n, n)
    for j in 1:n, i in 1:n
        (x[i] - centre[1])^2 + (x[j] - centre[2])^2 < R^2 && (c[i, j] = cin)
    end
    return c
end

"""
Relative ℓ² error of the scattered field `usc` (what `solve` returns: the package
solves `u_sc + ω²K[m u_sc] = -ω²K[m u_inc]`) against the Mie scattered field
`u_Mie - e^{iωx₂}`, on `nsamp` random grid points.
"""
function mie_error(usc::AbstractVector{ComplexF64}, n::Int, ω::Real, R, cin, centre; nsamp = 20_000)
    x = collect(gridpoints(n))
    N = n * n
    idx = nsamp >= N ? collect(1:N) : sort!(randperm(MersenneTwister(11), N)[1:nsamp])
    i = @. mod(idx - 1, n) + 1;  j = @. div(idx - 1, n) + 1
    uref = mie_field(x[i], x[j], ω, R, 1 / cin, centre) .- cis.(ω .* x[j])
    return norm(usc[idx] - uref) / norm(uref), length(idx)
end

# ------------------------------------------------------------------ table
R, cin, centre, ppw, only, fwdonly, sizes = 0.25, 0.8, (0.5, 0.5), 8, nothing, false, Int[]
for a in ARGS
    if a == "--costs-only";         global fwdonly = true    # per-operation costs, no factorisation
    elseif startswith(a, "--only="); global only = split(a[8:end], ',')
    elseif startswith(a, "--ppw="); global ppw = parse(Int, a[7:end])     # points per wavelength (h-refinement at fixed ω)
    elseif startswith(a, "--R=");   global R = parse(Float64, a[5:end])
    elseif startswith(a, "--cin="); global cin = parse(Float64, a[7:end])
    else push!(sizes, parse(Int, a)) end
end

const ROWS = (
    ("direct",    "exact solve of H",        (; mode = :direct)),      # the iteration floor: sparsification error alone
    ("sweep",     "layered sweep",           (; mode = :sweep)),
    ("lsweep8",   "L-sweeps pad 8",          (; mode = :lsweep, nbox = 64, pad = 8)),
    ("lsweep16",  "L-sweeps pad 16",         (; mode = :lsweep, nbox = 64, pad = 16)),
    ("lsweep24",  "L-sweeps pad 24",         (; mode = :lsweep, nbox = 64, pad = 24)),
    ("lsweep8i6", "L-sweeps pad 8, inner 6", (; mode = :lsweep, nbox = 64, pad = 8, inner = 6)),
)

# Two work counts, both measured rather than inferred.  `sweeps` is the number of
# preconditioner applications performed inside the outer Krylov solve -- sweeps
# proper, or LU solves for the exact-H row; `fwd` the number of forward maps, i.e.
# applications of the dense Lippmann-Schwinger operator I + ω²K[m ·], each an FFT
# convolution.  A left-preconditioned GMRES(20) applies both once per iteration
# and once more at every restart, so each exceeds the iteration count by
# 1 + restarts; with inner iterations the forward maps stay at one per outer
# iteration while the sweeps multiply.  `solve` evaluates one further forward map
# for its true residual, which is excluded here.
function row(P, b, n, ω, label; kwargs...)
    M = SweepPreconditioner(P; kwargs...)
    M.nsweeps[] = 0
    apply_M(M, b)                      # warm up; also the sweep count of one application
    per_apply = M.nsweeps[]
    M.nsweeps[] = 0
    P.nA[] = 0
    r = solve(P, b; M = M)
    sweeps = M.nsweeps[] - per_apply   # solve() times one application before GMRES
    fwd = P.nA[] - 1                   # less solve()'s own true-residual evaluation
    err, ns = mie_error(r.u, n, ω, R, cin, centre)
    @printf("  %-26s setup %7.2fs  apply %7.3fs  iters %3d  sweeps %4d  fwd %4d  solve %7.2fs  relres %.1e  Mie err %.2e (%d pts)  peak %5.1fGB\n",
            label, r.tsetup, r.tapply, r.iters, sweeps, fwd, r.tsolve, r.relres, err, ns,
            Sys.maxrss() / 2^30)
end

# Two per-operation costs that depend on the grid alone -- not on the medium, and
# not on which preconditioner a row uses -- so one measurement per size serves
# every row and no factorisation has to be built.  The forward map is an FFT
# convolution on the padded grid plus a pointwise multiply; the sparse matvec is
# H, the sparsified operator, which carries nine nonzeros per row at any size.
if fwdonly
    @printf("%-6s %-6s %-11s %-11s %-9s %s\n",
            "ω/2π", "n", "t_fwd", "t_Hmv", "nnz(H)/row", "(best of 5)")
    for nn in sizes
        n, ω = ppw * nn - 1, 2π * nn
        c = disc_velocity(n, R, cin, centre)
        P = LSProblem(n, ω, c; b = 8)
        u = rhs(P, plane_wave(n, ω)); y = similar(u)
        apply_A!(y, P, u)                                   # warm up
        tf = minimum(@elapsed(apply_A!(y, P, u)) for _ in 1:5)
        g = P.g
        K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
        astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
        H = build_H(g, ω, perturbation(c), astar, bstar; C = 12.0)
        v = randn(ComplexF64, size(H, 2)); w = similar(v)
        mul!(w, H, v)
        th = minimum(@elapsed(mul!(w, H, v)) for _ in 1:5)
        @printf("%-6d %-6d %-11.4f %-11.4f %-9.1f\n", nn, n, tf, th, nnz(H) / size(H, 1))
        flush(stdout); P = nothing; H = nothing; GC.gc()
    end
    exit(0)
end

for nn in sizes
    n, ω = ppw * nn - 1, 2π * nn
    c = disc_velocity(n, R, cin, centre)
    P = LSProblem(n, ω, c; b = 8)
    b = rhs(P, plane_wave(n, ω))
    q = max(1, round(Int, ntot(P.g) / 64))
    @printf("ω/2π = %d, n = %d (%d points per wavelength), N = %d, disc R = %.2f (ωR = %.1f), c_in = %.2f, threads = %d, CDD %dx%d\n",
            nn, n, ppw, n^2, R, ω * R, cin, Threads.nthreads(), q, q)
    for (key, label, kw) in ROWS
        (only === nothing || key in only) || continue
        row(P, b, n, ω, label; kw...)
    end
end
