using SparsifyAndSweep
using SparsifyAndSweep: CLOG, EULERGAMMA, green2d, kernel_table, ktab
using SpecialFunctions
using LinearAlgebra, Test

@testset "lattice constant" begin
    # CLOG is defined by
    #   ∫ log|y| φ(y) dy − h² Σ_{j≠0} log|jh| φ(jh) = h²(log h + CLOG) φ(0) + O(h⁴)
    # Recompute it for a Gaussian test function, for which the integral is known.
    s = 1.0
    exact = π * s^2 * (log(2s^2) - EULERGAMMA)
    est = Float64[]
    for h in (1 / 16, 1 / 32)
        M = ceil(Int, 13 / h)
        acc = 0.0
        for j2 in -M:M, j1 in -M:M
            (j1 == 0 && j2 == 0) && continue
            r2 = (j1 * h)^2 + (j2 * h)^2
            acc += 0.5 * log(r2) * exp(-r2 / (2s^2))
        end
        push!(est, (exact - h^2 * acc) / h^2 - log(h))
    end
    extrap = est[2] + (est[2] - est[1]) / 3     # the defect converges at O(h²)
    @test isapprox(extrap, CLOG; atol = 1e-7)
end

@testset "self weight" begin
    ω = 2π * 4
    # Both schemes correct the same logarithmic singularity, so they differ only
    # through their lattice constants:  (k_lattice − k_disk)/h² → (c_disk − CLOG)/2π
    # with c_disk = −½logπ − ½.
    predicted = (-0.5log(π) - 0.5 - CLOG) / (2π)
    d = [real((self_weight(ω, 1 / L; scheme = :lattice) -
               self_weight(ω, 1 / L; scheme = :disk)) * L^2) for L in (256, 1024)]
    @test abs(d[2] - predicted) < abs(d[1] - predicted)   # converging
    @test isapprox(d[2], predicted; rtol = 5e-3)
    # magnitude sanity: k₀ is O(h²) with the expected log growth
    for L in (64, 256)
        k0 = self_weight(ω, 1 / L)
        @test isapprox(imag(k0) * L^2, 0.25; rtol = 1e-12)   # imaginary part is h²/4
        @test 0 < real(k0) * L^2 < 2
    end
end

@testset "Toeplitz apply by FFT" begin
    n, ω = 20, 2π * 3
    h = 1 / (n + 1)
    k0 = self_weight(ω, h)
    C = ToeplitzConv(ω, h, n, k0)
    T = kernel_table(ω, h, n, k0)
    N = n^2
    K = Matrix{ComplexF64}(undef, N, N)
    for j2 in 1:n, j1 in 1:n, i2 in 1:n, i1 in 1:n
        K[i1+(i2-1)*n, j1+(j2-1)*n] = ktab(T, n, i1 - j1, i2 - j2)
    end
    x = randn(ComplexF64, N)
    @test norm(C * x - K * x) / norm(K * x) < 1e-13
end

# --------------------------------------------------------------------------- #
# The Nyström weights must reproduce the true volume potential.  For a radially
# symmetric density ψ supported in |y| < R, Graf's addition theorem gives
#   V(ρ) = (iπ/2)[ H₀(ωρ)∫₀^ρ J₀(ωs)ψ(s)s ds + J₀(ωρ)∫_ρ^R H₀(ωs)ψ(s)s ds ].
# --------------------------------------------------------------------------- #

"Composite Simpson on [a,b] with `2M` panels."
function simpson(f, a, b, M)
    h = (b - a) / (2M)
    h == 0 && return zero(f(b))
    s = f(a) + f(b)
    for k in 1:(2M-1)
        s += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return s * h / 3
end

hankel0(z) = complex(besselj0(z), bessely0(z))

function exact_volume_potential(ω, ρ, ψ, R; M = 8_000)
    # s·H₀(ωs) → 0 as s → 0, but the expression evaluates to Inf*0 there
    outer = simpson(s -> s == 0 ? zero(ComplexF64) : hankel0(ω * s) * ψ(s) * s, ρ, R, M)
    ρ == 0 && return (im * π / 2) * outer
    inner = simpson(s -> besselj0(ω * s) * ψ(s) * s, 0.0, ρ, M)
    return (im * π / 2) * (hankel0(ω * ρ) * inner + besselj0(ω * ρ) * outer)
end

function potential_error(n, ω, ψr, R, ctr, scheme)
    h = 1 / (n + 1)
    C = ToeplitzConv(ω, h, n, self_weight(ω, h; scheme = scheme))
    x = gridpoints(n)
    ψ = [ψr(hypot(x[i] - ctr, x[j] - ctr)) for i in 1:n, j in 1:n]
    V = reshape(C * vec(ComplexF64.(ψ)), n, n)
    e, s = 0.0, 0.0
    for (i, j) in ((n ÷ 2 + 1, n ÷ 2 + 1), (n ÷ 3, n ÷ 2), (n ÷ 4, n ÷ 4),
                   (2n ÷ 3, 3n ÷ 4), (n ÷ 8, n ÷ 8))
        ρ = hypot(x[i] - ctr, x[j] - ctr)
        ρ >= R && continue
        ex = exact_volume_potential(ω, ρ, ψr, R)
        e = max(e, abs(V[i, j] - ex))
        s = max(s, abs(ex))
    end
    return e / s
end

@testset "volume potential accuracy" begin
    ω, R, ctr = 2π * 4, 0.3, 0.5
    ψr(s) = SparsifyAndSweep.bump(s / R)          # C^∞, supported in s < R
    ns = (63, 127, 255)

    el = [potential_error(n, ω, ψr, R, ctr, :lattice) for n in ns]
    ed = [potential_error(n, ω, ψr, R, ctr, :disk) for n in ns]
    ratel = [log2(el[k] / el[k+1]) for k in 1:(length(ns)-1)]
    rated = [log2(ed[k] / ed[k+1]) for k in 1:(length(ns)-1)]
    @info "volume potential" ns lattice = el rate_lattice = ratel disk = ed rate_disk = rated

    @test el[1] < 1e-4                            # accurate already on 63²
    @test el[end] < 1e-6
    @test all(>(3.5), ratel)                      # fourth order
    @test all(r -> 1.7 < r < 2.4, rated)          # the disk weight is only 2nd order
    @test all(el .< ed)                           # and uniformly worse
end
