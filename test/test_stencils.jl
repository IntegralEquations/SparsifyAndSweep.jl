using SparsifyAndSweep
using SparsifyAndSweep: OFFSETS, PMLDIRS, PMLProfile, dsigma, global_profile,
                        left_profile, kernel_table, ktab, ilo, ihi
using LinearAlgebra, Test

@testset "sparsifying stencils α, β" begin
    n, f = 63, 8
    ω = 2π * f
    h = 1 / (n + 1)
    k0 = self_weight(ω, h)
    astar, bstar, ratio = sparsify_stencils(n, ω, h, k0)

    @test length(astar) == 9 && length(bstar) == 9
    @test isapprox(norm(astar), 1; atol = 1e-12)
    @test ratio < 1e-3                                   # far field is annihilated

    # The square lattice, μ and μᶜ are invariant under the dihedral group, so the
    # minimal singular vector must be too: corners equal, edges equal.
    A = reshape(astar, 3, 3)
    B = reshape(bstar, 3, 3)
    for S in (A, B)
        corners = [S[1, 1], S[1, 3], S[3, 1], S[3, 3]]
        edges = [S[1, 2], S[2, 1], S[2, 3], S[3, 2]]
        @test maximum(abs, corners .- corners[1]) < 1e-8 * abs(S[2, 2])
        @test maximum(abs, edges .- edges[1]) < 1e-8 * abs(S[2, 2])
    end

    # α* really is a discrete (−Δ − ω²): comparing the edge coefficient with the
    # compact 9-point Laplacian h²(−Δ) ≈ [−1 −4 −1; −4 20 −4; −1 −4 −1]/6 fixes
    # the scale, and then the row sum must be −scale·h²ω² (the Laplacian part
    # sums to zero).
    scale = real(A[2, 1]) / (-4 / 6)
    @test isapprox(sum(real, astar), -scale * h^2 * ω^2; rtol = 0.05)

    # β* = α* K_{μ,μ}, and Σβ* must match the same scale (it multiplies ω²m u).
    @test isapprox(sum(real, bstar), scale * h^2; rtol = 0.05)

    # α* annihilates samples of *any* homogeneous Helmholtz solution, because the
    # columns of K_{μ,μᶜ} are Green's functions centred outside μ.
    for θ in (0.0, 0.3, 0.7, 1.1)
        r = (cos(θ), sin(θ))
        v = [cis(ω * h * (r[1] * t1 + r[2] * t2)) for (t1, t2) in OFFSETS]
        @test abs(sum(astar .* v)) < 1e-3
    end

    # β* is the Green's-function-weighted average of α*, hence O(h²)
    @test abs(bstar[5]) / h^2 < 1
end

@testset "PML stencils γ" begin
    ω, h = 2π * 8, 1 / 64
    dσ0 = (0.0, 0.0, 0.0)

    # free space: exact annihilation of the eight fitted directions
    γ = pml_stencil(ω, h, dσ0, dσ0)
    @test isapprox(norm(γ), 1; atol = 1e-12)
    for (r1, r2) in PMLDIRS
        v = [cis(ω * h * (r1 * t1 + r2 * t2)) for (t1, t2) in OFFSETS]
        @test abs(sum(γ .* v)) < 1e-12
    end
    # and small error on an unfitted direction
    r = (cos(0.3), sin(0.3))
    v = [cis(ω * h * (r[1] * t1 + r[2] * t2)) for (t1, t2) in OFFSETS]
    @test abs(sum(γ .* v)) < 1e-5

    # with stretching: exact annihilation of the *modified* plane waves
    g = Grid(63, 8)
    prof = global_profile(g, ω, 8.0)
    d1 = dsigma(prof, h, -3)          # inside the left PML
    d2 = dsigma(prof, h, 20)          # interior in x₂
    @test d1 != dσ0 && d2 == dσ0
    γs = pml_stencil(ω, h, d1, d2)
    for (r1, r2) in PMLDIRS
        v = [cis(ω * (r1 * complex(t1 * h, d1[t1+2]) + r2 * complex(t2 * h, d2[t2+2])))
             for (t1, t2) in OFFSETS]
        @test abs(sum(γs .* v)) < 1e-10
    end
end

@testset "PML profiles" begin
    g = Grid(63, 8)
    ω, C = 2π * 8, 8.0
    p = global_profile(g, ω, C)
    @test p(0.5) == 0.0
    @test p(-g.h) == 0.0                       # σ vanishes at the inner PML edge
    @test p(1 + g.h) == 0.0
    @test p(-g.h - g.b * g.h) ≈ -C / ω         # full amplitude at the outer edge
    @test p(1 + g.h + g.b * g.h) ≈ C / ω
    @test p(-2g.h) < 0 && p(1 + 2g.h) > 0      # correct signs for decay

    q = left_profile(C / ω, 4 * g.h, 0.25)
    @test q(0.3) == 0.0
    @test q(0.25) == 0.0
    @test q(0.25 - 4g.h) ≈ -C / ω
end

@testset "slice partition" begin
    for (n, b) in ((63, 8), (127, 8), (255, 8), (100, 8), (37, 4), (20, 8))
        g = Grid(n, b)
        rs = slice_ranges(g)
        @test first(first(rs)) == ilo(g)
        @test last(last(rs)) == ihi(g)
        for k in 2:length(rs)
            @test first(rs[k]) == last(rs[k-1]) + 1        # contiguous, disjoint
        end
        @test sum(length, rs) == ntot(g)
        if length(rs) > 1
            @test length(rs[1]) == 2b && length(rs[end]) == 2b
            for k in 2:(length(rs)-1)
                # interior slices have width b, except a short tail folded in
                @test 1 <= length(rs[k]) <= 2b
            end
            @test count(k -> length(rs[k]) != b, 2:(length(rs)-1)) <= 1
        end
    end
end
