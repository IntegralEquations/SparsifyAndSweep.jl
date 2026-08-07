using SparsifyAndSweep
using SparsifyAndSweep: C3, green, kval, mval, boxindices, physindices, inbox, nslices,
                        phys_indices, leafstats, boxlin, boxidx, boxlen
using LinearAlgebra, SparseArrays, Test

@testset "3D grid bookkeeping" begin
    g = Grid{3}(15, 3)
    @test dim(g) == 3
    @test ntot(g) == 15 + 2 + 6
    @test nunk(g) == ntot(g)^3
    @test nphys(g) == 15^3
    # linear indices must be a bijection onto 1:nunk, x₁ fastest
    seen = falses(nunk(g))
    for i in boxindices(g)
        k = lin(g, i)
        @test !seen[k]
        seen[k] = true
    end
    @test all(seen)
    @test lin(g, (ilo(g), ilo(g), ilo(g))) == 1
    @test lin(g, (ilo(g) + 1, ilo(g), ilo(g))) == 2          # x₁ fastest
    @test lin(g, (ihi(g), ihi(g), ihi(g))) == nunk(g)
    @test plin(g, (1, 1, 1)) == 1
    @test plin(g, (2, 1, 1)) == 2
    @test plin(g, (15, 15, 15)) == nphys(g)
    @test length(phys_indices(g)) == nphys(g) && allunique(phys_indices(g))
end

@testset "3D neighbourhood and directions" begin
    @test length(offsets(Val(3))) == 27
    @test allunique(offsets(Val(3)))
    @test offsets(Val(3))[1] == (-1, -1, -1)
    @test offsets(Val(3))[2] == (0, -1, -1)                   # x₁ fastest
    @test (0, 0, 0) in offsets(Val(3))
    @test length(pmldirs(Val(3))) == 26                       # eq. R of §3.1
    @test all(d -> isapprox(sqrt(sum(abs2, d)), 1; atol = 1e-14), pmldirs(Val(3)))
    @test !any(d -> all(iszero, d), pmldirs(Val(3)))
end

@testset "3D lattice constant and self weight" begin
    # C3 is defined by ∫φ/|y| − h³Σ_{j≠0}φ/|jh| = h² C3 φ(0) + O(h⁴)
    s = 1.0
    exact = 4π * s^2
    est = Float64[]
    for h in (1 / 8, 1 / 16)
        M = ceil(Int, 8 / h)
        acc = 0.0
        for j3 in -M:M, j2 in -M:M, j1 in -M:M
            (j1 == 0 && j2 == 0 && j3 == 0) && continue
            r2 = (j1 * h)^2 + (j2 * h)^2 + (j3 * h)^2
            acc += exp(-r2 / (2s^2)) / sqrt(r2)
        end
        push!(est, (exact - h^3 * acc) / h^2)
    end
    @test isapprox(est[2] + (est[2] - est[1]) / 3, C3; atol = 2e-6)

    ω, h = 2π * 3, 1 / 32
    k0 = self_weight(Val(3), ω, h)
    @test imag(k0) ≈ h^3 * ω / (4π)          # the analytic part is h³ S(0)
    @test real(k0) ≈ h^2 * C3 / (4π)         # the singular correction is O(h²)
    @test_throws ArgumentError self_weight(Val(3), ω, h; scheme = :disk)
end

@testset "3D Green's function and Toeplitz apply" begin
    ω, r = 2π * 3, 0.21
    @test green(Val(3), ω, r) ≈ cis(ω * r) / (4π * r)

    n = 9
    h = 1 / (n + 1)
    k0 = self_weight(Val(3), ω, h)
    C = ToeplitzConv(Val(3), ω, h, n, k0)
    K = Kernel(Val(3), ω, h, k0, n)
    N = n^3
    A = Matrix{ComplexF64}(undef, N, N)
    g = Grid{3}(n, 1)
    for jc in physindices(g), ic in physindices(g)
        A[plin(g, ic), plin(g, jc)] = kval(K, ic .- jc)
    end
    x = randn(ComplexF64, N)
    @test norm(C * x - A * x) / norm(A * x) < 1e-12
end

@testset "3D sparsifying stencils" begin
    n, f = 31, 4
    ω = 2π * f
    h = 1 / (n + 1)
    K = Kernel(Val(3), ω, h, self_weight(Val(3), ω, h), n + 1)
    astar, bstar, ratio = sparsify_stencils(K, n)

    @test length(astar) == 27 && length(bstar) == 27
    @test isapprox(norm(astar), 1; atol = 1e-12)
    @test ratio < 1e-3

    # the cubic lattice is invariant under the octahedral group, so the stencil
    # can only depend on the *type* of the offset
    byclass = Dict{Int,Vector{ComplexF64}}()
    for (p, t) in enumerate(offsets(Val(3)))
        push!(get!(byclass, sum(abs, t), ComplexF64[]), astar[p])
    end
    @test sort(collect(keys(byclass))) == [0, 1, 2, 3]
    @test length(byclass[1]) == 6 && length(byclass[2]) == 12 && length(byclass[3]) == 8
    for (_, v) in byclass
        @test maximum(abs, v .- v[1]) < 1e-8
    end
    # magnitudes decay from the centre outwards
    @test abs(byclass[0][1]) > abs(byclass[1][1]) > abs(byclass[2][1]) > abs(byclass[3][1])

    # α* + ω²β*⊙m is a discrete (−Δ − ω² + ω²m) up to one common scale, so the
    # row sums must satisfy Σα* = −ω²h²·(Σβ*/h²)
    scale = real(sum(bstar)) / h^2
    @test isapprox(real(sum(astar)), -scale * h^2 * ω^2; rtol = 0.02)

    # α* annihilates samples of any homogeneous Helmholtz solution
    for d in ((1.0, 0.0, 0.0), (0.4, 0.5, sqrt(1 - 0.16 - 0.25)))
        v = [cis(ω * h * sum(d[k] * t[k] for k in 1:3)) for t in offsets(Val(3))]
        @test abs(sum(astar .* v)) < 1e-3
    end
end

@testset "3D PML stencils" begin
    ω, h = 2π * 4, 1 / 32
    dσ0 = ntuple(_ -> (0.0, 0.0, 0.0), 3)
    γ = pml_stencil(ω, h, dσ0)
    @test length(γ) == 27
    @test isapprox(norm(γ), 1; atol = 1e-12)
    # 27 unknowns, 26 fitted waves: the null space is one-dimensional and the
    # annihilation of the fitted directions is exact
    for d in pmldirs(Val(3))
        v = [cis(ω * h * sum(d[k] * t[k] for k in 1:3)) for t in offsets(Val(3))]
        @test abs(sum(γ .* v)) < 1e-12
    end
    d = (0.3, 0.5, sqrt(1 - 0.09 - 0.25))
    v = [cis(ω * h * sum(d[k] * t[k] for k in 1:3)) for t in offsets(Val(3))]
    @test abs(sum(γ .* v)) < 1e-5

    # with stretching, the *modified* plane waves are annihilated exactly
    g = Grid{3}(31, 4)
    prof = SparsifyAndSweep.global_profile(g, ω, 12.0)
    dσ = (SparsifyAndSweep.dsigma(prof, h, -2), SparsifyAndSweep.dsigma(prof, h, 10),
          SparsifyAndSweep.dsigma(prof, h, 10))
    @test dσ[1] != (0.0, 0.0, 0.0) && dσ[2] == (0.0, 0.0, 0.0)
    γs = pml_stencil(ω, h, dσ)
    for d in pmldirs(Val(3))
        v = [cis(ω * sum(d[k] * complex(t[k] * h, dσ[k][t[k]+2]) for k in 1:3))
             for t in offsets(Val(3))]
        @test abs(sum(γs .* v)) < 1e-9
    end
end

@testset "3D sparse system and sweep" begin
    n, b, f = 15, 3, 2
    ω = 2π * f
    g = Grid{3}(n, b)
    m = perturbation(velocity(Val(3), :converging, n))
    K = Kernel(Val(3), ω, g.h, self_weight(Val(3), ω, g.h), n + 1)
    astar, bstar, _ = sparsify_stencils(K, n)
    H = build_H(g, ω, m, astar, bstar)

    @test size(H) == (nunk(g), nunk(g))
    @test nnz(H) <= 27 * nunk(g)
    rc = zeros(Int, nunk(g))
    for j in 1:size(H, 2), p in nzrange(H, j)
        rc[rowvals(H)[p]] += 1
    end
    @test rc[lin(g, (n ÷ 2, n ÷ 2, n ÷ 2))] == 27          # full 3^3 stencil
    @test minimum(rc) >= 8                                  # a corner keeps 2×2×2

    # an interior row is exactly α* + ω²β*⊙m
    i = (n ÷ 2, n ÷ 2, n ÷ 2)
    for (p, t) in enumerate(offsets(Val(3)))
        @test H[lin(g, i), lin(g, i .+ t)] ≈
              astar[p] + ω^2 * bstar[p] * mval(g, m, i .+ t)
    end

    S = rhs_operator(g, astar)
    @test size(S) == (nunk(g), nphys(g))

    F = sweep_setup(g, ω, m, H)
    @test dim(F) == 3
    @test nslices(F) > 3
    f0 = S * randn(ComplexF64, nphys(g))
    ue = H \ f0
    err = norm(sweep_solve(F, f0) - ue) / norm(ue)
    @info "3D sweep vs exact H⁻¹" err
    @test err < 0.3
    # linearity, and the outermost slices of the two fronts are exact
    f1 = randn(ComplexF64, nunk(g))
    f2 = randn(ComplexF64, nunk(g))
    @test norm(sweep_solve(F, 2 * f1 + f2) - (2 * sweep_solve(F, f1) + sweep_solve(F, f2))) /
          norm(sweep_solve(F, f1)) < 1e-10
    for s in (1, nslices(F))
        v = randn(ComplexF64, length(F.gidx[s]))
        w = SparsifyAndSweep.apply_T(F, s, v)
        @test norm(H[F.gidx[s], F.gidx[s]] * w - v) / norm(v) < 1e-9
    end
end

@testset "3D recursive sweep (levels = 2)" begin
    # §3.1: each quasi-2D slice subproblem is swept again along x₂ instead of
    # being factorised.  Same algorithm one level down, so the leaves must get
    # more numerous and smaller while the answer stays a good approximation.
    n, b, f = 15, 3, 2
    ω = 2π * f
    g = Grid{3}(n, b)
    m = perturbation(velocity(Val(3), :converging, n))
    K = Kernel(Val(3), ω, g.h, self_weight(Val(3), ω, g.h), n + 1)
    astar, bstar, _ = sparsify_stencils(K, n)
    H = build_H(g, ω, m, astar, bstar)
    f0 = rhs_operator(g, astar) * randn(ComplexF64, nphys(g))
    ue = H \ f0

    F1 = sweep_setup(g, ω, m, H; levels = 1)
    F2 = sweep_setup(g, ω, m, H; levels = 2)
    @test F1.levels == 1 && F2.levels == 2
    n1, mx1 = leafstats(F1)
    n2, mx2 = leafstats(F2)
    @test n2 > n1                       # every slice is cut again
    @test mx2 < mx1                     # into strictly smaller subproblems
    @test n1 == nslices(F1)             # levels = 1: one leaf per slice

    e1 = norm(sweep_solve(F1, f0) - ue) / norm(ue)
    e2 = norm(sweep_solve(F2, f0) - ue) / norm(ue)
    @info "3D recursive vs nonrecursive" e1 e2 n1 n2 mx1 mx2
    @test e2 < 0.4                      # still a usable preconditioner
    @test e2 < 3 * e1                   # and not much worse than levels = 1

    # the recursion must not break linearity or the outer front layout
    g1 = randn(ComplexF64, nunk(g))
    g2 = randn(ComplexF64, nunk(g))
    @test norm(sweep_solve(F2, 3 * g1 + g2) -
               (3 * sweep_solve(F2, g1) + sweep_solve(F2, g2))) /
          norm(sweep_solve(F2, g1)) < 1e-10
    @test F2.mid == F1.mid && F2.ranges == F1.ranges
    @test_throws ArgumentError sweep_setup(g, ω, m, H; levels = 4)
end

@testset "3D end-to-end solve" begin
    n, f = 31, 4
    ω = 2π * f
    P = LSProblem(n, ω, velocity(Val(3), :converging, n); b = 4)
    @test dim(P) == 3
    bvec = rhs(P, plane_wave(Val(3), n, ω))

    Md = SweepPreconditioner(P; mode = :direct)
    rd = solve(P, bvec; M = Md, tol = 1e-6, maxiter = 100)
    @test rd.converged && rd.iters <= 5

    for lv in (1, 2)
        Ms = SweepPreconditioner(P; mode = :sweep, levels = lv)
        rs = solve(P, bvec; M = Ms, tol = 1e-6, maxiter = 100)
        @info "3D iterations" levels = lv direct = rd.iters sweep = rs.iters
        @test rs.converged
        @test rs.iters <= 10
        @test rs.relres < 1e-5
        # every preconditioner must land on the same solution of the dense system
        @test norm(rs.u - rd.u) / norm(rd.u) < 1e-4
    end
end

@testset "3D Ying boundary stencils" begin
    n, f = 15, 2
    ω = 2π * f
    h = 1 / (n + 1)
    K = Kernel(Val(3), ω, h, self_weight(Val(3), ω, h), n + 1)
    bnd = boundary_stencils(K, n; buffer = 2)

    @test length(bnd.offsets) == 26                     # 3³ − 1 boundary cases
    # faces keep 18 points, edges 12, corners 8
    @test length(bnd.offsets[(1, 0, 0)]) == 18
    @test length(bnd.offsets[(1, 1, 0)]) == 12
    @test length(bnd.offsets[(1, 1, 1)]) == 8
    for (s, c) in bnd.coef
        @test length(c) == length(bnd.offsets[s])
        @test isapprox(norm(c), 1; atol = 1e-12)
    end
    # the octahedral symmetry of the lattice makes all six faces equivalent
    faces = [bnd.ratio[s] for s in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                    (0, 0, 1), (0, 0, -1))]
    @test maximum(abs, faces .- faces[1]) < 1e-10

    P = LSProblem(n, ω, velocity(Val(3), :converging, n); b = 3)
    bvec = rhs(P, plane_wave(Val(3), n, ω))
    M = SweepPreconditioner(P; mode = :direct, boundary = :ying)
    @test size(M.H, 1) == n^3                            # no PML extension
    r = solve(P, bvec; M = M, tol = 1e-8, maxiter = 100)
    @test r.converged && r.relres < 1e-6
end
