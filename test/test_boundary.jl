using SparsifyAndSweep
using SparsifyAndSweep: OFFSETS, kernel_table, ktab, in_I
using LinearAlgebra, SparseArrays, Test

@testset "clipping signature and neighbourhoods" begin
    n = 10
    @test clip_signature(n, 5, 5) == (0, 0)
    @test clip_signature(n, 1, 5) == (-1, 0)
    @test clip_signature(n, n, 5) == (1, 0)
    @test clip_signature(n, 5, 1) == (0, -1)
    @test clip_signature(n, n, n) == (1, 1)
    @test clip_signature(n, 1, n) == (-1, 1)

    @test clipped_offsets(0, 0) == OFFSETS                       # 9 points
    for s in ((1, 0), (-1, 0), (0, 1), (0, -1))
        @test length(clipped_offsets(s...)) == 6                 # edge
    end
    for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        @test length(clipped_offsets(s...)) == 4                 # corner
    end
    # the clipped neighbourhood must not step outside Ω
    @test all(t -> t[1] <= 0, clipped_offsets(1, 0))
    @test all(t -> t[1] >= 0, clipped_offsets(-1, 0))
    @test all(t -> t[1] <= 0 && t[2] <= 0, clipped_offsets(1, 1))
end

@testset "far sets match Ying's Eₙ and Cₙ" begin
    n, b = 63, 2
    # Eₙ = {j : -n < j₁ < -b, -n < j₂ < n} for the +x₁ edge
    @test far_range(1, n, b) == (-n+1):(-b-1)
    @test far_range(0, n, b) == (-n+1):(n-1)
    # mirrored for the -x₁ edge
    @test far_range(-1, n, b) == (b+1):(n-1)
    # the far set never meets the (clipped) neighbourhood, so no constraint is
    # dropped and none is doubly imposed
    for s in (1, -1)
        r = far_range(s, n, b)
        @test all(j -> abs(j) > 1, r)
    end
end

@testset "boundary stencils" begin
    n, f = 63, 8
    ω = 2π * f
    h = 1 / (n + 1)
    k0 = self_weight(ω, h)
    bnd = boundary_stencils(n, ω, h, k0; buffer = 2)

    @test length(bnd.offsets) == 8 && length(bnd.coef) == 8
    for (s, c) in bnd.coef
        @test length(c) == length(bnd.offsets[s])
        @test isapprox(norm(c), 1; atol = 1e-12)
    end

    # the lattice and the far sets are dihedrally symmetric, so all four edges
    # must give the same singular-value ratio, and likewise all four corners
    edges = [bnd.ratio[s] for s in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    corners = [bnd.ratio[s] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    @test maximum(abs, edges .- edges[1]) < 1e-10
    @test maximum(abs, corners .- corners[1]) < 1e-10

    # one-sided annihilation is genuinely harder than the interior problem
    _, _, interior_ratio = sparsify_stencils(n, ω, h, k0)
    @test interior_ratio < edges[1]
    @test edges[1] < 1e-2

    # the defining property: B kills the Green's function of any source at least
    # `buffer` layers inside the domain
    T = kernel_table(ω, h, n + 1, k0)
    for s in ((1, 0), (1, 1))
        o = bnd.offsets[s]
        c = bnd.coef[s]
        for j in ((-20, 5), (-40, -13), (-7, 30), (-15, -25))
            (s == (1, 1) && j[2] > -3) && continue        # must be in Cₙ
            v = [ktab(T, n + 1, t[1] - j[1], t[2] - j[2]) for t in o]
            @test abs(sum(c .* v)) / norm(v) < 5e-2
        end
    end
end

@testset "Ying system structure" begin
    n, b = 31, 4
    ω = 2π * 4
    g = Grid(n, b)
    m = perturbation(velocity(:converging, n))
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    bnd = boundary_stencils(n, ω, g.h, k0; buffer = 2)
    H = build_H_ying(g, ω, m, astar, bstar, bnd)

    @test size(H) == (n^2, n^2)                       # no PML extension
    @test size(H, 1) < nunk(g)                        # smaller than the PML system

    rc = zeros(Int, n^2)
    for j in 1:size(H, 2), p in nzrange(H, j)
        rc[rowvals(H)[p]] += 1
    end
    lidx(i1, i2) = i1 + (i2 - 1) * n
    @test rc[lidx(n ÷ 2, n ÷ 2)] == 9                 # interior
    @test rc[lidx(1, n ÷ 2)] == 6                     # edge
    @test rc[lidx(n, n)] == 4                         # corner

    # interior rows must be exactly the α/β stencil
    i1, i2 = n ÷ 2, n ÷ 2
    for p in 1:9
        t1, t2 = OFFSETS[p]
        @test H[lidx(i1, i2), lidx(i1 + t1, i2 + t2)] ≈
              astar[p] + ω^2 * bstar[p] * m[i1+t1, i2+t2]
    end
    # boundary rows must be exactly B, with no ω²βm term
    for p in eachindex(bnd.offsets[(1, 1)])
        t1, t2 = bnd.offsets[(1, 1)][p]
        @test H[lidx(n, n), lidx(n + t1, n + t2)] ≈ bnd.coef[(1, 1)][p]
    end

    S = rhs_operator_ying(g, astar, bnd)
    @test size(S) == (n^2, n^2)
    r = randn(ComplexF64, n^2)
    fv = S * r
    R = reshape(r, n, n)
    @test fv[lidx(i1, i2)] ≈ sum(astar[p] * R[i1+OFFSETS[p][1], i2+OFFSETS[p][2]]
                                 for p in 1:9)
end

@testset "Ying preconditioner solves the dense system" begin
    n, f = 63, 8
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    bvec = rhs(P, plane_wave(n, ω))
    uref = dense_matrix(P) \ bvec

    M = SweepPreconditioner(P; mode = :direct, boundary = :ying)
    @test M.boundary === :ying
    @test size(M.H, 1) == n^2
    r = solve(P, bvec; M = M, tol = 1e-10, maxiter = 200)
    @test r.converged
    @test norm(r.u - uref) / norm(uref) < 1e-8

    # as a direct solver it is usable but distinctly worse than the PML surrogate
    eying = norm(apply_M(M, bvec) - uref) / norm(uref)
    epml = norm(apply_M(SweepPreconditioner(P; mode = :direct), bvec) - uref) / norm(uref)
    @info "boundary treatment as a direct solver" epml eying
    @test eying < 0.1
    @test epml < eying
end

@testset "boundary option validation" begin
    n, ω = 31, 2π * 4
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    @test_throws ArgumentError SweepPreconditioner(P; boundary = :ying, mode = :sweep)
    @test_throws ArgumentError SweepPreconditioner(P; boundary = :nonsense)
    @test_throws ArgumentError SweepPreconditioner(P; mode = :nonsense)
end
