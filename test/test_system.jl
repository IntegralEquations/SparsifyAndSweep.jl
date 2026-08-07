using SparsifyAndSweep
using SparsifyAndSweep: OFFSETS, in_Ih, in_I, lin, ilo, ihi, phys_indices
using LinearAlgebra, SparseArrays, Test

@testset "sparse system structure" begin
    n, b = 31, 4
    ω = 2π * 4
    g = Grid(n, b)
    c = velocity(:converging, n)
    m = perturbation(c)
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    H = build_H(g, ω, m, astar, bstar)

    @test size(H) == (nunk(g), nunk(g))
    @test nnz(H) <= 9 * nunk(g)

    # every row has at least four entries (a corner of Ihη keeps 2×2)
    rc = zeros(Int, nunk(g))
    for j in 1:size(H, 2), p in nzrange(H, j)
        rc[rowvals(H)[p]] += 1
    end
    @test minimum(rc) >= 4
    # interior rows (away from all boundaries) have the full 9-point stencil
    @test rc[lin(g, n ÷ 2, n ÷ 2)] == 9

    # an interior row must equal α* + ω²β*⊙m at its nine neighbours
    i1, i2 = n ÷ 2, n ÷ 2
    row = lin(g, i1, i2)
    for p in 1:9
        t1, t2 = OFFSETS[p]
        expect = astar[p] + ω^2 * bstar[p] * (in_I(g, i1 + t1, i2 + t2) ?
                                              m[i1+t1, i2+t2] : 0.0)
        @test H[row, lin(g, i1 + t1, i2 + t2)] ≈ expect
    end

    # a PML row must be a unit-norm modified-plane-wave stencil
    prow = lin(g, -b + 1, n ÷ 2)
    vals = [H[prow, lin(g, -b + 1 + t1, n ÷ 2 + t2)] for (t1, t2) in OFFSETS]
    @test isapprox(norm(vals), 1; atol = 1e-10)
end

@testset "right-hand-side operator" begin
    n, b = 31, 4
    ω = 2π * 4
    g = Grid(n, b)
    k0 = self_weight(ω, g.h)
    astar, _, _ = sparsify_stencils(n, ω, g.h, k0)
    S = rhs_operator(g, astar)
    @test size(S) == (nunk(g), nphys(g))

    r = randn(ComplexF64, nphys(g))
    f = S * r
    R = reshape(r, n, n)
    rp(i1, i2) = in_I(g, i1, i2) ? R[i1, i2] : zero(ComplexF64)
    for (i1, i2) in ((5, 7), (0, 0), (n + 1, n + 1), (1, n))
        @test f[lin(g, i1, i2)] ≈ sum(astar[p] * rp(i1 + OFFSETS[p][1], i2 + OFFSETS[p][2])
                                      for p in 1:9)
    end
    # zero outside Ω^h
    @test f[lin(g, -b, 0)] == 0
    @test f[lin(g, 0, ihi(g))] == 0
end

@testset "physical index map" begin
    g = Grid(11, 3)
    idx = phys_indices(g)
    @test length(idx) == nphys(g)
    @test allunique(idx)
    @test idx[1] == lin(g, 1, 1)
    @test idx[end] == lin(g, 11, 11)
end
