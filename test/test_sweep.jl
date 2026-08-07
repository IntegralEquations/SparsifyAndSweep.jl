using SparsifyAndSweep
using SparsifyAndSweep: nslices, apply_T, collect_slice, SweepFactorization,
                        SweptSlab, DirectSlab, boxlen, leafstats
using LinearAlgebra, SparseArrays, Test

"""
Replace every approximate solution operator `T̃_[i]` by the exact Schur
complement inverse, keeping everything else in `sweep_solve` untouched.  The
twisted factorisation is exact in that case, so `sweep_solve` must then agree
with `H \\ f` to round-off — an unconditional check on the forward/backward
substitution, independent of the PML.
"""
function exact_factorization(g, H, mid)
    rs = slice_ranges(g)
    ℓ = length(rs)
    gidx = [collect_slice(g, r) for r in rs]
    blk(i, j) = Matrix(H[gidx[i], gidx[j]])
    S = Vector{Matrix{ComplexF64}}(undef, ℓ)
    if mid > 1
        S[1] = blk(1, 1)
        for i in 2:(mid-1)
            S[i] = blk(i, i) - blk(i, i - 1) * (S[i-1] \ blk(i - 1, i))
        end
    end
    if mid < ℓ
        S[ℓ] = blk(ℓ, ℓ)
        for i in (ℓ-1):-1:(mid+1)
            S[i] = blk(i, i) - blk(i, i + 1) * (S[i+1] \ blk(i + 1, i))
        end
    end
    Sm = blk(mid, mid)
    mid > 1 && (Sm -= blk(mid, mid - 1) * (S[mid-1] \ blk(mid - 1, mid)))
    mid < ℓ && (Sm -= blk(mid, mid + 1) * (S[mid+1] \ blk(mid + 1, mid)))
    S[mid] = Sm

    sub = SparsifyAndSweep.SlabSolver[DirectSlab(lu(sparse(S[i]))) for i in 1:ℓ]
    loc = [collect(1:length(gidx[i])) for i in 1:ℓ]
    naux = [length(gidx[i]) for i in 1:ℓ]
    Hlow = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, ℓ)
    Hup = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, ℓ)
    for i in 1:ℓ
        i > 1 && (Hlow[i] = H[gidx[i], gidx[i-1]])
        i < ℓ && (Hup[i] = H[gidx[i], gidx[i+1]])
    end
    top = SweptSlab(1, mid, gidx, loc, naux, sub, Hlow, Hup)
    return SweepFactorization{2}(g, rs, gidx, mid, top, 0, 1)
end

# Build a small problem once and reuse it.
function small_problem(; n = 47, b = 4, f = 6, kind = :converging)
    ω = 2π * f
    g = Grid(n, b)
    m = perturbation(velocity(kind, n))
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    H = build_H(g, ω, m, astar, bstar)
    return g, ω, m, H, astar
end

@testset "sweep with a single slice is exact" begin
    # ntot < 4b forces slice_ranges to return one slice, for which the auxiliary
    # problem *is* the whole system: the sweep must then reproduce H \ f exactly.
    n, b = 11, 8      # ntot = 29 < 4b = 32
    ω = 2π * 3
    g = Grid(n, b)
    @test length(slice_ranges(g)) == 1
    m = perturbation(velocity(:converging, n))
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    H = build_H(g, ω, m, astar, bstar)
    F = sweep_setup(g, ω, m, H)
    @test nslices(F) == 1
    f = randn(ComplexF64, nunk(g))
    @test norm(sweep_solve(F, f) - H \ f) / norm(H \ f) < 1e-10
end

@testset "sweep approximates H⁻¹" begin
    g, ω, m, H, astar = small_problem()
    S = rhs_operator(g, astar)
    r = randn(ComplexF64, nphys(g))
    f = S * r
    ue = H \ f
    for fronts in (1, 2)
        F = sweep_setup(g, ω, m, H; fronts = fronts)
        @test nslices(F) > 3
        err = norm(sweep_solve(F, f) - ue) / norm(ue)
        @info "sweep vs exact H⁻¹" fronts err
        @test err < 0.2                   # a preconditioner, not a solver
    end
end

@testset "twisted substitution is exact with exact Schur complements" begin
    g, ω, m, H, _ = small_problem()
    ℓ = length(slice_ranges(g))
    f = randn(ComplexF64, nunk(g))
    ue = H \ f
    for mid in (1, 2, cld(ℓ, 2), ℓ - 1, ℓ)      # every front layout, not just the default
        F = exact_factorization(g, H, mid)
        @test norm(sweep_solve(F, f) - ue) / norm(ue) < 1e-9
    end
end

@testset "sweep converges to H⁻¹ as the moving PML improves" begin
    # The only error in the sweep is the moving-PML approximation of the Schur
    # complements, so widening the PML must drive it to zero.  A bug in the
    # (twisted) forward/backward substitution would instead leave a floor.
    n, ω = 79, 2π * 10
    errs = map((4, 12)) do b
        g = Grid(n, b)
        m = perturbation(velocity(:converging, n))
        k0 = self_weight(ω, g.h)
        astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
        H = build_H(g, ω, m, astar, bstar)
        f = randn(ComplexF64, nunk(g))
        ue = H \ f
        map((1, 2)) do fronts
            F = sweep_setup(g, ω, m, H; fronts = fronts)
            norm(sweep_solve(F, f) - ue) / norm(ue)
        end
    end
    @info "sweep error vs PML width" b4 = errs[1] b12 = errs[2]
    for k in 1:2
        @test errs[2][k] < errs[1][k] / 2
    end
end

@testset "levels is a no-op in 2D beyond the first axis" begin
    # in 2D the slice subproblems are already quasi-1D, so sweeping them again
    # just wraps each in a one-slice sweep: same answer, more leaves
    g, ω, m, H, _ = small_problem()
    f = randn(ComplexF64, nunk(g))
    F1 = sweep_setup(g, ω, m, H; levels = 1)
    F2 = sweep_setup(g, ω, m, H; levels = 2)
    n1, _ = leafstats(F1)
    n2, _ = leafstats(F2)
    @test n1 == nslices(F1)
    @test n2 >= n1
    @test norm(sweep_solve(F2, f) - sweep_solve(F1, f)) / norm(sweep_solve(F1, f)) < 0.5
    @test_throws ArgumentError sweep_setup(g, ω, m, H; levels = 3)
end

@testset "front layout" begin
    g, ω, m, H, _ = small_problem()
    ℓ = length(slice_ranges(g))
    @test sweep_setup(g, ω, m, H; fronts = 1).mid == ℓ
    @test sweep_setup(g, ω, m, H; fronts = 2).mid == cld(ℓ, 2)
    @test_throws ArgumentError sweep_setup(g, ω, m, H; fronts = 3)
end

@testset "T̃ of the outermost slices is exact" begin
    # A slice at the tail of its front has no auxiliary PML, so its subproblem
    # is the corresponding diagonal block of H itself.
    g, ω, m, H, _ = small_problem()
    for fronts in (1, 2)
        F = sweep_setup(g, ω, m, H; fronts = fronts)
        ℓ = nslices(F)
        exact = fronts == 1 ? (1,) : (1, ℓ)          # fronts = 1 pads slice ℓ
        for s in exact
            v = randn(ComplexF64, length(F.gidx[s]))
            w = apply_T(F, s, v)
            @test norm(H[F.gidx[s], F.gidx[s]] * w - v) / norm(v) < 1e-9
        end
    end
end

@testset "sweep is linear and deterministic" begin
    g, ω, m, H, _ = small_problem()
    for fronts in (1, 2)
        F = sweep_setup(g, ω, m, H; fronts = fronts)
        f1 = randn(ComplexF64, nunk(g))
        f2 = randn(ComplexF64, nunk(g))
        a = 0.3 - 0.7im
        @test norm(sweep_solve(F, a * f1 + f2) -
                   (a * sweep_solve(F, f1) + sweep_solve(F, f2))) /
              norm(sweep_solve(F, f1)) < 1e-10
        @test sweep_solve(F, f1) == sweep_solve(F, f1)
    end
end
