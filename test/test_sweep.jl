using SparsifyAndSweep
using SparsifyAndSweep: nslices, apply_T, collect_slice, SweepFactorization,
                        SweptSlab, DirectSlab, boxlen, leafstats
using LinearAlgebra, SparseArrays, Test, Random

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
    return SweepFactorization{2}(g, rs, gidx, mid, top, 0, 1, 1)
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
        f = randn(MersenneTwister(3), ComplexF64, nunk(g))   # seeded: the factor-2 margin is thin
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

@testset "the sweep axis is a free choice" begin
    # Nothing in the sweep singles out x₁.  On the transposed medium mᵀ, sweeping
    # H(mᵀ) along x₁ must reproduce the x₂ sweep of H(m), transposed.  The fitted
    # PML stencils are null vectors, unique only up to a unimodular factor, so the
    # PML rows of H(m) and H(mᵀ) differ by phases; that leaves every solution
    # unchanged as long as the right-hand side vanishes on those rows, which
    # `rhs_operator` guarantees.
    n, b, f = 47, 4, 6
    ω = 2π * f
    g = Grid(n, b)
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    m = perturbation(velocity(:multi, n))
    @test m != permutedims(m)                    # or the test would be vacuous
    H1 = build_H(g, ω, m, astar, bstar)
    H2 = build_H(g, ω, permutedims(m), astar, bstar)
    nt = ntot(g)
    swap(v) = vec(permutedims(reshape(v, nt, nt)))
    S = rhs_operator(g, astar)
    f1 = S * randn(MersenneTwister(5), ComplexF64, nphys(g))
    @test norm(H2 \ swap(f1) - swap(H1 \ f1)) / norm(H1 \ f1) < 1e-8    # the premise
    for fronts in (1, 2)
        F2 = sweep_setup(g, ω, m, H1; fronts = fronts, axis = 2)
        F1 = sweep_setup(g, ω, permutedims(m), H2; fronts = fronts, axis = 1)
        @test F2.axis == 2 && F1.axis == 1
        u2 = sweep_solve(F2, f1)
        @test norm(u2 - swap(sweep_solve(F1, swap(f1)))) / norm(u2) < 1e-8
    end
    # the slices of an x₂ sweep are x₂ slabs
    F2 = sweep_setup(g, ω, m, H1; axis = 2)
    for (s, r) in enumerate(F2.ranges)
        i2 = [div(k - 1, nt) - g.b for k in F2.gidx[s]]
        @test extrema(i2) == (first(r), last(r))
        @test length(F2.gidx[s]) == length(r) * nt
    end
end

@testset "one front sweeps either way" begin
    g, ω, m, H, astar = small_problem()
    ℓ = length(slice_ranges(g))
    Fp = sweep_setup(g, ω, m, H; fronts = 1, axis = 1)
    Fm = sweep_setup(g, ω, m, H; fronts = 1, axis = -1)
    @test Fp.mid == ℓ && Fm.mid == 1
    @test Fp.axis == 1 && Fm.axis == -1
    # the slice at the tail of the front carries no auxiliary PML and is exact:
    # slice 1 sweeping towards +x₁, slice ℓ sweeping towards -x₁
    for (F, s) in ((Fp, 1), (Fm, ℓ))
        v = randn(ComplexF64, length(F.gidx[s]))
        w = apply_T(F, s, v)
        @test norm(H[F.gidx[s], F.gidx[s]] * w - v) / norm(v) < 1e-9
    end
    f = rhs_operator(g, astar) * randn(ComplexF64, nphys(g))
    ue = H \ f
    for F in (Fp, Fm)
        @test norm(sweep_solve(F, f) - ue) / norm(ue) < 0.2
    end
    @test_throws ArgumentError sweep_setup(g, ω, m, H; axis = 3)
    @test_throws ArgumentError sweep_setup(g, ω, m, H; axis = 0)
end

"Constant-contrast disc of radius `R` about the centre of Ω, `m = m_in` inside."
function disc_medium(n, R, m_in)
    x = gridpoints(n)
    c = ones(Float64, n, n)
    for j in 1:n, i in 1:n
        (x[i] - 0.5)^2 + (x[j] - 0.5)^2 < R^2 && (c[i, j] = 1 / sqrt(1 - m_in))
    end
    return perturbation(c)
end

@testset "alternating sweep: bi-directional refinement" begin
    # Zepeda-Núñez & Zhao, Alg. 2: u = M₁ f + M₂ (f − H M₁ f).  On a disc a single
    # sweep misses the waves that turn back along its axis; the orthogonal sweep of
    # the residual recovers those that do not also turn back along the other axis.
    n, b, f0 = 63, 4, 8
    ω = 2π * f0
    g = Grid(n, b)
    k0 = self_weight(ω, g.h)
    astar, bstar, _ = sparsify_stencils(n, ω, g.h, k0)
    S = rhs_operator(g, astar)
    for (name, m) in (("lens", perturbation(velocity(:converging, n))),
                      ("disc", disc_medium(n, 0.2, -1.0)))
        H = build_H(g, ω, m, astar, bstar)
        f = S * randn(MersenneTwister(1), ComplexF64, nphys(g))
        ue = H \ f
        err(u) = norm(u - ue) / norm(ue)
        F1 = sweep_setup(g, ω, m, H; axis = 1)
        F2 = sweep_setup(g, ω, m, H; axis = 2)
        A = alternating_setup(g, ω, m, H; axes = (1, 2))
        @test A isa AlternatingSweep && length(A.F) == 2
        @test (A.F[1].axis, A.F[2].axis) == (1, 2)
        @test nstencils(A) == F1.nstencil + F2.nstencil
        u1, u2, u12 = sweep_solve(F1, f), sweep_solve(F2, f), alternating_solve(A, f)
        @test u12 ≈ u1 + sweep_solve(F2, f - H * u1)                  # the algorithm, literally
        e1, e2, e12 = err(u1), err(u2), err(u12)
        @info "single vs alternating sweep error" name e1 e2 e12
        @test e12 < 0.5 * min(e1, e2)
        # linear and deterministic
        f2 = S * randn(ComplexF64, nphys(g))
        a = 0.3 - 0.7im
        @test norm(alternating_solve(A, a * f + f2) - (a * u12 + alternating_solve(A, f2))) /
              norm(u12) < 1e-10
        @test alternating_solve(A, f) == u12
        # a one-entry list is a plain sweep; alternating the direction on one axis works too
        @test alternating_solve(alternating_setup(g, ω, m, H; axes = (1,)), f) == u1
        A11 = alternating_setup(g, ω, m, H; axes = (1, -1), fronts = 1)
        @test (A11.F[1].mid, A11.F[2].mid) == (nslices(A11.F[1]), 1)
        @test err(alternating_solve(A11, f)) < err(sweep_solve(A11.F[1], f))
        @test_throws ArgumentError alternating_setup(g, ω, m, H; axes = ())
    end
end
