# L-sweeps: CDD geometry, accuracy against the exact sparse solve, determinism
# across worker counts, and use as the preconditioner of the dense system.
using Test
using SparsifyAndSweep
using SparsifyAndSweep: cdd_ranges, node_exists, DIRS, message_tag
using LinearAlgebra, SparseArrays, Random

@testset "L-sweeps" begin
    @testset "CDD partition" begin
        rs = cdd_ranges(-8:136, 4)
        @test length(rs) == 4
        @test first(rs[1]) == -8 && last(rs[end]) == 136
        @test all(last(rs[k]) + 1 == first(rs[k+1]) for k in 1:3)
        @test sum(length, rs) == 145
        @test cdd_ranges(1:5, 10) == [k:k for k in 1:5]
        # receive directions: interior boxes receive from all eight, corners from three
        @test count(k -> node_exists(2, 2, k, 3, 3), 1:8) == 8
        @test count(k -> node_exists(1, 1, k, 3, 3), 1:8) == 3
        @test count(k -> node_exists(1, 2, k, 3, 3), 1:8) == 5
        @test node_exists(1, 1, 0, 3, 3)
        tags = [message_tag(i, j, k, m, 3, 4) for i in 1:3, j in 1:4, k in 1:8, m in 1:3]
        @test allunique(tags) && minimum(tags) == 0
        @test row_partition(5, 3, 2) == [mod(i - 1, 2) for i in 1:5, j in 1:3]
    end

    nn = 16
    n, ω, b = 8nn - 1, 2π * nn, 8
    g = Grid{2}(n, b)
    K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
    astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
    rng = MersenneTwister(7)
    f = zeros(ComplexF64, nunk(g))
    f[SparsifyAndSweep.phys_indices(g)] = randn(rng, ComplexF64, n^2)   # sources everywhere

    @testset "accuracy against the exact sparse solve" begin
        for (field, tol) in ((:constant, 2e-3), (:converging, 5e-2))
            m = perturbation(velocity(field, n))
            H = build_H(g, ω, m, astar, bstar; C = 12.0)
            u = lu(H) \ f
            err(boxes, pad) = (F = lsweep_setup(g, ω, m, astar, bstar; boxes = boxes, pad = pad,
                                                nworkers = 1);
                               norm(lsweep_solve(F, f) - u) / norm(u))
            for boxes in ((2, 2), (3, 3))
                @test err(boxes, 8) < tol
            end
            if field === :constant
                @test err((3, 3), 16) < err((3, 3), 8)       # thicker absorber, smaller error
            end
        end
    end

    @testset "shifted checkerboard" begin
        rs = cdd_ranges(-8:136, 4; shift = 18)
        @test length(rs) == 5
        @test first(rs[1]) == -8 && last(rs[end]) == 136
        @test all(last(rs[k]) + 1 == first(rs[k+1]) for k in 1:4)
        @test length(rs[1]) == 18                                   # a part-box in front
        @test first.(rs[2:end]) == first.(cdd_ranges(-8:136, 4)) .+ 18   # every cut moved up
        @test_throws ArgumentError cdd_ranges(-8:136, 4; shift = 40)
        m = perturbation(velocity(:converging, n))
        H = build_H(g, ω, m, astar, bstar; C = 12.0)
        u = lu(H) \ f
        F0 = lsweep_setup(g, ω, m, astar, bstar; boxes = (2, 2), nworkers = 1)
        F1 = lsweep_setup(g, ω, m, astar, bstar; boxes = (2, 2), shift = (0.5, 0.5), nworkers = 1)
        @test (F1.q, F1.r) == (3, 3)
        e0 = norm(lsweep_solve(F0, f) - u) / norm(u)
        e1 = norm(lsweep_solve(F1, f) - u) / norm(u)
        @test e1 < 5e-2                                             # as good a board as the first
        A = AlternatingSweep{2}(H, [F0, F1])
        u01 = alternating_solve(A, f)
        @test u01 ≈ lsweep_solve(F0, f) + lsweep_solve(F1, f - H * lsweep_solve(F0, f))
        e01 = norm(u01 - u) / norm(u)
        @info "L-sweep, shifted L-sweep, alternated" e0 e1 e01
        @test e01 < 0.5 * min(e0, e1)
        @test SparsifyAndSweep.sweeps_per_solve(A) == 2
        @test nstencils(A) == F0.nstencil + F1.nstencil
        @test_throws ArgumentError lsweep_setup(g, ω, m, astar, bstar; boxes = (2, 2), shift = (1.0, 0.0))
    end

    @testset "one row or one column degenerates gracefully" begin
        m = perturbation(velocity(:converging, n))
        H = build_H(g, ω, m, astar, bstar; C = 12.0)
        u = lu(H) \ f
        for boxes in ((1, 3), (3, 1), (1, 1))
            F = lsweep_setup(g, ω, m, astar, bstar; boxes = boxes, nworkers = 1)
            # one box is the solve of H itself, exact up to the single-precision factors
            @test norm(lsweep_solve(F, f) - u) / norm(u) < (boxes == (1, 1) ? 1e-4 : 5e-2)
        end
        F = lsweep_setup(g, ω, m, astar, bstar; boxes = (1, 1), nworkers = 1, factors = :umfpack)
        @test norm(lsweep_solve(F, f) - u) / norm(u) < 1e-10  # and exact with UMFPACK's own
    end

    @testset "sequential and threaded runs agree bitwise" begin
        m = perturbation(velocity(:converging, n))
        F1 = lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3), nworkers = 1)
        F2 = lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3),
                          nworkers = max(2, Threads.nthreads()))
        u1 = lsweep_solve(F1, f)
        u2 = lsweep_solve(F2, f)
        @test u1 == u2
        @test lsweep_solve(F2, f) == u2                        # and repeatable
    end

    @testset "compact single-precision factors" begin
        m = perturbation(velocity(:converging, n))
        Fu = lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3), nworkers = 1)
        Fc = lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3), nworkers = 1, factors = :compact)
        Fc2 = lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3), factors = :compact,
                           nworkers = max(2, Threads.nthreads()))
        x = randn(MersenneTwister(7), ComplexF64, Fu.boxes[(1, 1)].geom.n)
        xu = Fu.boxes[(1, 1)].fact \ x
        @test norm(Fc.boxes[(1, 1)].fact \ x - xu) / norm(xu) < 1e-5   # float32 factors, double accumulation
        uu, uc = lsweep_solve(Fu, f), lsweep_solve(Fc, f)
        @test norm(uc - uu) / norm(uu) < 1e-4
        @test lsweep_solve(Fc2, f) == uc                        # bitwise across worker counts
        @test_throws ArgumentError lsweep_setup(g, ω, m, astar, bstar; boxes = (3, 3), factors = :foo)
    end

    @testset "three dimensions: beams" begin
        n3, b3 = 15, 4
        ω3 = 2π * 2
        g3 = Grid{3}(n3, b3)
        m3 = perturbation(velocity(Val(3), :converging, n3))
        K3 = Kernel(Val(3), ω3, g3.h, self_weight(Val(3), ω3, g3.h), n3 + 1)
        a3, b3s, _ = sparsify_stencils(K3, n3; farfield = n3)
        f3 = zeros(ComplexF64, nunk(g3)); f3[SparsifyAndSweep.phys_indices(g3)] .= 1
        u3 = lu(build_H(g3, ω3, m3, a3, b3s; C = 12.0)) \ f3
        F3 = lsweep_setup(g3, ω3, m3, a3, b3s; boxes = (2, 2), pad = 4, nworkers = 1)
        bx = F3.boxes[(1, 1)]
        @test length(bx.geom.ext) == 3 && bx.geom.ext[3] == ilo(g3):ihi(g3)   # a beam spans axis 3
        v3 = lsweep_solve(F3, f3)
        @test norm(v3 - u3) / norm(u3) < 5e-2
        F3t = lsweep_setup(g3, ω3, m3, a3, b3s; boxes = (2, 2), pad = 4,
                           nworkers = max(2, Threads.nthreads()))
        @test lsweep_solve(F3t, f3) == v3                       # bitwise across worker counts
        P3 = LSProblem(n3, ω3, velocity(Val(3), :converging, n3); b = b3)
        r3 = solve(P3, rhs(P3, plane_wave(Val(3), n3, ω3));
                   M = SweepPreconditioner(P3; mode = :lsweep, boxes = (2, 2), pad = 4))
        @test r3.converged && r3.relres < 1e-6 && r3.iters <= 6
    end

    @testset "as the preconditioner of the dense system" begin
        c = velocity(:converging, n)
        P = LSProblem(n, ω, c; b = b)
        bb = rhs(P, plane_wave(n, ω))
        Ms = SweepPreconditioner(P; mode = :sweep)
        Ml = SweepPreconditioner(P; mode = :lsweep, boxes = (3, 3))
        rs = solve(P, bb; M = Ms)
        rl = solve(P, bb; M = Ml)
        @test rl.converged && rl.relres < 1e-6
        @test rl.iters <= rs.iters + 4
        Mc = SweepPreconditioner(P; mode = :lsweep, boxes = (3, 3), factors = :compact)
        rc = solve(P, bb; M = Mc)
        @test rc.converged && rc.relres < 1e-6
        @test rc.iters <= rl.iters + 1                          # same preconditioner to 1e-4
        # two-level: a few inner GMRES steps on H bring the outer count towards the exact-H floor (3)
        Mi = SweepPreconditioner(P; mode = :lsweep, boxes = (3, 3), inner = 8, inner_tol = 1e-4)
        ri = solve(P, bb; M = Mi)
        @test ri.converged && ri.relres < 1e-6
        @test ri.iters <= 5
        # the shifted alternation: two L-sweeps per application, fewer outer iterations
        Ma = SweepPreconditioner(P; mode = :lsweep, boxes = (3, 3), shifts = ((0, 0), (0.5, 0.5)))
        @test Ma.fact isa AlternatingSweep && (Ma.fact.F[2].q, Ma.fact.F[2].r) == (4, 4)
        Ma.nsweeps[] = 0
        apply_M(Ma, bb)
        @test Ma.nsweeps[] == 2
        ra = solve(P, bb; M = Ma)
        @test ra.converged && ra.relres < 1e-6
        @test ra.iters <= rl.iters
        @test_throws ArgumentError SweepPreconditioner(P; mode = :foo)
        @test_throws ArgumentError SweepPreconditioner(P; mode = :lsweep, boxes = (40, 40))
        @test_throws ArgumentError SweepPreconditioner(P; mode = :lsweep, shifts = ())
    end
end
