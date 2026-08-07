using SparsifyAndSweep
using LinearAlgebra, Test

@testset "LSProblem is a usable linear operator" begin
    # GMRES now comes from IterativeSolvers, so the package's job is to present
    # the operator and the preconditioner in the form it expects
    n, f = 31, 4
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    @test size(P) == (n^2, n^2)
    @test size(P, 1) == n^2
    @test eltype(P) == ComplexF64
    x = randn(ComplexF64, n^2)
    y = similar(x)
    mul!(y, P, x)
    @test y ≈ apply_A(P, x)
    @test P * x ≈ apply_A(P, x)

    # and the preconditioner must be ldiv!-able, in both forms
    M = SweepPreconditioner(P; mode = :direct)
    @test M \ x ≈ apply_M(M, x)
    z = similar(x)
    ldiv!(z, M, x)
    @test z ≈ apply_M(M, x)
    w = copy(x)
    ldiv!(M, w)
    @test w ≈ apply_M(M, x)
end

@testset "preconditioned solve matches the dense LU solution" begin
    n, f = 63, 8
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    uI = plane_wave(n, ω)
    bvec = rhs(P, uI)
    uref = dense_matrix(P) \ bvec

    for mode in (:direct, :sweep), fronts in (1, 2)
        mode === :direct && fronts == 2 && continue
        M = SweepPreconditioner(P; mode = mode, fronts = fronts)
        r = solve(P, bvec; M = M, tol = 1e-10, maxiter = 200)
        @test r.converged
        @test norm(r.u - uref) / norm(uref) < 1e-8
    end
end

@testset "H as a direct solver approximates the dense solution" begin
    # §4 of the paper: the sparsified system is itself a usable discretisation.
    n, f = 63, 8
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    bvec = rhs(P, plane_wave(n, ω))
    uref = dense_matrix(P) \ bvec
    M = SweepPreconditioner(P; mode = :direct)
    u = apply_M(M, bvec)
    err = norm(u - uref) / norm(uref)
    @info "H as a direct solver" err
    @test err < 0.02
end

@testset "low iteration counts" begin
    for kind in (:converging, :diverging, :multi, :random)
        for f in (8, 16)
            n = 8f - 1
            ω = 2π * f
            P = LSProblem(n, ω, velocity(kind, n); b = 8)
            bvec = rhs(P, plane_wave(n, ω))
            M = SweepPreconditioner(P)
            r = solve(P, bvec; M = M, tol = 1e-6, restart = 20, maxiter = 100)
            @info "iterations" kind f r.iters r.relres
            @test r.converged
            @test r.iters <= 12
            @test r.relres < 1e-5
        end
    end
end

@testset "unpreconditioned GMRES is much slower, and degrades with ω" begin
    # The point of the preconditioner is not just that it wins at one size, but
    # that the plain iteration count grows with ω while the preconditioned one
    # does not.  (The gap keeps widening: at ω/2π = 64 it is 528 against 6, see
    # examples/preconditioner_benefit.jl.)
    res = map((8, 16)) do f
        n = 8f - 1
        ω = 2π * f
        P = LSProblem(n, ω, velocity(:multi, n); b = 8)
        bvec = rhs(P, plane_wave(n, ω))
        rp = solve(P, bvec; M = SweepPreconditioner(P), tol = 1e-6, maxiter = 100)
        ru = solve(P, bvec; tol = 1e-6, restart = 20, maxiter = 2000)
        @info "preconditioned vs plain GMRES" f rp.iters ru.iters ru.converged
        @test ru.converged && rp.converged
        @test 3 * rp.iters <= ru.iters
        (pre = rp.iters, plain = ru.iters)
    end
    @test res[2].plain > res[1].plain                 # plain degrades with ω
    @test res[2].pre <= res[1].pre + 2                # preconditioned does not
end
