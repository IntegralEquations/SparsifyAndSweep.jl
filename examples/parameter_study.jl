# Where do the remaining GMRES iterations come from, and how sensitive is the
# preconditioner to its parameters?
#
#   julia --project=. examples/parameter_study.jl [freq] [field]
#
# Reports, at a fixed frequency:
#   * the iteration count with H inverted exactly (isolates sparsification error)
#   * the sweep error ‖T̃f − H⁻¹f‖/‖H⁻¹f‖ and iteration count for several
#     (b, C, nsamp) combinations
#   * the unpreconditioned GMRES baseline

using SparsifyAndSweep
using LinearAlgebra, SparseArrays, Printf

f = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
kind = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :converging
n = 8f - 1
ω = 2π * f
c = velocity(kind, n)
P = LSProblem(n, ω, c; b = 8)
bvec = rhs(P, plane_wave(n, ω))
println("f=$f n=$n N=$(n^2) field=$kind")

# 1. exact H⁻¹ as preconditioner -> isolates the sparsification error
Md = SweepPreconditioner(P; mode = :direct)
rd = solve(P, bvec; M = Md, tol = 1e-6)
@printf("H-direct (exact H^-1): iters=%d  setup=%.1fs\n", rd.iters, Md.tsetup)

# 2. sweep with varying b / C / nsamp
uex = Md.fact \ (Md.S * bvec)
for (bb, C, ns) in ((8, 8.0, 200), (8, 8.0, n), (8, 4.0, 200), (8, 12.0, 200),
                    (12, 8.0, 200))
    Pb = LSProblem(n, ω, c; b = bb)
    M = SweepPreconditioner(Pb; mode = :sweep, C = C, nsamp = ns)
    r = solve(Pb, bvec; M = M, tol = 1e-6)
    err = bb == 8 ? norm(sweep_solve(M.fact, M.S * bvec) - uex) / norm(uex) : NaN
    @printf("b=%2d C=%4.1f nsamp=%4d : iters=%2d  setup=%5.1fs  apply=%.2fs  sweepErr=%.2e  stencils=%d\n",
            bb, C, ns, r.iters, M.tsetup, r.tapply, err, M.nstencil)
    flush(stdout)
end

# 3. unpreconditioned baseline
ru = solve(P, bvec; tol = 1e-6, restart = 20, maxiter = 4000)
@printf("no preconditioner    : iters=%d converged=%s relres=%.2e  time=%.1fs\n",
        ru.iters, ru.converged, ru.relres, ru.tsolve)
