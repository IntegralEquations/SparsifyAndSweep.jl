# Head-to-head: GMRES with and without the sparsify-and-sweep preconditioner,
# plus a breakdown of where the remaining iterations come from.
#
#   julia --project=. examples/preconditioner_benefit.jl [maxfreq]
#
# Three preconditioners are compared at each frequency:
#   none    — plain GMRES(20) on the dense Lippmann–Schwinger system
#   H-exact — the sparsified system H inverted exactly by a sparse LU.  This
#             isolates the error of the *sparsification*.
#   sweep   — the O(N) moving-PML sweeping factorisation of H (the real method)

using SparsifyAndSweep
using LinearAlgebra, Printf

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
freqs = [f for f in (8, 16, 32, 64) if f <= maxf]
kind = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :converging

println("velocity field: $kind      h = λ/8, b = 8, GMRES(20), tol = 1e-6")
@printf("\n%-7s %-9s | %-8s %-8s | %-8s | %-8s %-8s %-8s %-9s\n",
        "ω/(2π)", "N", "none", "T_none", "H-exact", "sweep", "T_setup", "T_solve", "sweepErr")
println("-"^88)

for f in freqs
    n = 8f - 1
    ω = 2π * f
    P = LSProblem(n, ω, velocity(kind, n); b = 8)
    bvec = rhs(P, plane_wave(n, ω))

    ru = solve(P, bvec; tol = 1e-6, restart = 20, maxiter = 4000)
    Md = SweepPreconditioner(P; mode = :direct)
    rd = solve(P, bvec; M = Md, tol = 1e-6)
    Ms = SweepPreconditioner(P; mode = :sweep)
    rs = solve(P, bvec; M = Ms, tol = 1e-6)

    # how well the sweep approximates H⁻¹ on this right-hand side
    fv = Ms.S * bvec
    err = norm(sweep_solve(Ms.fact, fv) - (Md.fact \ fv)) / norm(Md.fact \ fv)

    @printf("%-7d %-9s | %-8d %-8.1f | %-8d | %-8d %-8.2f %-8.2f %-9.2e\n",
            f, "$(n)^2", ru.iters, ru.tsolve, rd.iters, rs.iters,
            rs.tsetup, rs.tsolve, err)
    flush(stdout)
end
