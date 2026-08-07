# Reproduce the 2D numerical tables of Liu & Ying (§2.5).
#
#   julia --project=. examples/tables.jl [maxfreq]
#
# For each of the four test velocity fields and each angular frequency
# ω/(2π) ∈ {16, 32, …, maxfreq} it reports setup / apply / solve times, the
# GMRES iteration count and the final relative residual.  The grid is refined
# with the frequency so that h = λ/8 throughout (n = 8·ω/(2π) − 1).

using SparsifyAndSweep
using LinearAlgebra, Printf

const FIELDS = [
    (:converging, "(i)   converging lens"),
    (:diverging, "(ii)  diverging lens"),
    (:multi, "(iii) 32 narrow converging lenses"),
    (:random, "(iv)  random field"),
]

function run_table(kind::Symbol, label::AbstractString, freqs; b = 8, C = 12.0,
                   tol = 1e-6, restart = 20, fronts = 2)
    println("\n", "="^78)
    println("velocity field ", label)
    println("="^78)
    @printf("%-7s %-9s %-10s %-10s %-8s %-6s %-10s\n",
            "ω/(2π)", "N", "T_setup", "T_apply", "T_solve", "N_iter", "relres")
    for f in freqs
        n = 8f - 1
        ω = 2π * f
        c = velocity(kind, n)
        P = LSProblem(n, ω, c; b = b)
        uI = plane_wave(n, ω)
        bvec = rhs(P, uI)
        M = SweepPreconditioner(P; mode = :sweep, C = C, fronts = fronts)
        r = solve(P, bvec; M = M, tol = tol, restart = restart)
        @printf("%-7d %-9s %-10.2f %-10.3f %-8.2f %-6d %-10.2e\n",
                f, "$(n)^2", r.tsetup, r.tapply, r.tsolve, r.iters, r.relres)
        flush(stdout)
    end
end

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
freqs = [f for f in (16, 32, 64, 128, 256) if f <= maxf]

println("Sparsify-and-sweep preconditioner, 2D")
println("h = λ/8, b = 8, C = 12, two sweeping fronts, GMRES(20), tol = 1e-6")
println("lens widths: σ = ", SparsifyAndSweep.default_width(:converging),
        " (single lens), σ = ", SparsifyAndSweep.default_width(:multi), " (32 lenses)")
for (kind, label) in FIELDS
    run_table(kind, label, freqs)
end
