# Reproduce the 3D numerical tables of Liu & Ying (§3.2, Tables 5–8, the
# nonrecursive approach).
#
#   julia --project=. examples/tables3d.jl [maxfreq]
#
# Same code path as the 2D tables — the solver is parametrised on the dimension,
# so only `Val(3)` and `b = 4` change.  The grid is refined with the frequency so
# that h = λ/8 throughout (n = 8·ω/(2π) − 1), giving N = n³ unknowns.
#
# Costs in 3D: each slice is a quasi-2D subproblem, so the sweep costs
# O(b²N^{4/3}) to set up and O(bN log N) to apply, against O(N²) and O(N^{4/3})
# for a direct nested-dissection factorisation of the same sparse system.  That
# gap — not the 2D one — is what the method is really for.

using SparsifyAndSweep
using LinearAlgebra, Printf

const FIELDS = [
    (:converging, "(i)   converging lens"),
    (:diverging, "(ii)  diverging lens"),
    (:multi, "(iii) 256 narrow converging lenses"),
    (:random, "(iv)  random field"),
]

function run_table3d(kind::Symbol, label::AbstractString, freqs; b = 4, C = 12.0,
                     tol = 1e-6, restart = 20)
    println("\n", "="^78)
    println("velocity field ", label)
    println("="^78)
    @printf("%-7s %-9s %-9s %-10s %-10s %-8s %-6s %-10s\n",
            "ω/(2π)", "N", "Nunk", "T_setup", "T_apply", "T_solve", "N_iter", "relres")
    for f in freqs
        n = 8f - 1
        ω = 2π * f
        c = velocity(Val(3), kind, n)
        P = LSProblem(n, ω, c; b = b)
        bvec = rhs(P, plane_wave(Val(3), n, ω))
        M = SweepPreconditioner(P; C = C)
        r = solve(P, bvec; M = M, tol = tol, restart = restart, maxiter = 100)
        @printf("%-7d %-9s %-9d %-10.2f %-10.3f %-8.2f %-6d %-10.2e\n",
                f, "$(n)^3", nunk(P.g), r.tsetup, r.tapply, r.tsolve, r.iters, r.relres)
        flush(stdout)
        M = nothing
        GC.gc()
    end
end

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
freqs = [f for f in (4, 8, 16, 32) if f <= maxf]

println("Sparsify-and-sweep preconditioner, 3D")
println("h = λ/8, b = 4, C = 12, two sweeping fronts, GMRES(20), tol = 1e-6")
for (kind, label) in FIELDS
    run_table3d(kind, label, freqs)
end
