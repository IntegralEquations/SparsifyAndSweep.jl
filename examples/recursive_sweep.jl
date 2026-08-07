# Recursive vs nonrecursive sweep in 3D (§3.1, "the recursive approach", ref. [20]).
#
#   julia --project=. examples/recursive_sweep.jl [maxfreq]
#
# levels = 1  each quasi-2D slice subproblem is factorised directly:
#             O(b³n³) per slice, O(b²N^{4/3}) overall
# levels = 2  each slice is swept again along x₂ into quasi-1D subproblems:
#             O(b⁴N) setup, O(b²N) application
#
# The paper reports that the recursive approach costs "zero or one more
# iteration" than the nonrecursive one; the question is how much setup it saves.

using SparsifyAndSweep
using SparsifyAndSweep: leafstats
using LinearAlgebra, Printf

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
freqs = [f for f in (4, 8, 16) if f <= maxf]

@printf("%-7s %-9s %-8s %-9s %-9s %-6s %-11s %-8s %-9s\n",
        "ω/(2π)", "N", "levels", "T_setup", "T_apply", "iters", "sweepErr",
        "leaves", "max leaf")
for f in freqs
    n = 8f - 1
    ω = 2π * f
    P = LSProblem(n, ω, velocity(Val(3), :converging, n); b = 4)
    bvec = rhs(P, plane_wave(Val(3), n, ω))
    # Reference for the sweepErr column: the same H inverted exactly.  In 3D that
    # is an O(N²) nested-dissection factorisation — far more expensive than the
    # sweeps being measured — so only do it while it is affordable, and otherwise
    # report the recursive sweep against the nonrecursive one.
    exact = nunk(P.g) <= 150_000
    local fv, uex, ref1
    if exact
        Md = SweepPreconditioner(P; mode = :direct)
        fv = Md.S * bvec
        uex = Md.fact \ fv
        Md = nothing
        GC.gc()
    end
    for lv in (1, 2)
        M = SweepPreconditioner(P; levels = lv)
        r = solve(P, bvec; M = M, tol = 1e-6, maxiter = 100)
        nl, mx = leafstats(M.fact)
        if !exact
            lv == 1 && (fv = M.S * bvec)
            lv == 1 ? (ref1 = sweep_solve(M.fact, fv)) : nothing
            uex = ref1
        end
        e = norm(sweep_solve(M.fact, fv) - uex) / norm(uex)
        @printf("%-7d %-9s %-8d %-9.1f %-9.3f %-6d %-11.3e %-8d %-9d\n",
                f, "$(n)^3", lv, M.tsetup, r.tapply, r.iters, e, nl, mx)
        flush(stdout)
        M = nothing
        GC.gc()
    end
end
