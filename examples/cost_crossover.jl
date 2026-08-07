# When does the O(N) sweep actually beat an exact sparse LU of the same
# surrogate?  In 2D the answer is "later than you might expect".
#
#   julia --project=. examples/cost_crossover.jl [maxfreq]
#
# Three preconditioners, all applied to the same dense Lippmann–Schwinger system:
#
#   PML+sweep  the 2018 method: PML boundary stencils, moving-PML sweep   O(N)
#   PML+LU     the same surrogate H, inverted exactly by UMFPACK          O(N^3/2)
#   Ying+LU    the 2015 surrogate (one-sided ∂Ω stencils), exact LU       O(N^3/2)
#
# The 2015 paper's cost is dominated by the nested-dissection factorisation,
# O(N^3/2) setup and O(N log N) solve in 2D; the sweep replaces that with O(N)
# and O(bN).  The point of the 2018 paper is that this matters — but in 2D the
# exponent gap is only 3/2 vs 1, so the crossover sits around N ~ 3e5 and the
# margin is still modest at N ~ 1e6.  (In 3D the gap is O(N^2) vs O(N^4/3),
# which is where the method really pays; 3D is not implemented here.)

using SparsifyAndSweep
using LinearAlgebra, SparseArrays, Printf

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
freqs = [f for f in (16, 32, 64, 128) if f <= maxf]

@printf("%-7s %-9s %-12s %-10s %-10s %-10s %-10s\n",
        "ω/(2π)", "N", "variant", "Nunk", "T_setup", "T_apply", "iters")
for f in freqs
    n = 8f - 1
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    bvec = rhs(P, plane_wave(n, ω))
    # the 2015 surrogate is only built up to 511² here: its eight boundary
    # stencils each cost an O(n²) sweep, which starts to dominate its setup
    variants = f <= 64 ? (:sweep, :direct, :ying) : (:sweep, :direct)
    for v in variants
        M, lbl = if v === :sweep
            SweepPreconditioner(P), "PML+sweep"
        elseif v === :direct
            SweepPreconditioner(P; mode = :direct), "PML+LU"
        else
            SweepPreconditioner(P; mode = :direct, boundary = :ying), "Ying+LU"
        end
        r = solve(P, bvec; M = M, tol = 1e-6, maxiter = 200)
        @printf("%-7d %-9s %-12s %-10d %-10.2f %-10.3f %-10d\n",
                f, "$(n)^2", lbl, size(M.H, 1), M.tsetup, r.tapply, r.iters)
        flush(stdout)
        M = nothing
        GC.gc()
    end
end
