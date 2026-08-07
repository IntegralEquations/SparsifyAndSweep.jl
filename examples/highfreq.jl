# One (field, frequency) pair per process, so that the sweeping factorisation of
# the previous run is definitely released before the next one is built.  Used to
# reach the last row of the paper's tables, ω/(2π) = 256 (N = 2047² ≈ 4.2·10⁶).
#
#   julia --project=. examples/highfreq.jl <field> <freq> [C]
#
# e.g.  julia --project=. examples/highfreq.jl diverging 256

using SparsifyAndSweep
using LinearAlgebra, Printf

kind = Symbol(ARGS[1])
f = parse(Int, ARGS[2])
C = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 12.0
n = 8f - 1
ω = 2π * f

t0 = time()
c = velocity(kind, n)
P = LSProblem(n, ω, c; b = 8)
bvec = rhs(P, plane_wave(n, ω))
tprob = time() - t0

M = SweepPreconditioner(P; C = C)
r = solve(P, bvec; M = M, tol = 1e-6, restart = 20)

@printf("%-11s %-7d %-9s C=%-5.1f %-10.2f %-10.3f %-8.2f %-6d %-10.2e  slices=%d  rss=%.1fGB\n",
        kind, f, "$(n)^2", C, r.tsetup, r.tapply, r.tsolve, r.iters, r.relres,
        length(slice_ranges(P.g)),
        parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30)
