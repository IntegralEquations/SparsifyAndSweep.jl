# The layered moving-PML sweep against L-sweeps as the inner solver of the
# preconditioner, on the converging lens: setup and application time, iteration
# count and total solve time.  `inner` runs a few GMRES steps on the sparse
# system inside every application (the two-level scheme), which brings the
# outer count towards the exact-H floor of 3.  L-sweeps rows use the default
# compact single-precision factors unless the row says `umfpack`.
#
#   julia --project=. -t 16 examples/lsweep_compare.jl 16 32 64 128
#
# At the large sizes run one row per process, so that the reported peak
# resident set belongs to that row alone:
#
#   julia --project=. -t 16 examples/lsweep_compare.jl 256 --only=sweep
#   julia --project=. -t 16 examples/lsweep_compare.jl 256 --only=lsweep8
using SparsifyAndSweep, LinearAlgebra, Printf

const ROWS = (
    ("sweep",       "layered sweep",                (; mode = :sweep)),
    ("lsweep8",     "L-sweeps pad 8",               (; mode = :lsweep, nbox = 64, pad = 8)),
    ("lsweep8u",    "L-sweeps pad 8, umfpack",      (; mode = :lsweep, nbox = 64, pad = 8, factors = :umfpack)),
    ("lsweep16",    "L-sweeps pad 16",              (; mode = :lsweep, nbox = 64, pad = 16)),
    ("lsweep24",    "L-sweeps pad 24",              (; mode = :lsweep, nbox = 64, pad = 24)),
    ("lsweep8i6",   "L-sweeps pad 8, inner 6",      (; mode = :lsweep, nbox = 64, pad = 8, inner = 6)),
    ("lsweep8b96",  "L-sweeps pad 8, 96-pt boxes",  (; mode = :lsweep, nbox = 96,  pad = 8)),
    ("lsweep8b128", "L-sweeps pad 8, 128-pt boxes", (; mode = :lsweep, nbox = 128, pad = 8)),
    ("lsweep16b128","L-sweeps pad 16, 128-pt boxes",(; mode = :lsweep, nbox = 128, pad = 16)),
)

function row(P, b, label; kwargs...)
    M = SweepPreconditioner(P; kwargs...)
    apply_M(M, b)                      # warm up: the first application compiles
    r = solve(P, b; M = M)
    @printf("  %-30s setup %7.2fs  apply %7.3fs  iters %3d  solve %7.2fs  relres %.1e  peak %5.1fGB\n",
            label, r.tsetup, r.tapply, r.iters, r.tsolve, r.relres, Sys.maxrss() / 2^30)
end

only = nothing
sizes = Int[]
for a in ARGS
    if startswith(a, "--only=")
        global only = split(a[8:end], ',')
    else
        push!(sizes, parse(Int, a))
    end
end

for nn in sizes
    n, ω = 8nn - 1, 2π * nn
    c = velocity(:converging, n)
    P = LSProblem(n, ω, c; b = 8)
    b = rhs(P, plane_wave(n, ω))
    q = max(1, round(Int, ntot(P.g) / 64))
    @printf("ω/2π = %d, n = %d, N = %d, threads = %d, CDD %dx%d boxes of ~64 points\n",
            nn, n, n^2, Threads.nthreads(), q, q)
    for (key, label, kw) in ROWS
        (only === nothing || key in only) || continue
        row(P, b, label; kw...)
    end
end
