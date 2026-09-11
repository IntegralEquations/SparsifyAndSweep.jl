# The layered moving-PML sweep against L-sweeps as the inner solver of the
# preconditioner in three dimensions, with and without inner GMRES iterations on
# the sparse system (the two-level scheme).  Converging lens, h = λ/8, b = 4,
# C = 12, as in examples/tables3d.jl.  L-sweeps uses beams: a q x q decomposition
# of the (x₁, x₂) plane, each box spanning the whole of x₃.  The layered sweep is
# serial; L-sweeps uses every Julia thread.
#
#   julia --project=. -t 16 examples/lsweep_compare3d.jl 4 8 --boxes=3 --pad=8
#   julia --project=. -t 16 examples/lsweep_compare3d.jl 8 --boxes=4 --pad=4 --nfact=8 --only=lsweep
# `--nfact` caps the factorisations in flight; sixteen 3D beams at once exceed this
# machine's memory at n = 63.
using SparsifyAndSweep, LinearAlgebra, Printf

qb, pad, inner, nfact, only, sizes = 3, 8, 6, Threads.nthreads(), nothing, Int[]
for a in ARGS
    if startswith(a, "--boxes=");  global qb = parse(Int, a[9:end])
    elseif startswith(a, "--pad=");  global pad = parse(Int, a[7:end])
    elseif startswith(a, "--inner="); global inner = parse(Int, a[9:end])
    elseif startswith(a, "--nfact="); global nfact = parse(Int, a[9:end])
    elseif startswith(a, "--only=");  global only = split(a[8:end], ',')
    else push!(sizes, parse(Int, a)) end
end

rows(qb, pad, inner, nfact) = (
    ("sweep",   "layered sweep",              (; mode = :sweep)),
    ("sweepi",  "layered sweep, inner $inner", (; mode = :sweep, inner = inner)),
    ("lsweep",  "L-sweeps $(qb)x$(qb) pad $pad",       (; mode = :lsweep, boxes = (qb, qb), pad = pad, nfact = nfact)),
    ("lsweepi", "L-sweeps $(qb)x$(qb) pad $pad, inner $inner", (; mode = :lsweep, boxes = (qb, qb), pad = pad, nfact = nfact, inner = inner)),
)

function row(P, b, label; kwargs...)
    M = SweepPreconditioner(P; C = 12.0, kwargs...)
    M.nsweeps[] = 0
    apply_M(M, b)                      # warm up; also the sweep count of one application
    per_apply = M.nsweeps[]
    M.nsweeps[] = 0
    r = solve(P, b; M = M, maxiter = 100)
    sweeps = M.nsweeps[] - per_apply   # solve() times one application before GMRES
    @printf("  %-34s setup %8.2fs  apply %7.3fs  outer %3d  inner %2d  inner*outer %4d  sweeps %4d  solve %8.2fs  relres %.1e  peak %5.1fGB\n",
            label, r.tsetup, r.tapply, r.iters, M.inner, r.iters * max(M.inner, 1), sweeps,
            r.tsolve, r.relres, Sys.maxrss() / 2^30)
end

for nn in sizes
    n, ω = 8nn - 1, 2π * nn
    c = velocity(Val(3), :converging, n)
    P = LSProblem(n, ω, c; b = 4)
    b = rhs(P, plane_wave(Val(3), n, ω))
    @printf("ω/2π = %d, n = %d, N = %d^3, Nunk = %d, threads = %d\n", nn, n, n, nunk(P.g), Threads.nthreads())
    for (key, label, kw) in rows(qb, pad, inner, nfact)
        (only === nothing || key in only) || continue
        row(P, b, label; kw...)
    end
end
