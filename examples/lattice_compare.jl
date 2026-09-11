# Four discs in a square lattice, each a constant-contrast disc with a sharp edge,
# against the sweeps: the twisted moving-PML sweep of Liu & Ying along x₁, the
# same sweep along x₂, and the bi-directional alternation of Zepeda-Núñez & Zhao
# (Alg. 2 of SIAM J. Sci. Comput. 38 (2016) B866-B888): sweep along x₁, form the
# residual, sweep it along x₂, add.  The single-axis rows are also run as the
# inner solver of the two-level scheme (their Alg. 3: GMRES on H to 1e-3 inside
# every application, flexible GMRES outside).
#
# The discs have radius R about the centres (1/2 ± a, 1/2 ± a) of the unit square
# and m = m_in inside (m = 1 - 1/c²; m_in = -1 is n² = 2, c_in = 1/√2), 0 outside.
# The incident wave is e^{iω d·x}.  Every row solves the same discrete system to
# the same tolerance, so the rows differ only in the work the preconditioner
# needs; the "vs direct" column checks that they all reached the same solution.
#
#   julia --project=. -t 8 examples/lattice_compare.jl 8 16 32 64
#   julia --project=. -t 8 examples/lattice_compare.jl 32 --dir=1,1 --only=x1,bidir
#   julia --project=. -t 8 examples/lattice_compare.jl 32 --m=-3 --ppw=12
using SparsifyAndSweep, LinearAlgebra, SparseArrays, Printf

function lattice_velocity(n::Int, R::Float64, m_in::Float64, centres)
    x = gridpoints(n)
    c_in = 1 / sqrt(1 - m_in)
    c = ones(Float64, n, n)
    for j in 1:n, i in 1:n, ctr in centres
        (x[i] - ctr[1])^2 + (x[j] - ctr[2])^2 < R^2 && (c[i, j] = c_in)
    end
    return c
end

# ------------------------------------------------------------------ arguments
R, a, m_in, ppw, dir, only, restart, maxiter, tol = 0.15, 0.2, -1.0, 8, (0.0, 1.0), nothing, 60, 300, 1e-6
sizes = Int[]
for arg in ARGS
    if startswith(arg, "--only=");        global only = split(arg[8:end], ',')
    elseif startswith(arg, "--ppw=");     global ppw = parse(Int, arg[7:end])
    elseif startswith(arg, "--R=");       global R = parse(Float64, arg[5:end])
    elseif startswith(arg, "--a=");       global a = parse(Float64, arg[5:end])
    elseif startswith(arg, "--m=");       global m_in = parse(Float64, arg[5:end])
    elseif startswith(arg, "--restart="); global restart = parse(Int, arg[11:end])
    elseif startswith(arg, "--maxiter="); global maxiter = parse(Int, arg[11:end])
    elseif startswith(arg, "--tol=");     global tol = parse(Float64, arg[7:end])
    elseif startswith(arg, "--dir=")
        d = parse.(Float64, split(arg[7:end], ',')); global dir = (d[1], d[2]) ./ hypot(d...)
    else push!(sizes, parse(Int, arg)) end
end
centres = [(0.5 + s1 * a, 0.5 + s2 * a) for s1 in (-1, 1) for s2 in (-1, 1)]

const ROWS = (
    ("direct",   "exact solve of H",                     (; mode = :direct)),      # the floor: sparsification error alone
    ("x1",       "x₁ sweep, two fronts",                 (; mode = :sweep, fronts = 2, axes = (1,))),
    ("x2",       "x₂ sweep, two fronts",                 (; mode = :sweep, fronts = 2, axes = (2,))),
    ("x1f1",     "x₁ sweep, one front",                  (; mode = :sweep, fronts = 1, axes = (1,))),
    ("alt",      "x₁ then −x₁, one front",               (; mode = :sweep, fronts = 1, axes = (1, -1))),
    ("rep1",     "x₁ then x₁ again, one front",          (; mode = :sweep, fronts = 1, axes = (1, 1))),    # control: same sweep on its own residual
    ("quad",     "x₁, −x₁, x₂, −x₂, one front",          (; mode = :sweep, fronts = 1, axes = (1, -1, 2, -2))),
    ("quad2",    "x₁, x₂, −x₁, −x₂, one front",          (; mode = :sweep, fronts = 1, axes = (1, 2, -1, -2))),
    ("lsweep",   "L-sweeps pad 8",                       (; mode = :lsweep, nbox = 64, pad = 8)),
    ("lsweep2",  "L-sweeps + half-box-shifted L-sweeps", (; mode = :lsweep, nbox = 64, pad = 8, shifts = ((0, 0), (0.5, 0.5)))),
    ("lsweepi6", "L-sweeps, inner 6",                    (; mode = :lsweep, nbox = 64, pad = 8, inner = 6)),
    ("lsweep2i6","L-sweeps + shifted, inner 6",          (; mode = :lsweep, nbox = 64, pad = 8, shifts = ((0, 0), (0.5, 0.5)), inner = 6)),
    ("bidir",    "bi-directional x₁,x₂, two fronts",     (; mode = :sweep, fronts = 2, axes = (1, 2))),
    ("bidirf1",  "bi-directional x₁,x₂, one front",      (; mode = :sweep, fronts = 1, axes = (1, 2))),
    ("bidir3",   "x₁,x₂,x₁, two fronts",                 (; mode = :sweep, fronts = 2, axes = (1, 2, 1))),
    ("rep2",     "x₁ then x₁ again, two fronts",         (; mode = :sweep, fronts = 2, axes = (1, 1))),    # control for the bi-directional row
    ("x1i6",     "x₁ two fronts, inner 6",               (; mode = :sweep, fronts = 2, axes = (1,), inner = 6)),
    ("bidiri6",  "bi-directional, inner 6",              (; mode = :sweep, fronts = 2, axes = (1, 2), inner = 6)),
    ("x1i12",    "x₁ two fronts, inner 12",              (; mode = :sweep, fronts = 2, axes = (1,), inner = 12)),
    ("bidiri12", "bi-directional, inner 12",             (; mode = :sweep, fronts = 2, axes = (1, 2), inner = 12)),
)

# Work counts, measured: `sweeps` is the number of slab sweeps performed inside the
# outer Krylov solve (an alternating application performs one per axis; the
# exact-H row counts LU solves), `fwd` the number of dense forward maps.  A
# left-preconditioned GMRES applies both once per iteration and once more at
# every restart; the two-level rows keep the forward maps at one per outer
# iteration while the sweeps multiply.
function row(P, b, label, udirect; kwargs...)
    M = SweepPreconditioner(P; kwargs...)
    M.nsweeps[] = 0
    apply_M(M, b)                       # warm up; also the sweep count of one application
    per_apply = M.nsweeps[]
    M.nsweeps[] = 0
    P.nA[] = 0
    r = solve(P, b; M = M, tol = tol, restart = restart, maxiter = maxiter)
    sweeps = M.nsweeps[] - per_apply    # solve() times one application before GMRES
    fwd = P.nA[] - 1                    # less solve()'s own true-residual evaluation
    dev = udirect === nothing ? NaN : norm(r.u - udirect) / norm(udirect)
    @printf("  %-36s setup %7.2fs  apply %7.3fs  outer %3d%s  sweeps %4d  fwd %4d  solve %7.2fs  relres %.1e  vs direct %.1e  peak %5.1fGB\n",
            label, r.tsetup, r.tapply, r.iters, r.converged ? " " : "!", sweeps, fwd, r.tsolve,
            r.relres, dev, Sys.maxrss() / 2^30)
    flush(stdout)
    return r.u
end

for nn in sizes
    n, ω = ppw * nn - 1, 2π * nn
    c = lattice_velocity(n, R, m_in, centres)
    P = LSProblem(n, ω, c; b = 8)
    b = rhs(P, plane_wave(Val(2), n, ω, dir))
    ℓ = length(slice_ranges(P.g))
    @printf("ω/2π = %d, n = %d (%d ppw outside, %.1f inside), N = %d, %d slices; 4 discs R = %.2f (%.1f λ across) at (½±%.2f, ½±%.2f), m_in = %.2f (n² = %.2f), incidence (%.2f, %.2f), GMRES(%d) to %.0e, threads = %d\n",
            nn, n, ppw, ppw / sqrt(1 - m_in), n^2, ℓ, R, 2R * nn, a, a, m_in, 1 - m_in, dir[1], dir[2], restart, tol, Threads.nthreads())
    udirect = nothing
    for (key, label, kw) in ROWS
        (only === nothing || key in only) || continue
        u = row(P, b, label, udirect; kw...)
        key == "direct" && (udirect = u)
    end
    GC.gc()
end
