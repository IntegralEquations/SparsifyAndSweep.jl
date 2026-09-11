# How many cores are actually busy during a solve, read from /proc rather than
# inferred from wall-clock scaling.  For each phase we report CPU seconds charged
# to the process (utime+stime) against elapsed wall seconds; the ratio is the mean
# number of cores occupied.  A ratio of 1.0 means the phase ran on one core.
# Per-thread deltas show which OS threads did the work.
#
#   julia --project=. -t 16 examples/cpu_usage.jl 256 sweep
#   julia --project=. -t 16 examples/cpu_usage.jl 256 lsweep
using SparsifyAndSweep, LinearAlgebra, Printf
const SS = SparsifyAndSweep
const HZ = 100.0                                   # getconf CLK_TCK

"(utime, stime) in seconds from a /proc stat file, parsed after the comm field."
function proc_times(path::AbstractString)
    s = read(path, String)
    f = split(@view s[findlast(')', s)+2:end])     # f[1] = state, so overall field k is f[k-2]
    return (parse(Int, f[12]) / HZ, parse(Int, f[13]) / HZ)
end
proc_cpu(path) = sum(proc_times(path))

self_times() = proc_times("/proc/self/stat")
thread_cpu() = Dict(t => proc_cpu("/proc/self/task/$t/stat") for t in readdir("/proc/self/task"))

"""
Run `f`, and report wall seconds, user and system CPU seconds and their ratio to
wall (mean cores busy), plus the GC time and bytes allocated inside `f`.  System
time is the kernel: page faults and mmap, which threads of one process contend
for and separate processes do not.
"""
function measure(label, f; threads = false)
    GC.gc()
    h0 = threads ? thread_cpu() : nothing
    g0 = Base.gc_num()
    (u0, s0), t0 = self_times(), time()
    f()
    t1, (u1, s1) = time(), self_times()
    g = Base.GC_Diff(Base.gc_num(), g0)
    wall, user, sys = t1 - t0, u1 - u0, s1 - s0
    @printf("  %-32s wall %7.3fs  user %8.3fs  sys %7.3fs  cores %5.2f  gc %6.3fs  alloc %6.2fGB\n",
            label, wall, user, sys, (user + sys) / wall, g.total_time / 1e9, g.allocd / 2^30)
    if threads
        h1 = thread_cpu()
        d = sort([(t, h1[t] - get(h0, t, 0.0)) for t in keys(h1)], by = x -> -x[2])
        active = [(t, v) for (t, v) in d if v > 0.02]
        @printf("    %d OS threads exist, %d used more than 20ms:\n", length(d), length(active))
        for (t, v) in active[1:min(end, 20)]
            @printf("      tid %-8s %6.3fs\n", t, v)
        end
    end
    return wall, user + sys
end

nn = parse(Int, ARGS[1])
mode = Symbol(length(ARGS) > 1 ? ARGS[2] : "sweep")
n, ω = 8nn - 1, 2π * nn
kw = mode === :lsweep ? (; mode = :lsweep, nbox = 64, pad = 8) : (; mode = :sweep)

@printf("omega/2pi = %d, n = %d, mode = %s, julia -t %d, BLAS %d threads\n",
        nn, n, mode, Threads.nthreads(), BLAS.get_num_threads())

c = velocity(:converging, n)
P = LSProblem(n, ω, c; b = 8)
b = rhs(P, plane_wave(n, ω))

M = nothing
measure("setup (SweepPreconditioner)", () -> (global M = SweepPreconditioner(P; kw...)))
apply_M(M, b)                                      # warm up
y = similar(b); apply_A!(y, P, b)

measure("one preconditioner application", () -> apply_M(M, b); threads = true)
measure("five preconditioner applications", () -> (for _ in 1:5; apply_M(M, b); end); threads = true)
measure("one dense operator application", () -> apply_A!(y, P, b))
r = nothing
measure("full solve phase (GMRES)", () -> (global r = solve(P, b; M = M)); threads = true)
@printf("  -> %d iterations, relres %.1e\n", r.iters, r.relres)
