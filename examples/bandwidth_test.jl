# Is the L-sweeps application bound by memory bandwidth?  Counts the factor bytes
# one application must stream (every node reads its box's L and U once), times the
# application and reports the implied traffic rate, and calibrates both against the
# DRAM read bandwidth this machine delivers on one thread and on all of them, with
# cores busy (CPU seconds / wall seconds, from /proc) alongside.
#
#   julia --project=. -t 16 examples/bandwidth_test.jl 256
using SparsifyAndSweep, LinearAlgebra, SparseArrays, Printf
const SS = SparsifyAndSweep
const HZ = 100.0

function proc_times()
    s = read("/proc/self/stat", String)
    f = split(@view s[findlast(')', s)+2:end])
    return (parse(Int, f[12]) / HZ, parse(Int, f[13]) / HZ)
end

"Best-of-n wall time of `g()`, and the cores busy during that best run."
function timed(g; n = 3)
    best, cores = Inf, 0.0
    for _ in 1:n
        GC.gc()
        (u0, s0), t0 = proc_times(), time()
        g()
        t1, (u1, s1) = time(), proc_times()
        if t1 - t0 < best
            best, cores = t1 - t0, (u1 - u0 + s1 - s0) / (t1 - t0)
        end
    end
    return best, cores
end

nn = parse(Int, ARGS[1])
n, ω = 8nn - 1, 2π * nn
@printf("omega/2pi = %d, n = %d, julia -t %d, JULIA_EXCLUSIVE=%s\n",
        nn, n, Threads.nthreads(), get(ENV, "JULIA_EXCLUSIVE", "0"))

# --- what the memory system can do: read 2 GB, on all threads and on one ---
let N = 2^28
    a = ones(Float64, N)
    function tsum(a)
        nt = Threads.nthreads()
        parts = zeros(Float64, nt)
        Threads.@threads :static for k in 1:nt
            lo = (k - 1) * div(length(a), nt) + 1
            hi = k == nt ? length(a) : k * div(length(a), nt)
            acc = 0.0
            @inbounds @simd for i in lo:hi; acc += a[i]; end
            parts[k] = acc
        end
        return sum(parts)
    end
    tsum(a); sum(a)
    tt, _ = timed(() -> tsum(a))
    ts, _ = timed(() -> sum(a))
    @printf("DRAM read bandwidth: %5.1f GB/s on %d threads, %5.1f GB/s on one thread\n",
            8N / 2^30 / tt, Threads.nthreads(), 8N / 2^30 / ts)
end

g = Grid{2}(n, 8)
m = perturbation(velocity(:converging, n))
K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
F = lsweep_setup(g, ω, m, astar, bstar; nbox = 64, pad = 8)
f = zeros(ComplexF64, nunk(g)); f[SS.phys_indices(g)] .= 1

# --- what one application must stream: every node reads its box's L and U once ---
lunz = Dict(ij => nnz(b.fact) for (ij, b) in F.boxes)
per_node_bytes = 24                                   # ComplexF64 value + Int64 index
factor_gb = sum(values(lunz)) * per_node_bytes / 2^30
traffic_gb = sum(lunz[(nd[1], nd[2])] for nd in F.nodes) * per_node_bytes / 2^30
@printf("factors: %d boxes, %.1f GB in total; one application streams %.1f GB over %d nodes\n",
        length(F.boxes), factor_gb, traffic_gb, length(F.nodes))

lsweep_solve(F, f)                                    # warm up
wall, cores = timed(() -> lsweep_solve(F, f))
@printf("one application: %6.3fs   cores busy %5.2f   implied factor traffic %5.1f GB/s\n",
        wall, cores, traffic_gb / wall)
