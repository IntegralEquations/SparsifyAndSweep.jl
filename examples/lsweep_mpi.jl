# One L-sweeps setup and application under MPI, timed as the slowest rank.
# Run it through examples/run_mpi_bound.jl, which binds each rank to its own
# cores; without binding the launcher pins a rank and all of its threads to a
# single core and the timings are meaningless.
#
#   julia --project=test examples/run_mpi_bound.jl 4 4 examples/lsweep_mpi.jl 128
# (MPI.jl lives in the test environment, hence --project=test)
using MPI, SparsifyAndSweep, LinearAlgebra, Printf

comm = MPIComm()
rank, nr = myrank(comm), nranks(comm)
nn = isempty(ARGS) ? 128 : parse(Int, ARGS[1])
n, ω = 8nn - 1, 2π * nn
g = Grid{2}(n, 8)
m = perturbation(velocity(:converging, n))
K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
f = zeros(ComplexF64, nunk(g))
f[SparsifyAndSweep.phys_indices(g)] .= 1

MPI.Barrier(comm.comm)
ts = @elapsed F = lsweep_setup(g, ω, m, astar, bstar; nbox = 64, comm = comm)
ts = MPI.Allreduce(ts, MPI.MAX, comm.comm)

lsweep_solve(F, f)                      # warm up: the first application compiles
ta = Inf
for _ in 1:3
    MPI.Barrier(comm.comm)
    t = @elapsed lsweep_solve(F, f)
    global ta = min(ta, MPI.Allreduce(t, MPI.MAX, comm.comm))
end
rssmax = MPI.Allreduce(Sys.maxrss() / 2^30, MPI.MAX, comm.comm)

rank == 0 && @printf("ω/2π=%3d  %2d ranks x %2d threads: %d boxes, %d on rank 0; setup %6.2fs  apply %6.3fs  peak/rank %4.1fGB\n",
                     nn, nr, Threads.nthreads(), SparsifyAndSweep.nboxes(F),
                     SparsifyAndSweep.nowned(F), ts, ta, rssmax)
