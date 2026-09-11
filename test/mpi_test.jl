# Distributed L-sweeps: run under mpiexec (see mpi_launch_test.jl).  Every rank
# builds the distributed factorization and, for reference, the single-process
# one; the two must agree bitwise, and every rank must hold the same result.
using MPI
using SparsifyAndSweep
using LinearAlgebra, SparseArrays, Random, Test, Printf

comm = MPIComm()
rank, nr = myrank(comm), nranks(comm)

nn = 16
n, ω, b = 8nn - 1, 2π * nn, 8
g = Grid{2}(n, b)
m = perturbation(velocity(:converging, n))
K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
H = build_H(g, ω, m, astar, bstar; C = 12.0)
f = zeros(ComplexF64, nunk(g))
f[SparsifyAndSweep.phys_indices(g)] = randn(MersenneTwister(11), ComplexF64, n^2)

ok = true
for boxes in ((3, 3), (4, 5))
    Fm = lsweep_setup(g, ω, m, astar, bstar; boxes = boxes, comm = comm)
    Fl = lsweep_setup(g, ω, m, astar, bstar; boxes = boxes)
    um = lsweep_solve(Fm, f)
    ul = lsweep_solve(Fl, f)
    nown = SparsifyAndSweep.nowned(Fm)
    # everyone must hold rank 0's vector
    u0 = copy(um)
    MPI.Bcast!(u0, 0, comm.comm)
    same_all = MPI.Allreduce(um == u0, MPI.LAND, comm.comm)
    same_loc = MPI.Allreduce(um == ul, MPI.LAND, comm.comm)
    uex = lu(H) \ f
    err = norm(um - uex) / norm(uex)
    errl = norm(ul - uex) / norm(uex)
    maxdiff = MPI.Allreduce(maximum(abs, um - ul), MPI.MAX, comm.comm)
    global ok &= same_all && same_loc && err < 5e-2
    rank == 0 && @printf("boxes=%s ranks=%d threads=%d: owned on rank 0 = %d, identical on all ranks = %s, bitwise = local %s (max |diff| %.1e), rel err vs H\\f: mpi %.2e local %.2e\n",
                         boxes, nr, Threads.nthreads(), nown, same_all, same_loc, maxdiff, err, errl)
end
# and through the dense-system preconditioner (outer GMRES replicated on every rank)
P = LSProblem(n, ω, velocity(:converging, n); b = b)
M = SweepPreconditioner(P; mode = :lsweep, boxes = (3, 3), comm = comm)
r = solve(P, rhs(P, plane_wave(n, ω)); M = M)
it0 = MPI.Bcast(r.iters, 0, comm.comm)
global ok &= MPI.Allreduce(r.converged && r.relres < 1e-6 && r.iters == it0, MPI.LAND, comm.comm)
rank == 0 && @printf("preconditioned GMRES on %d ranks: iters = %d, relres = %.1e\n", nr, r.iters, r.relres)
MPI.Barrier(comm.comm)
ok || (rank == 0 && println("MPI L-sweep test FAILED"); exit(1))
rank == 0 && println("MPI L-sweep test passed")
