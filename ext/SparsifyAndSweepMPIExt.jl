"""
    module SparsifyAndSweepMPIExt

Distributed-memory backend of the L-sweeps, loaded automatically with `using MPI`.

The CDD geometry and the dependency graph of the sweep are replicated; every
rank assembles and factorises only the boxes it owns (row-based by default) and
the trace vectors between boxes on different ranks travel as small point-to-point
messages, one per (target node, contribution), matched by tag.  MPI calls are
serialised behind a lock so that the worker tasks of a rank can post and test
messages concurrently under any thread level from `:serialized` up.  The
approximate solution is assembled by an all-reduce of the owned entries, so
every rank ends with the full vector and the outer Krylov iteration can run
replicated.
"""
module SparsifyAndSweepMPIExt

using MPI
using SparsifyAndSweep
using SparsifyAndSweep: MPIComm, AbstractComm
import SparsifyAndSweep: nranks, myrank, max_tag, post_recv!, post_send!, test_request!,
                         wait_requests!, allreduce_sum!

function SparsifyAndSweep.MPIComm(comm::MPI.Comm = MPI.COMM_WORLD)
    MPI.Initialized() || MPI.Init(threadlevel = :multiple)
    ub = try
        Int(MPI.tag_ub())
    catch
        32767
    end
    return MPIComm(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm), ReentrantLock(), ub)
end

nranks(c::MPIComm) = c.size
myrank(c::MPIComm) = c.rank
max_tag(c::MPIComm) = c.tag_ub

post_recv!(c::MPIComm, buf::Vector{ComplexF64}, src::Int, tag::Int) =
    lock(c.lock) do
        MPI.Irecv!(buf, c.comm; source = src, tag = tag)
    end
post_send!(c::MPIComm, buf::Vector{ComplexF64}, dst::Int, tag::Int) =
    lock(c.lock) do
        MPI.Isend(buf, c.comm; dest = dst, tag = tag)
    end
test_request!(c::MPIComm, req) =
    lock(c.lock) do
        MPI.Test(req)
    end
function wait_requests!(c::MPIComm, reqs)
    isempty(reqs) && return nothing
    lock(c.lock) do
        MPI.Waitall(MPI.Request[r for r in reqs])
    end
    return nothing
end
function allreduce_sum!(c::MPIComm, u::Vector{ComplexF64})
    MPI.Allreduce!(u, +, c.comm)
    return u
end

end # module
