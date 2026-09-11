# Launch the distributed-memory test (test/mpi_test.jl) under mpiexec, with
# three processes of two threads each, and check that it passed.
using Test
using MPI

const nranks = 3
cmd = `$(MPI.mpiexec()) -n $nranks $(Base.julia_cmd()) -t 2 --project=$(Base.active_project()) $(joinpath(@__DIR__, "mpi_test.jl"))`
@info "running the MPI test: $cmd"
proc = run(ignorestatus(cmd))
@test success(proc)
