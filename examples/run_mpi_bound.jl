# Launch a script under mpiexec with `nranks` ranks of `nthreads` threads each,
# every rank bound to its own set of cores.
#
#   julia --project=test examples/run_mpi_bound.jl 4 4 examples/lsweep_mpi.jl 128
using MPI
nr, nt = parse(Int, ARGS[1]), parse(Int, ARGS[2])
script, rest = ARGS[3], ARGS[4:end]
cmd = `$(MPI.mpiexec()) -n $nr --map-by slot:PE=$nt --bind-to core
       $(Base.julia_cmd()) -t $nt --project=$(Base.active_project()) $script $rest`
run(cmd)
