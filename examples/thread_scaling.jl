# Where the two inner solvers spend their time, and which stages actually thread.
# One process per thread count, with BLAS pinned to the same count so that a run
# at -t 1 does not quietly use all sixteen cores inside UMFPACK's dense kernels.
#
#   for T in 1 4 8 16; do
#     OPENBLAS_NUM_THREADS=$T julia --project=. -t $T examples/thread_scaling.jl 32 64 128 256
#   done
#
# Stages: `stencil` and `H` are common to both solvers and serial; `fact` is the
# factorization, `apply` one application of the inner solver, and `dense` one
# application of the Lippmann-Schwinger operator itself -- the FFT convolution,
# which the package never hands to FFTW's threads, so it is serial by construction.
using SparsifyAndSweep, LinearAlgebra, Printf
const SS = SparsifyAndSweep

function best(f; n = 3)
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

@printf("# julia -t %d, BLAS %d threads\n", Threads.nthreads(), BLAS.get_num_threads())
@printf("%-5s %-7s %8s %8s %9s %9s %9s %9s %9s\n",
        "w/2pi", "n", "stencil", "H", "fact_sw", "apply_sw", "fact_L", "apply_L", "dense")

for nn in parse.(Int, ARGS)
    n, ω = 8nn - 1, 2π * nn
    g = Grid{2}(n, 8)
    c = velocity(:converging, n)
    m = perturbation(c)

    K = Kernel(Val(2), ω, g.h, self_weight(Val(2), ω, g.h), n + 1)
    tstencil = @elapsed ((astar, bstar, _) = sparsify_stencils(K, n; farfield = n))
    tH = @elapsed H = build_H(g, ω, m, astar, bstar)

    f = zeros(ComplexF64, nunk(g))
    f[SS.phys_indices(g)] .= 1

    tfac_s = @elapsed Fs = sweep_setup(g, ω, m, H)
    sweep_solve(Fs, f)                      # warm up
    tapp_s = best(() -> sweep_solve(Fs, f))
    Fs = nothing; GC.gc()                   # free before building the other

    tfac_l = @elapsed Fl = lsweep_setup(g, ω, m, astar, bstar; nbox = 64, pad = 8)
    lsweep_solve(Fl, f)
    tapp_l = best(() -> lsweep_solve(Fl, f))
    Fl = nothing; GC.gc()

    P = LSProblem(n, ω, c; b = 8)
    b = rhs(P, plane_wave(n, ω))
    y = similar(b)
    apply_A!(y, P, b)
    tdense = best(() -> apply_A!(y, P, b))
    P = nothing; GC.gc()

    @printf("%-5d %-7d %8.3f %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
            nn, n, tstencil, tH, tfac_s, tapp_s, tfac_l, tapp_l, tdense)
end
