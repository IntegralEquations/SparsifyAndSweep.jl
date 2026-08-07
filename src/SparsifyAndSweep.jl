"""
    SparsifyAndSweep

Julia implementation of the preconditioner of

> F. Liu and L. Ying, *Sparsify and sweep: an efficient preconditioner for the
> Lippmann-Schwinger equation*, SIAM J. Sci. Comput. **40** (2018), B379-B404.

for the 2D Lippmann-Schwinger equation

    u(x) + ω² ∫_Ω G(x-y) m(y) u(y) dy = -ω² ∫_Ω G(x-y) m(y) u_I(y) dy .

The dense Nyström system `(I + ω² K M) u = g` is first sparsified into a
9-point stencil system `H ũ = f` (a PML-truncated discretisation of the
Helmholtz operator obtained by data fitting against the Green's function and
against complex-stretched plane waves), and `H` is then inverted approximately
by a moving-PML *sweeping* factorisation.  Both stages cost `O(N)`.

Basic use:

```julia
using SparsifyAndSweep
n, ω = 255, 2π*32
c  = velocity(:converging, n)
P  = LSProblem(n, ω, c; b = 8)
M  = SweepPreconditioner(P)
uI = plane_wave(n, ω)
b  = rhs(P, uI)
r  = solve(P, b; M = M)
r.iters, r.relres
```
"""
module SparsifyAndSweep

using LinearAlgebra
using SparseArrays
using SpecialFunctions
using FFTW
using Random
using Printf
using IterativeSolvers

export Grid, ntot, nunk, nphys, dim, lin, plin, offsets, ilo, ihi
export velocity, perturbation, plane_wave, gridpoints, smooth_random
export LSProblem, apply_A, apply_A!, rhs, dense_matrix
export SweepPreconditioner, apply_M, apply_M!
export sparsify_stencils, pml_stencil, build_H, rhs_operator
export boundary_stencils, BoundaryStencils, build_H_ying, rhs_operator_ying
export clipped_offsets, clip_signature, far_range
export sweep_setup, sweep_solve, slice_ranges, leafstats
export SlabSolver, DirectSlab, SweptSlab, SweepFactorization
export solve, SolveReport
export self_weight, green, green2d, ToeplitzConv, Kernel, kval
export farfield_residual, pmldirs, default_farfield

include("grid.jl")
include("kernel.jl")
include("media.jl")
include("stencils.jl")
include("system.jl")
include("sweep.jl")
include("solver.jl")

end # module
