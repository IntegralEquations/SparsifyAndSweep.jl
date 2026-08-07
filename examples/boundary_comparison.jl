# How much does the PML boundary stencil of Liu & Ying (2018) actually buy over
# the one-sided boundary stencils of Ying (2015)?
#
#   julia --project=. examples/boundary_comparison.jl [maxfreq]
#
# Everything else is held fixed — same dense Lippmann–Schwinger system, same
# interior α/β stencils, same exact sparse LU of the surrogate — so the only
# difference is how the radiation condition is discretised:
#
#   :pml   extend the grid by b layers, γ stencils fitted to complex-stretched
#          plane waves, zero Dirichlet on the outer ring        (2018, §2.2.2)
#   :ying  no extension, one-sided stencils B along ∂Ω that annihilate the
#          Green's functions of all sources ≥ buffer layers in  (2015, §3.1)
#
# Reported per case:
#   Nunk    unknowns in the surrogate H
#   err     ‖H⁻¹Sb − u_dense‖/‖u_dense‖, i.e. the surrogate used as a *direct*
#           solver, against the true dense solution
#   iters   GMRES(20) iterations to 1e-6 using H⁻¹ as the preconditioner
#   T_set   time to build and factorise H

using SparsifyAndSweep
using LinearAlgebra, SparseArrays, Printf

maxf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
freqs = [f for f in (8, 16, 32, 64) if f <= maxf]
buffers = (2, 4, 8)

for kind in (:converging, :multi)
    println("\n", "="^92)
    println("velocity field: ", kind)
    println("="^92)
    @printf("%-7s %-8s | %-8s %-10s %-6s %-8s | %-8s %-10s %-6s %-8s\n",
            "ω/(2π)", "N", "PML Nunk", "PML err", "iters", "T_set",
            "Ying Nunk", "Ying err", "iters", "T_set")
    for f in freqs
        n = 8f - 1
        ω = 2π * f
        P = LSProblem(n, ω, velocity(kind, n); b = 8)
        bvec = rhs(P, plane_wave(n, ω))

        # reference: the true solution of the dense system, to 1e-12
        Mref = SweepPreconditioner(P; mode = :direct)
        uex = solve(P, bvec; M = Mref, tol = 1e-12, maxiter = 300).u

        cols = String[]
        for M in (Mref, SweepPreconditioner(P; mode = :direct, boundary = :ying))
            err = norm(apply_M(M, bvec) - uex) / norm(uex)
            r = solve(P, bvec; M = M, tol = 1e-6, maxiter = 200)
            push!(cols, @sprintf("%-8d %-10.3e %-6d %-8.2f",
                                 size(M.H, 1), err, r.iters, M.tsetup))
        end
        @printf("%-7d %-8s | %s | %s\n", f, "$(n)^2", cols[1], cols[2])
        flush(stdout)
    end
end

# Ying's construction has one free parameter of its own: the buffer width b,
# the number of layers next to ∂Ω on which m is assumed to vanish.  Sources
# closer than that get no annihilation constraint.
println("\n", "="^92)
println("sensitivity of the :ying boundary to its buffer width b (field = converging)")
println("="^92)
@printf("%-7s %-8s", "ω/(2π)", "N")
for b in buffers
    @printf(" | b=%-2d err      iters", b)
end
println()
for f in freqs
    n = 8f - 1
    ω = 2π * f
    P = LSProblem(n, ω, velocity(:converging, n); b = 8)
    bvec = rhs(P, plane_wave(n, ω))
    uex = solve(P, bvec; M = SweepPreconditioner(P; mode = :direct),
                tol = 1e-12, maxiter = 300).u
    @printf("%-7d %-8s", f, "$(n)^2")
    for b in buffers
        M = SweepPreconditioner(P; mode = :direct, boundary = :ying, buffer = b)
        err = norm(apply_M(M, bvec) - uex) / norm(uex)
        r = solve(P, bvec; M = M, tol = 1e-6, maxiter = 200)
        @printf(" | %-10.3e %-5d", err, r.iters)
        flush(stdout)
    end
    println()
end
