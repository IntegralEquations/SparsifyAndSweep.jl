# The PML amplitude C (σ_max = C/ω) is not specified in the paper.  It is the
# total attenuation in nepers across a PML of b layers, so it controls how well
# each auxiliary problem imitates the true radiation condition.  Because the
# sweep composes ℓ ≈ ntot/b such approximations, the residual reflection of a
# single moving PML compounds, and the best C grows slowly with the number of
# slices.
#
#   julia --project=. examples/pml_amplitude_study.jl

using SparsifyAndSweep
using LinearAlgebra, Printf

Cs = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
freqs = (32, 64, 128)

for kind in (:converging, :diverging, :multi, :random)
    println("\nfield = $kind   (iterations; b = 8, two fronts, GMRES(20), tol 1e-6)")
    @printf("%-8s", "ω/2π")
    for C in Cs
        @printf(" C=%-5.0f", C)
    end
    println("   slices")
    for f in freqs
        n = 8f - 1
        ω = 2π * f
        P = LSProblem(n, ω, velocity(kind, n); b = 8)
        bvec = rhs(P, plane_wave(n, ω))
        @printf("%-8d", f)
        for C in Cs
            M = SweepPreconditioner(P; C = C)
            r = solve(P, bvec; M = M, tol = 1e-6, maxiter = 200)
            @printf(" %-7d", r.iters)
            flush(stdout)
        end
        @printf("   %d\n", length(slice_ranges(P.g)))
    end
end
