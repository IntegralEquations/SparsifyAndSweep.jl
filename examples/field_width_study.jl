# The paper specifies its velocity fields only qualitatively ("a converging
# Gaussian centred at (0.5,0.5)", "32 randomly placed converging Gaussians with
# narrow width"), with colour bars showing c ∈ [0.7, 1.3].  The Gaussian widths
# are not given, and they matter: a wider lens bends rays more, which is exactly
# what the moving-PML approximation of the Schur complements is sensitive to.
#
# This script measures that sensitivity so the comparison with Tables 1–4 can be
# read with the right amount of scepticism.
#
#   julia --project=. examples/field_width_study.jl

using SparsifyAndSweep
using LinearAlgebra, Printf

function counts(kind, widths, freqs; shape = :gaussian, fronts = 2)
    println("\nfield = $kind, shape = $shape, fronts = $fronts")
    @printf("%-10s", "width")
    for f in freqs
        @printf(" ω/2π=%-4d", f)
    end
    println("   (iterations)")
    for w in widths
        @printf("%-10.3f", w)
        for f in freqs
            n = 8f - 1
            ω = 2π * f
            P = LSProblem(n, ω, velocity(kind, n; width = w, shape = shape); b = 8)
            M = SweepPreconditioner(P; fronts = fronts)
            r = solve(P, rhs(P, plane_wave(n, ω)); M = M, tol = 1e-6)
            @printf(" %-9d", r.iters)
            flush(stdout)
        end
        println()
    end
end

freqs = (16, 32, 64)
counts(:diverging, (0.05, 0.075, 0.1, 0.125, 0.15), freqs)
counts(:converging, (0.05, 0.075, 0.1, 0.125, 0.15), freqs)
counts(:multi, (0.015, 0.02, 0.03, 0.04), freqs)
# for reference, the compactly supported bump profile used before
counts(:diverging, (0.2, 0.3, 0.4), freqs; shape = :bump)
