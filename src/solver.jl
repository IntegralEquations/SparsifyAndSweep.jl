# Top level: the dense Lippmann–Schwinger operator, the sparsify-and-sweep
# preconditioner, and the driver that puts them together with GMRES.

"""
    LSProblem(n, ω, c; b = 8, quad = :lattice, withconv = true)

Discretized Lippmann-Schwinger equation (eq. 6)

    (I + ω² K M) u = g,      g = -ω² K M u_I,

on `I = {1,…,n}^D` with `h = 1/(n+1)`.  `c` is the velocity field sampled on `I`
and `m = 1 - 1/c²`.  `K` is applied by FFT in `O(N log N)`.

`withconv = false` skips building the FFT operator.
"""
struct LSProblem{D,C}
    g::Grid{D}
    ω::Float64
    m::Array{Float64,D}
    k0::ComplexF64
    conv::C
end

function LSProblem(n::Integer, ω::Real, c::AbstractArray{<:Real,D}; b::Integer = 8,
                   quad::Symbol = :lattice, withconv::Bool = true) where {D}
    size(c) == ntuple(_ -> n, D) ||
        throw(DimensionMismatch("velocity field must be $(ntuple(_->n,D))"))
    grid = Grid{D}(n, b)
    k0 = self_weight(Val(D), ω, grid.h; scheme = quad)
    conv = withconv ? ToeplitzConv(Val(D), ω, grid.h, n, k0) : nothing
    return LSProblem{D,typeof(conv)}(grid, Float64(ω), perturbation(c), k0, conv)
end

_needconv(P::LSProblem{D,Nothing}) where {D} =
    error("this LSProblem was built with withconv = false; the FFT operator " *
          "needed by apply_A / rhs / dense_matrix was not constructed")
_needconv(P::LSProblem) = nothing

dim(::LSProblem{D}) where {D} = D

"""
Default half-width `R` of the far field `μᶜ = {-R,…,R}^D \\ μ` used to fit `α`.

2D uses the paper's `R = n`.  3D caps it: `(2n+1)³` rows is already `10^8` at
`n = 255`, while the stencil stops changing well before `R = 24` because
Green's functions centred far away become linearly dependent on a `3^D`-point
patch.  [`farfield_residual`](@ref) checks a given truncation against the full
far field.
""" 
default_farfield(P::LSProblem{2}) = P.g.n
default_farfield(P::LSProblem{3}) = min(P.g.n, 24)

Base.size(P::LSProblem) = (nphys(P.g), nphys(P.g))
Base.size(P::LSProblem, d::Integer) = nphys(P.g)
Base.eltype(::LSProblem) = ComplexF64

# `LSProblem` is itself the linear operator GMRES iterates on
LinearAlgebra.mul!(y::AbstractVector, P::LSProblem, x::AbstractVector) = apply_A!(y, P, x)
Base.:*(P::LSProblem, x::AbstractVector) = apply_A(P, x)

"`y ← u + ω² K (m ⊙ u)`, the dense Lippmann-Schwinger operator."
function apply_A!(y::AbstractVector, P::LSProblem, u::AbstractVector)
    _needconv(P)
    t = vec(P.m) .* u
    mul!(y, P.conv, t)
    @. y = u + P.ω^2 * y
    return y
end
apply_A(P::LSProblem, u::AbstractVector) = apply_A!(similar(u, ComplexF64), P, u)

"Right-hand side `g = -ω² K (m ⊙ u_I)` for a given incident field."
function rhs(P::LSProblem, uI::AbstractArray)
    _needconv(P)
    t = vec(P.m) .* vec(ComplexF64.(uI))
    return -P.ω^2 * (P.conv * t)
end

"Dense `NxN` matrix of the Lippmann-Schwinger operator."
function dense_matrix(P::LSProblem)
    N = nphys(P.g)
    A = Matrix{ComplexF64}(undef, N, N)
    e = zeros(ComplexF64, N)
    for j in 1:N
        e[j] = 1
        apply_A!(view(A, :, j), P, e)
        e[j] = 0
    end
    return A
end

# --------------------------------------------------------------------------- #

"""
    SweepPreconditioner(P; C = 12.0, nsamp = 200, fronts = 2, mode = :sweep,
                        boundary = :pml, buffer = 2)

The preconditioner of Algorithms 1 and 2.  `mode = :sweep` uses the moving-PML
sweeping factorisation of `H`; `mode = :direct` factorises `H` exactly with a
sparse LU, which isolates the error of the sparsification from the error of the
sweep (§4 of the paper).  `fronts = 2` (the configuration used for the paper's
tables) sweeps from both ends towards a middle slice; `fronts = 1` is the plain
left-to-right sweep of Algorithms 1-2.  `levels = 2` additionally sweeps each
slice subproblem along `x₂` — the recursive approach of §3.1, which changes the
3D setup cost from `O(b²N^{4/3})` to `O(b⁴N)` and only applies in 3D.

`boundary` selects how the radiation condition is discretised, which is the one
place where the 2018 paper departs from Ying (2015):

* `:pml` (default) — extend the grid by `b` PML layers and use the `γ` stencils
  of §2.2.2, with zero Dirichlet data on the outer ring.
* `:ying` — no extension; carry the radiation condition on one-sided stencils
  `B` along `∂Ω`, as in Ying (2015).  `buffer` is his `b`, the number of layers
  next to `∂Ω` on which `m` is assumed to vanish.  Only available with
  `mode = :direct`: the moving-PML sweep needs the PML-extended grid.
"""
struct SweepPreconditioner{F}
    P::LSProblem
    astar::Vector{ComplexF64}
    bstar::Vector{ComplexF64}
    ratio::Float64                       # σ_min/σ_max of K_{μ,μᶜ}
    H::SparseMatrixCSC{ComplexF64,Int}
    S::SparseMatrixCSC{ComplexF64,Int}   # right-hand-side operator
    fact::F
    pidx::Vector{Int}
    mode::Symbol
    boundary::Symbol
    bnd::Any
    tsetup::Float64
    nstencil::Int
end

function SweepPreconditioner(P::LSProblem; C::Real = 12.0, nsamp::Integer = 200,
                             fronts::Integer = 2, mode::Symbol = :sweep,
                             boundary::Symbol = :pml, buffer::Integer = 2,
                             levels::Integer = 1,
                             farfield::Integer = default_farfield(P))
    mode in (:sweep, :direct) || throw(ArgumentError("mode must be :sweep or :direct"))
    boundary in (:pml, :ying) || throw(ArgumentError("boundary must be :pml or :ying"))
    boundary === :ying && mode !== :direct &&
        throw(ArgumentError("boundary = :ying requires mode = :direct; the " *
                            "moving-PML sweep needs the PML-extended grid"))
    t0 = time()
    g = P.g
    D = dim(P)
    K = Kernel(Val(D), P.ω, g.h, P.k0, g.n + 1)
    astar, bstar, ratio = sparsify_stencils(K, g.n; farfield = farfield)

    if boundary === :ying
        bnd = boundary_stencils(K, g.n; buffer = buffer)
        H = build_H_ying(g, P.ω, P.m, astar, bstar, bnd)
        S = rhs_operator_ying(g, astar, bnd)
        return SweepPreconditioner(P, astar, bstar, ratio, H, S, lu(H),
                                   collect(1:nphys(g)), mode, boundary, bnd,
                                   time() - t0, 0)
    end

    H = build_H(g, P.ω, P.m, astar, bstar; C = C)
    S = rhs_operator(g, astar)
    nst = 0
    fact = if mode === :sweep
        F = sweep_setup(g, P.ω, P.m, H; C = C, nsamp = nsamp, fronts = fronts,
                        levels = levels)
        nst = F.nstencil
        F
    else
        lu(H)
    end
    return SweepPreconditioner(P, astar, bstar, ratio, H, S, fact, phys_indices(g),
                               mode, boundary, nothing, time() - t0, nst)
end

"""
    apply_M!(y, M, r)

Apply the preconditioner: build the sparse right-hand side `f = S r`, solve
`H ũ ≈ f` by sweeping (or exactly), and restrict `ũ` to `Ω`.
"""
function apply_M!(y::Vector{ComplexF64}, M::SweepPreconditioner, r::Vector{ComplexF64})
    f = M.S * r
    ũ = M.mode === :sweep ? sweep_solve(M.fact, f) : M.fact \ f
    @inbounds for (k, i) in enumerate(M.pidx)
        y[k] = ũ[i]
    end
    return y
end
apply_M(M::SweepPreconditioner, r::Vector{ComplexF64}) =
    apply_M!(similar(r), M, r)

# so the preconditioner can go straight into IterativeSolvers as Pl or Pr
LinearAlgebra.ldiv!(y::AbstractVector, M::SweepPreconditioner, r::AbstractVector) =
    apply_M!(y, M, ComplexF64.(r))
function LinearAlgebra.ldiv!(M::SweepPreconditioner, r::AbstractVector)
    r .= apply_M(M, ComplexF64.(r))
    return r
end
Base.:\(M::SweepPreconditioner, r::AbstractVector) = apply_M(M, ComplexF64.(r))

# --------------------------------------------------------------------------- #

"""
    SolveReport

Everything the numerical tables of the paper report, plus the solution.
"""
struct SolveReport
    u::Vector{ComplexF64}
    iters::Int
    resnorms::Vector{Float64}
    converged::Bool
    relres::Float64        # true relative residual of the *dense* system
    tsetup::Float64
    tapply::Float64        # cost of one preconditioner application
    tsolve::Float64
    ratio::Float64
    nstencil::Int
end

"""
    solve(P, b; M = nothing, tol = 1e-6, restart = 20, maxiter = 400, side = :left)

Solve the dense Lippmann-Schwinger system with `IterativeSolvers.gmres`,
optionally preconditioned by `M`.

`side = :left` passes the preconditioner as `Pl`, which is MATLAB's
`gmres(A,b,restart,tol,maxit,M)` convention and the one the paper's iteration
counts are reported under; the residual GMRES monitors is then the preconditioned
one.  `side = :right` passes it as `Pr` and monitors the true residual.  Either
way, `relres` below is recomputed against the unpreconditioned system.
"""
function solve(P::LSProblem, b::Vector{ComplexF64};
               M::Union{Nothing,SweepPreconditioner} = nothing,
               tol::Real = 1e-6, restart::Int = 20, maxiter::Int = 400,
               side::Symbol = :left, verbose::Bool = false)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    tapply = 0.0
    tsetup = 0.0
    kwargs = (; reltol = float(tol), abstol = 0.0, restart = restart,
              maxiter = maxiter, log = true)
    if M === nothing
        tsolve = @elapsed u, hist = IterativeSolvers.gmres(P, b; kwargs...)
    else
        tsetup = M.tsetup
        tapply = @elapsed apply_M(M, b)
        tsolve = @elapsed u, hist = side === :left ?
            IterativeSolvers.gmres(P, b; Pl = M, kwargs...) :
            IterativeSolvers.gmres(P, b; Pr = M, kwargs...)
    end

    relres = norm(apply_A(P, u) - b) / norm(b)
    verbose && @printf("  iters = %3d   relres = %.3e   setup = %.3fs   solve = %.3fs\n",
                       hist.iters, relres, tsetup, tsolve)
    return SolveReport(u, hist.iters, Float64.(hist[:resnorm]), hist.isconverged,
                       relres, tsetup, tapply, tsolve,
                       M === nothing ? NaN : M.ratio,
                       M === nothing ? 0 : M.nstencil)
end
