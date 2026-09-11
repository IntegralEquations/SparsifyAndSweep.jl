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
    nA::Base.RefValue{Int}               # forward maps applied, for accounting
end

function LSProblem(n::Integer, ω::Real, c::AbstractArray{<:Real,D}; b::Integer = 8,
                   quad::Symbol = :lattice, withconv::Bool = true) where {D}
    size(c) == ntuple(_ -> n, D) ||
        throw(DimensionMismatch("velocity field must be $(ntuple(_->n,D))"))
    grid = Grid{D}(n, b)
    k0 = self_weight(Val(D), ω, grid.h; scheme = quad)
    conv = withconv ? ToeplitzConv(Val(D), ω, grid.h, n, k0) : nothing
    return LSProblem{D,typeof(conv)}(grid, Float64(ω), perturbation(c), k0, conv, Ref(0))
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
    P.nA[] += 1
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
                        axes = (1,), boundary = :pml, buffer = 2)

The preconditioner of Algorithms 1 and 2.  `mode = :sweep` uses the moving-PML
sweeping factorisation of `H`; `mode = :direct` factorises `H` exactly with a
sparse LU, which isolates the error of the sparsification from the error of the
sweep (§4 of the paper).  `fronts = 2` (the configuration used for the paper's
tables) sweeps from both ends towards a middle slice; `fronts = 1` is the plain
left-to-right sweep of Algorithms 1-2.  `levels = 2` additionally sweeps each
slice subproblem along `x₂` — the recursive approach of §3.1, which changes the
3D setup cost from `O(b²N^{4/3})` to `O(b⁴N)` and only applies in 3D.

`axes` lists the sweeps performed per application, as signed axes (see
[`sweep_setup`](@ref)): `(1,)` is the single sweep along `x₁`; `(2,)` the same
along `x₂`; `(1, 2)` the bi-directional preconditioner of Zepeda-Núñez & Zhao
(SIAM J. Sci. Comput. 38 (2016) B866-B888, Alg. 2) — sweep along `x₁`, form the
residual, sweep it along `x₂`, add — which needs two factorisations; `(1, -1)`
with `fronts = 1` alternates the direction on one axis.  Their Alg. 3 is
`axes = (1, 2)` together with `inner > 0` and `inner_tol = 1e-3`.

`shifts` does the same for `mode = :lsweep`: one L-sweep per entry, each on a
checkerboard shifted by that fraction of a box on each axis, applied in turn to
the residual of the previous ones.  `((0, 0), (0.5, 0.5))` is the shifted
alternation: the residual of an L-sweep lies on the layers either side of its
box interfaces, and the half-box-shifted board holds those layers in the
interiors of its boxes.  `owner` applies to the unshifted board only; shifted
boards carry one box more per axis and use the row partition.

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
    inner::Int                           # inner GMRES iterations on H per application (0: one sweep)
    inner_tol::Float64
    nsweeps::Base.RefValue{Int}          # sweeps applied so far, for accounting
end

function SweepPreconditioner(P::LSProblem; C::Real = 12.0, nsamp::Integer = 200,
                             fronts::Integer = 2, mode::Symbol = :sweep,
                             boundary::Symbol = :pml, buffer::Integer = 2,
                             levels::Integer = 1,
                             axes = (1,),
                             farfield::Integer = default_farfield(P),
                             nbox::Integer = 64, pad::Integer = P.g.b,
                             boxes::Union{Nothing,Tuple{Int,Int}} = nothing,
                             comm::AbstractComm = LocalComm(),
                             owner::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
                             nworkers::Integer = Threads.nthreads(),
                             nfact::Integer = nworkers,
                             factors::Symbol = :compact,
                             shifts = ((0, 0),),
                             inner::Integer = 0, inner_tol::Real = 1e-3)
    mode in (:sweep, :direct, :lsweep) ||
        throw(ArgumentError("mode must be :sweep, :direct or :lsweep"))
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
                                   time() - t0, 0, Int(inner), Float64(inner_tol), Ref(0))
    end

    H = build_H(g, P.ω, P.m, astar, bstar; C = C)
    S = rhs_operator(g, astar)
    nst = 0
    fact = if mode === :sweep
        ax = collect(Int, axes)
        isempty(ax) && throw(ArgumentError("axes must name at least one sweep axis"))
        F = length(ax) == 1 ?
            sweep_setup(g, P.ω, P.m, H; C = C, nsamp = nsamp, fronts = fronts,
                        levels = levels, axis = ax[1]) :
            alternating_setup(g, P.ω, P.m, H; axes = ax, C = C, nsamp = nsamp,
                              fronts = fronts, levels = levels)
        nst = nstencils(F)
        F
    elseif mode === :lsweep
        sh = [Tuple(float.(s)) for s in shifts]
        isempty(sh) && throw(ArgumentError("shifts must name at least one checkerboard"))
        Fs = [lsweep_setup(g, P.ω, P.m, astar, bstar; C = C, nsamp = nsamp, nbox = nbox,
                           boxes = boxes, pad = pad, shift = s, comm = comm,
                           owner = all(iszero, s) ? owner : nothing,
                           nworkers = nworkers, nfact = nfact, factors = factors)
              for s in sh]
        F = length(Fs) == 1 ? Fs[1] : AlternatingSweep{D}(H, Fs)
        nst = nstencils(F)
        F
    else
        lu(H)
    end
    return SweepPreconditioner(P, astar, bstar, ratio, H, S, fact, phys_indices(g),
                               mode, boundary, nothing, time() - t0, nst,
                               Int(inner), Float64(inner_tol), Ref(0))
end

"One approximate (or exact) solve of the sparse system with whichever factorization was built."
inner_solve(fact::SweepFactorization, f::AbstractVector{ComplexF64}) = sweep_solve(fact, f)
inner_solve(fact::LSweepFactorization, f::AbstractVector{ComplexF64}) = lsweep_solve(fact, f)
inner_solve(fact::AlternatingSweep, f::AbstractVector{ComplexF64}) = alternating_solve(fact, f)
inner_solve(fact, f::AbstractVector{ComplexF64}) = fact \ f

"Sweeps performed by one `inner_solve`: one, or one per factorisation of an alternating sweep."
sweeps_per_solve(fact::AlternatingSweep) = length(fact.F)
sweeps_per_solve(fact) = 1

"Wraps a sweep factorization as a left preconditioner for the inner GMRES on `H`; counts every sweep it performs."
struct InnerPrec{F}
    fact::F
    nsweeps::Base.RefValue{Int}
end
LinearAlgebra.ldiv!(y::AbstractVector, P::InnerPrec, x::AbstractVector) =
    (P.nsweeps[] += sweeps_per_solve(P.fact); y .= inner_solve(P.fact, ComplexF64.(x)); y)
LinearAlgebra.ldiv!(P::InnerPrec, x::AbstractVector) =
    (P.nsweeps[] += sweeps_per_solve(P.fact); x .= inner_solve(P.fact, ComplexF64.(x)); x)

"""
    inner_apply(M, f)

`H ũ ≈ f`: one application of the sweep when `M.inner == 0`; otherwise up to
`M.inner` GMRES iterations on `H` preconditioned by the sweep, to relative
tolerance `M.inner_tol` — the two-level scheme of Zepeda-Núñez & Zhao (2016),
which confines the dense operator to the few outer iterations.
"""
function inner_apply(M::SweepPreconditioner, f::Vector{ComplexF64})
    if M.inner <= 0
        M.nsweeps[] += sweeps_per_solve(M.fact)
        return inner_solve(M.fact, f)
    end
    # the inner GMRES may stop short of its cap, and being left-preconditioned it
    # sweeps once more than it iterates (on the initial residual); the wrapper
    # counts the sweeps it actually performs
    x, _ = IterativeSolvers.gmres(M.H, f; Pl = InnerPrec(M.fact, M.nsweeps), reltol = M.inner_tol,
                                  abstol = 0.0, restart = M.inner, maxiter = M.inner,
                                  log = true)
    return x
end

"""
    apply_M!(y, M, r)

Apply the preconditioner: build the sparse right-hand side `f = S r`, solve
`H ũ ≈ f` by sweeping (or exactly), and restrict `ũ` to `Ω`.
"""
function apply_M!(y::AbstractVector{ComplexF64}, M::SweepPreconditioner, r::Vector{ComplexF64})
    f = M.S * r
    ũ = inner_apply(M, f)
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
    fgmres(P, b, M; tol, restart, maxiter)

Right-preconditioned *flexible* GMRES(restart) for the dense system, for a
preconditioner that is not a fixed linear operator (the two-level scheme,
`M.inner > 0`).  The preconditioned basis vectors are stored, so the iterate is
`x = Z y` with the `Z` that were actually applied, and the monitored residual
is the true one.  Returns `(x, iters, resnorms, converged)`.
"""
function fgmres(P::LSProblem, b::Vector{ComplexF64}, M::SweepPreconditioner;
                tol::Real, restart::Int, maxiter::Int)
    N = length(b)
    x = zeros(ComplexF64, N)
    r = copy(b)
    bnorm = norm(b)
    β = norm(r)
    resnorms = Float64[β / bnorm]
    iters = 0
    converged = β <= tol * bnorm
    w = similar(b)
    while !converged && iters < maxiter
        V = Matrix{ComplexF64}(undef, N, restart + 1)
        Z = Matrix{ComplexF64}(undef, N, restart)
        Hh = zeros(ComplexF64, restart + 1, restart)
        V[:, 1] = r ./ β
        k = 0
        y = ComplexF64[]
        while k < restart && iters < maxiter
            k += 1
            iters += 1
            Z[:, k] = apply_M(M, V[:, k])
            apply_A!(w, P, view(Z, :, k))
            @inbounds for i in 1:k                          # modified Gram-Schmidt
                Hh[i, k] = dot(view(V, :, i), w)
                w .-= Hh[i, k] .* view(V, :, i)
            end
            Hh[k+1, k] = norm(w)
            V[:, k+1] = w ./ Hh[k+1, k]
            e1 = zeros(ComplexF64, k + 1); e1[1] = β
            y = Hh[1:k+1, 1:k] \ e1                         # small least-squares problem
            res = norm(e1 - Hh[1:k+1, 1:k] * y)
            push!(resnorms, res / bnorm)
            if res <= tol * bnorm
                converged = true
                break
            end
        end
        x .+= view(Z, :, 1:k) * y
        apply_A!(r, P, x)
        r .= b .- r
        β = norm(r)
        converged = β <= tol * bnorm                       # true residual decides
        resnorms[end] = β / bnorm
    end
    return x, iters, resnorms, converged
end

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
one.  `side = :right` passes it as `Pr` and monitors the true residual.  A two-level
preconditioner (`inner > 0`) is not a fixed linear operator, so it is applied
through right-preconditioned flexible GMRES ([`fgmres`](@ref)) instead, and
`side` is ignored.  Either way, `relres` below is recomputed against the
unpreconditioned system.
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
        iters, resnorms, converged = hist.iters, Float64.(hist[:resnorm]), hist.isconverged
    elseif M.inner > 0
        # the two-level preconditioner is not a fixed linear operator: flexible GMRES
        tsetup = M.tsetup
        tapply = @elapsed apply_M(M, b)
        tsolve = @elapsed u, iters, resnorms, converged =
            fgmres(P, b, M; tol = float(tol), restart = restart, maxiter = maxiter)
    else
        tsetup = M.tsetup
        tapply = @elapsed apply_M(M, b)
        tsolve = @elapsed u, hist = side === :left ?
            IterativeSolvers.gmres(P, b; Pl = M, kwargs...) :
            IterativeSolvers.gmres(P, b; Pr = M, kwargs...)
        iters, resnorms, converged = hist.iters, Float64.(hist[:resnorm]), hist.isconverged
    end

    relres = norm(apply_A(P, u) - b) / norm(b)
    verbose && @printf("  iters = %3d   relres = %.3e   setup = %.3fs   solve = %.3fs\n",
                       iters, relres, tsetup, tsolve)
    return SolveReport(u, iters, resnorms, converged,
                       relres, tsetup, tapply, tsolve,
                       M === nothing ? NaN : M.ratio,
                       M === nothing ? 0 : M.nstencil)
end
