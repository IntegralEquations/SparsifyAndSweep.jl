# Scenario 1 of Taus et al. on the sparsified operator H with fitted PML: a 2x2
# Cartesian decomposition with the corner at the centre of the lens, a source in
# the corner box, and the three stages of an L-sweep -- local solve, two straight
# transmissions, one diagonal transmission through an L-shaped trace.  Each box is
# compared with the global solution of H, so the cost is dominated by lu(H).
#
#   julia --project=. -t 16 examples/lsweep_transmission.jl 32 64 128
using SparsifyAndSweep, LinearAlgebra, SparseArrays, Printf
const SS = SparsifyAndSweep

# local operator on Ω_ij = [lo1,hi1]x[lo2,hi2]: 2 unstretched ε-layers, then `pad` fitted-PML layers
function localop(g, Ht, m, ω, C, lo1, hi1, lo2, hi2, pad)
    h = g.h; off = SS.offsets(Val(2)); amp = C / ω
    r1 = (lo1 - 2 - pad):(hi1 + 2 + pad); r2 = (lo2 - 2 - pad):(hi2 + 2 + pad)
    box = [(i, j) for j in r2 for i in r1]
    loc = Dict(p => k for (k, p) in enumerate(box))
    unstretched(i, j) = lo1 - 2 <= i <= hi1 + 2 && lo2 - 2 <= j <= hi2 + 2
    p1 = SS.PMLProfile(amp, pad * h, (lo1 - 3) * h, (hi1 + 3) * h)
    p2 = SS.PMLProfile(amp, pad * h, (lo2 - 3) * h, (hi2 + 3) * h)
    Ir = Int[]; Jc = Int[]; Vv = ComplexF64[]
    for p in box
        row = loc[p]
        if unstretched(p...)
            for k in nzrange(Ht, lin(g, p))
                jg = Ht.rowval[k]; j = (mod(jg - 1, ntot(g)) - g.b, div(jg - 1, ntot(g)) - g.b)
                haskey(loc, j) || continue
                push!(Ir, row); push!(Jc, loc[j]); push!(Vv, Ht.nzval[k])
            end
        else
            κ = ω * sqrt(1 - SS.mval(g, m, p))
            γ = SS.pml_stencil(κ, h, (SS.dsigma(p1, h, p[1]), SS.dsigma(p2, h, p[2])))
            for (q, t) in enumerate(off)
                j = p .+ t; haskey(loc, j) || continue
                push!(Ir, row); push!(Jc, loc[j]); push!(Vv, γ[q])
            end
        end
    end
    A = sparse(Ir, Jc, Vv, length(box), length(box))
    return (A = A, F = lu(A), box = box, loc = loc, r1 = r1, r2 = r2)
end

# polarized RHS on local problem L from traces (dicts point -> value) on Γ1, Γ2, using L's own blocks
function polarized_rhs(L, Γ1, Γ2, val)
    i1 = [L.loc[p] for p in Γ1]; i2 = [L.loc[p] for p in Γ2]
    v1 = [val[p] for p in Γ1]; v2 = [val[p] for p in Γ2]
    r = zeros(ComplexF64, length(L.box))
    r[i1] = L.A[i1, i2] * v2
    r[i2] = -L.A[i2, i1] * v1
    return r
end

function main(nn; b = 8, C = 12.0, s = 64, field = :converging, pads = (8, 16, 24))
    n = 8*nn - 1; ω = 2π*nn
    # the 2x2 CDD puts the far edge of the outer boxes at (n+1)/2 + s, and localop
    # extends each by 2 + pad layers; all of that must stay inside the padded grid
    need = 2 * (s + 2 + maximum(pads) - b) + 1
    n >= need || throw(ArgumentError(
        "ω/2π = $nn gives n = $n, too small for a 2x2 CDD of $s-point boxes with " *
        "pad up to $(maximum(pads)): needs n >= $need, i.e. ω/2π >= $(cld(need + 1, 8))"))
    g = Grid{2}(n, b); h = g.h
    m = perturbation(velocity(field, n))
    K = Kernel(Val(2), ω, h, self_weight(Val(2), ω, h), n + 1)
    astar, bstar, _ = sparsify_stencils(K, n; farfield = n)
    H = build_H(g, ω, m, astar, bstar; C = C); Ht = sparse(transpose(H))
    c = (n + 1) ÷ 2                       # CDD corner at the lens centre
    src = (c - s + 16, c - s + 16)        # source well inside Ω11
    f = zeros(ComplexF64, nunk(g)); f[lin(g, src)] = 1
    u = lu(H) \ f
    boxes = Dict(:Ω11 => (c - s + 1, c, c - s + 1, c), :Ω21 => (c - s + 1, c, c + 1, c + s),
                 :Ω12 => (c + 1, c + s, c - s + 1, c), :Ω22 => (c + 1, c + s, c + 1, c + s))
    @printf("%s medium, ω/2π=%d, 2x2 CDD of %d-point boxes, corner at lens centre (m=%.2f)\n",
            field, nn, s, m[c, c])
    for pad in pads
        Lo = Dict(k => localop(g, Ht, m, ω, C, v..., pad) for (k, v) in boxes)
        sol = Dict{Symbol,Dict{NTuple{2,Int},ComplexF64}}()
        asdict(L, x) = Dict(p => x[L.loc[p]] for p in L.box)
        # stage 1: local solve in the source box
        L = Lo[:Ω11]; fl = zeros(ComplexF64, length(L.box)); fl[L.loc[src]] = 1
        sol[:Ω11] = asdict(L, L.F \ fl)
        # stage 2: up into Ω21 (interface j = c | c+1) and right into Ω12 (interface i = c | c+1)
        L = Lo[:Ω21]
        Γ1 = [(i, c) for i in L.r1]; Γ2 = [(i, c + 1) for i in L.r1]
        sol[:Ω21] = asdict(L, L.F \ polarized_rhs(L, Γ1, Γ2, sol[:Ω11]))
        L = Lo[:Ω12]
        Γ1 = [(c, j) for j in L.r2]; Γ2 = [(c + 1, j) for j in L.r2]
        sol[:Ω12] = asdict(L, L.F \ polarized_rhs(L, Γ1, Γ2, sol[:Ω11]))
        # stage 3: diagonal into Ω22 through the L-shaped trace (bottom arm from Ω12, left arm from Ω21)
        L = Lo[:Ω22]
        bottom1 = [(i, c) for i in L.r1 if i >= c];     bottom2 = [(i, c + 1) for i in L.r1 if i >= c + 1]
        left1   = [(c, j) for j in L.r2 if j >= c + 1]; left2   = [(c + 1, j) for j in L.r2 if j >= c + 2]
        val = Dict{NTuple{2,Int},ComplexF64}()
        for p in vcat(bottom1, bottom2); val[p] = sol[:Ω12][p]; end
        for p in vcat(left1, left2);     val[p] = sol[:Ω21][p]; end
        val[(c, c)] = sol[:Ω11][(c, c)]      # corner DOF belongs to Ω11: take its value from there
        sol[:Ω22] = asdict(L, L.F \ polarized_rhs(L, vcat(bottom1, left1), vcat(bottom2, left2), val))
        # compare with the global solution on each box interior
        @printf("  pad=%2d (%.0fλ):", pad, pad / 8)
        for k in (:Ω11, :Ω21, :Ω12, :Ω22)
            lo1, hi1, lo2, hi2 = boxes[k]
            pts = [(i, j) for j in lo2:hi2 for i in lo1:hi1]
            err = norm([sol[k][p] - u[lin(g, p)] for p in pts]) / norm([u[lin(g, p)] for p in pts])
            @printf("  %s %.2e", k, err)
        end
        println()
    end
end

media = get(ENV, "LSWEEP_MEDIA", "converging,constant")
for nn in parse.(Int, ARGS), f in Symbol.(split(media, ','))
    main(nn; field = f)
end
