# SparsifyAndSweep.jl

A Julia implementation of the preconditioner of

> F. Liu and L. Ying, **Sparsify and sweep: an efficient preconditioner for the
> Lippmann–Schwinger equation**, SIAM J. Sci. Comput. **40** (2018), B379–B404.
> [doi:10.1137/17M1132057](https://doi.org/10.1137/17M1132057)

for the Lippmann–Schwinger (LS) equation in 2D and 3D,

```
u(x) + ω² ∫_Ω G(x−y) m(y) u(y) dy  =  −ω² ∫_Ω G(x−y) m(y) u_I(y) dy ,   x ∈ Ω = (0,1)^D
```

with `G(x) = (i/4)H₀⁽¹⁾(ω|x|)` (2D) or `e^{iω|x|}/(4π|x|)` (3D), and `m = 1 − 1/c²`
compactly supported in `Ω`.

The solver is parametrised on the dimension with basic types `Grid{D}`, `Kernel{D}`,
`ToeplitzConv{D}`, `SweepFactorization{D}`.  The solver makes use of the
neighbourhood `μ` which has `3^D` points, while a PML stencil fitted to `3^D − 1`
directions; and slices along `x₁` have a `(D−1)`-dimensional cross-section.
Only the Green's function, its singular quadrature correction, and a few default
parameters are dimension-specific.

The dense Nyström system `(I + ω²KM)u = g` is **sparsified** into a 9-point
stencil system `Hũ = f`, and `H` is then inverted approximately by a moving-PML
**sweeping** factorisation.  Both stages are `O(N)`; the resulting preconditioner
makes GMRES converge in a handful of iterations, essentially independently of
the frequency.

---

## Reproducing the paper's Tables 1–4

`h = λ/8` (`n = 8·ω/(2π) − 1`), PML and slice width `b = 8`, PML amplitude
`C = 12`, two sweeping fronts, GMRES(20), relative tolerance `10⁻⁶`, one core.
Raw output is in [`results/`](results/).

### Iteration counts, against the paper's Tables 1–4

`N_iter` here / `N_iter` in the paper, GMRES(20) to `10⁻⁶`:

| ω/(2π) | N | (i) converging | (ii) diverging — **Table 2** | (iii) 32 lenses — **Table 3** | (iv) random |
|---:|---:|:---:|:---:|:---:|:---:|
| 16  | 127²  | **4** / 5 | **3** / 4 | **8** / 9  | **5** / 7 |
| 32  | 255²  | **5** / 5 | **4** / 4 | **7** / 8  | **5** / 7 |
| 64  | 511²  | **6** / 5 | **4** / 5 | **9** / 9  | **6** / 8 |
| 128 | 1023² | **7** / 6 | **5** / 5 | **10** / 10 | **9** / 9 |
| 256 | 2047² | **9** / 7 | **8** / 6 | **13** / 11 | **11** / 9 |

Through ω/(2π) = 128 every entry is within one iteration of the paper, and
Table 3 matches exactly at the two largest sizes.  At 256 the gap widens to a
uniform +2, which is a `C` effect and not a field effect — see below for
details.

### Times (field (i), one core)

| ω/(2π) | N | T_setup | T_apply | T_solve | T_setup/N | T_apply/N |
|---:|---:|---:|---:|---:|---:|---:|
| 16  | 127²  | 0.82 s  | 0.021 s | 0.10 s | 5.1e−5 | 1.3e−6 |
| 32  | 255²  | 1.44 s  | 0.150 s | 0.60 s | 2.2e−5 | 2.3e−6 |
| 64  | 511²  | 3.75 s  | 0.420 s | 1.93 s | 1.4e−5 | 1.6e−6 |
| 128 | 1023² | 12.05 s | 0.887 s | 5.81 s | 1.2e−5 | 0.9e−6 |
| 256 | 2047² | 49.54 s | 2.226 s | 27.63 s | 1.2e−5 | 0.5e−6 |

Both per-unknown costs flatten out over a 1000-fold range in `N`, confirming
setup and application are `O(N)`.  The ω/(2π) = 256 runs (`N = 4190209`) need
about 13 GB of memory.


### 3D (§3.2, Tables 5–8, nonrecursive)

`h = λ/8` (`n = 8·ω/(2π) − 1`), `b = 4`, `C = 12`, two fronts, GMRES(20) to `10⁻⁶`.
Same code path as 2D — only `Val(3)` and `b` differ (`examples/tables3d.jl`).

| ω/(2π) | N | Nunk | T_setup | T_apply | (i) | (ii) | (iii) | (iv) |
|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|
| 4 | 31³ = 29 791  | 68 921  | ~15 s  | 0.59 s | **4** | **4** | **6** | **6** |
| 8 | 63³ = 250 047 | 389 017 | ~117 s | 2.9 s  | **4** | **4** | **8** | **6** |

The paper's Tables 5 and 6 give 5, 5, 5, 6 and 4, 5, 5, 5 for fields (i) and
(ii) at ω/(2π) = 4, 8, 16, 32; the counts here roughly confirm that, and are
flat in frequency, which is the more significant claim.  Setup grows 7× while
`N` grows 8.4×, consistent with the `O(b²N^{4/3})` estimate at these sizes.
The `(iii)` field uses 256 lenses in 3D (as in the paper) against 32 in 2D.

### The recursive sweep (§3.1)

Each slice subproblem of the 3D sweep is itself a quasi-2D sparse system.  The
nonrecursive method factorises it directly, `O(b³n³)` per slice and
`O(b²N^{4/3})` overall; the recursive method of ref. [20] sweeps *that* slab
along `x₂` into quasi-1D pieces instead, giving `O(b⁴N)` setup and `O(b²N)`
application.  It is the same algorithm one level down, so it is implemented as a
recursion over sweep axes and selected with `levels` (`examples/recursive_sweep.jl`):

| ω/(2π) | N | levels | T_setup | T_apply | iters | leaves | max leaf |
|---:|---:|:--:|---:|---:|:--:|---:|---:|
| 4 | 31³ | 1 | 20.0 s  | 0.58 s | 4 | 8   | 21 853 |
| 4 | 31³ | 2 | 25.8 s  | 1.16 s | 4 | 64  | 6 929  |
| 8 | 63³ | 1 | 167.1 s | 2.92 s | 4 | 16  | 69 277 |
| 8 | 63³ | 2 | **61.4 s** | 3.40 s | 5 | 256 | 12 337 |

At `N = 250 047` the recursion cuts setup by **2.7×** (167 s → 61 s), at the cost
of one extra iteration and a slightly slower apply.  That matches the paper,
which reports the recursive approach costing "zero or one more iteration".  At
`N = 29 791` it is a net loss (20 s → 26 s): the extra `γ` stencils and the
deeper bookkeeping outweigh the cheaper leaves while the quasi-2D factorisations
are still small.

The apply going *up* while the asymptotics go down is expected: `O(bN log N)` →
`O(b²N)` trades a log for a factor `b`, and the recursion replaces a few large
triangular solves with many small ones.

`levels = 1` is the default.  In 2D the slice subproblems are already quasi-1D,
so `levels = 2` there only wraps each in a one-slice sweep and buys nothing.

Two parameters that the paper does not pin down had to be got right before the
iteration counts matched; both are discussed below.

### Two sweeping fronts

Algorithms 1 and 2 as written sweep left to right, but remark 1 after Algorithm 2
notes that the radiation condition holds on all sides, and §2.5 states that the
tables were produced with *"two fronts sweeping toward the middle slice, and the
middle slice … padded with auxiliary PMLs on both sides"*.

That is what we term here the **twisted** block factorisation: eliminate
`D₁…D_{c-1}` rightwards and `D_ℓ…D_{c+1}` leftwards, meet at `D_c`, and
back-substitute outwards.  It is exact whenever the Schur complements are, and
it halves the number of approximate solution operators composed along any path
from a slice to the boundary.  The benefit therefore grows with the slice
count: at ω/(2π) = 16 it is a wash, at 128 it is worth two or three iterations.

`fronts = 1` recovers the plain sweep of Algorithms 1–2.

### The PML amplitude C

`σ_max = C/ω`, so `C` is the total attenuation in nepers across the `b` layers
of a PML; the paper does not specify this paramter.  It matters much more than
one would guess (`examples/pml_amplitude_study.jl`, iteration counts):

```
field = converging                          field = diverging
ω/2π   C=6  C=8  C=10 C=12 C=14 C=16  slices    C=6  C=8  C=10 C=12 C=14 C=16
32      5    5    5    5    5    5     32        4    4    4    4    4    4
64      7    6    6    6    6    6     64        6    5    4    4    5    5
128    11    8    7    7    7    7    128       11    8    6    5    5    6

field = multi                               field = random
ω/2π   C=6  C=8  C=10 C=12 C=14 C=16  slices    C=6  C=8  C=10 C=12 C=14 C=16
32      7    7    7    7    8    8     32        5    5    5    5    5    5
64      9    9    9    9    9    9     64        7    7    6    6    6    6
128    12   10   10   10   10   10    128       12   10    9    9    9    9
```

At 32 slices the choice is irrelevant; at 128 slices it is worth a factor of two
in iterations.  The mechanism is compounding: every moving PML leaves a small
residual reflection, and the sweep composes `ℓ ≈ ntot/b` of them, so it is the
number of *slices* rather than the frequency as such that sets how good each
absorber must be.  `C` has no effect at all on the accuracy of `H` as a
discretisation (the direct-solver error is 7.411e−3 at both `C = 8` and
`C = 12`) — it only buys sweep quality.

`C = 12` is optimal or tied-optimal in all twelve cells, which is why it is the
default.  The trend continues past 128: at ω/(2π) = 256 (256 slices) raising `C`
to 16 buys another iteration on the three fields that are not already saturated,

| ω/(2π) = 256 | (i) | (ii) | (iii) | (iv) |
|:--|:--:|:--:|:--:|:--:|
| `C = 12`  | 9 | 8 | 13 | 11 |
| `C = 16`  | 8 | 7 | 13 | 10 |
| paper     | 7 | 6 | 11 |  9 |

so a mildly slice-dependent `C ≈ max(10, 4log₂ℓ − 16)` tracks the observed
optima across the whole range.  This is just an empirical fit to five sizes;
the default is kept as the constant `C = 12`; it may be worthwhile to pass `C`
explicitly if you are running many more slices than this.

### The velocity fields are underspecified — but the preconditioner is robust

The paper describes its fields qualitatively ("a converging Gaussian centred at
(0.5,0.5)", "32 randomly placed converging Gaussians with narrow width") and its
colour bars show `c ∈ [0.7, 1.3]`, but the Gaussian widths are not given.  A
sensitivity study (`examples/field_width_study.jl`) shows the iteration count is
essentially blind to them:

```
field = diverging, gaussian                 field = multi, gaussian
width    ω/2π=16  ω/2π=32  ω/2π=64          width    ω/2π=16  ω/2π=32  ω/2π=64
0.050    3        4        5                0.015    8        7        8
0.075    3        4        5                0.020    7        7        9
0.100    3        4        5                0.030    7        8        9
0.125    3        4        5                0.040    7        8        9
0.150    3        4        5
```

Even the compactly supported `bump` profile of radius 0.4 — considerably wider
than any of these — gives 4, 4, 5.  So the reproduction does not depend on
guessing the paper's widths.  Defaults here are `σ = 0.1` for the single lens
and `σ = 0.02` for the 32 narrow ones, with amplitude 0.3.

### Why so few iterations

`none` is plain GMRES(20) on the dense system; `H-exact` replaces the sweep by
an exact sparse LU of `H`; `sweep` is the real method (`examples/preconditioner_benefit.jl`).

```
field (i) converging                         field (iii) 32 narrow lenses
ω/2π   N       none  T_none  H-exact sweep   none  T_none  H-exact sweep
8      63^2     12    0.0 s    3       4      24    0.0 s    3       8
16     127^2    17    0.0 s    3       4      39    0.1 s    3       8
32     255^2    33    0.3 s    3       5      75    1.1 s    3       7
64     511^2   136    8.9 s    3       6     197   18.4 s    3       9
```

The plain count grows roughly like `ω^1.3`; the preconditioned one does not.  On
the harder field (iii) at ω/(2π) = 64 that is 197 iterations and 18.4 s against
9 iterations and 6.5 s including setup.

Inverting the sparsified `H` *exactly* costs **3 iterations at every frequency**.
So the sparsification is essentially a perfect surrogate for the dense LS
operator, and all of the residual growth is the moving-PML approximation of the
Schur complements — which is exactly the knob `C`, `b` and `fronts` control.

### Cost scaling

See the timing columns above: `T_setup/N` and `T_apply/N` both flatten as `N`
grows by a factor of 64, exhibiting `O(N)` complexity.  The sweeping
factorisation dominates memory — about 14 GB at `N = 4.2·10⁶` with `b = 8`.

### Discretisation accuracy

The Nyström weights are validated against the *exact* volume potential of a
radially symmetric `C^∞` density, computed from Graf's addition theorem:

| n | `:lattice` (default) | rate | `:disk` | rate |
|---:|---:|---:|---:|---:|
| 63  | 6.0e−5 | –    | 3.7e−3 | –   |
| 127 | 3.7e−6 | 4.04 | 8.9e−4 | 2.05 |
| 255 | 2.3e−7 | 3.99 | 2.2e−4 | 2.02 |
| 511 | 1.6e−8 | 3.85 | 5.4e−5 | 2.01 |

and the sparsified system, used as a *direct* solver (§4 of the paper),
reproduces the dense LS solution to 0.74 % at ω/(2π) = 8, n = 63.

---

## Unstructured discretisations (in `Inti.jl`)

This package is deliberately limited to structured Cartesian grids: it knows
nothing about meshes, quadratures or elements.  Coupling to *unstructured*
Lippmann–Schwinger discretisations is done with the preconditioner,

```
P = I + T_C^Q (S^(C) − I) T_Q^C
```

which is wrapped in a package extension `IntiSparsifyAndSweepExt` in
[`Inti.jl`](https://github.com/IntegralEquations/Inti.jl)
(`ext/IntiSparsifyAndSweepExt/`).  It loads automatically when both packages
are loaded:

```julia
using Inti, SparsifyAndSweep
const SAS = Base.get_extension(Inti, :IntiSparsifyAndSweepExt)
```

| piece (in the extension) | what it does |
|:--|:--|
| `PhysicalGrid{D}` | places the unit-cube grid in a physical box by a *uniform* affine map |
| `interpolation_matrix` | `T_C^Q`, multilinear (`2^D` points, rows sum to 1) |
| `renormalized_transpose` | `T_Q^C`, Approach II — preserves constants |
| `HybridPreconditioner` | the operator above; takes any `to_grid`/`from_grid` |

`S^(C)` implements the action `b ↦ C⁻¹(Ab)`, which is what a
`SweepPreconditioner` computes — its `S` is that `A` and its `H` is that `C`.
The package allows three options: `mode = :direct` (≡ Ying 2015), `mode =
:sweep` (`O(N)`) and `levels = 2` (see above).

The core routines in this repository keeps the hard-coding `Ω = (0,1)^D` and
uses rescaling to treat the physical domain in the `Inti.jl` discretization:
under `x = Lx′` the equation is invariant with `ω′ = ωL` (2D:
`G(Lr)=(i/4)H₀(ωLr)` and `dy = L²dy′`; 3D: `G(Lr) = L⁻¹G′` and `dy = L³dy′`).
One implementation note is the `LSProblem(...; withconv = false)` possibility,
which skips building the FFT operator that a preconditioner-only use never
applies.

### Verified against the real Inti pipeline

`Inti.jl/docs/src/examples/lippmann_schwinger_sparsify_and_sweep.jl` drives a
genuine Vioreanu–Rokhlin / DIM discretisation of a penetrable disk (`k = 5`,
`η = 1.25`, `N_Q = 157 686`, `reltol = 1e-8`), checked against the exact Mie
series:

| pts/λ on the Cartesian grid | `N_cart` | `b` | PML/λ | iters | err vs Mie |
|---:|---:|---:|---:|---:|---:|
| 50.3 | 78 400 | 50 | 0.99 | **7** | 4.3e−9 |
| 15.9 | 7 744  | 16 | 1.01 | 9 | 3.5e−9 |
| 8.0  | 1 936  | 8  | 0.99 | 13 | 2.7e−9 |

against **29** unpreconditioned, and **6** for the hybrid solver's own `S^(C)` on
the full hybrid system.  The Cartesian grid need not match the unstructured
mesh — the transfer operators couple two independent resolutions — so the last
row buys most of the benefit from a grid 80× smaller than the mesh.

**The PML thickness is the parameter to watch.**  `b = 8` is a full wavelength at
the package's design point `h = λ/8`, but the hybrid solver runs its Cartesian
grid at ~50 points per wavelength, where `b = 8` is only `0.16λ` and the count
degrades from 7 to 12.  So `b` should track the *grid's* points-per-wavelength,
and the box must be padded by ~2 wavelengths rather than by a fraction of the
scatterer size.

### Implementation notes

* **Quadrature correction.**  The paper cites Duan–Rokhlin for a correction of
  order `O(h⁴log²(1/h))`.  Here `G` is split as
  `G(r) = −(1/2π)log(r)J₀(ωr) + S(r)` with `S` analytic, and the square-lattice
  correction is applied to the pure log kernel:

  ```
  k₀ = h²[ S(0) − (log h + c_log)/(2π) ],   S(0) = i/4 − (log(ω/2)+γ)/(2π)
  ```

  with `c_log = −1.3105329259115095`, the 2D square-lattice constant of
  `∫log|y|φ − h²Σ_{j≠0}log|jh|φ(jh) = h²(log h + c_log)φ(0) + O(h⁴)`.
  Measured convergence is fourth order (table above).  A cruder `:disk` weight
  (`∫_{|y|<h/√π} G`) is kept for comparison and is only second order.

* **Stencil solves.**  `α` is the smallest right singular vector of the
  `((2n+1)²−9) × 9` matrix `W[j,s] = k_{s−j}`.  `W` is never formed: it is
  reduced by a chunked Householder QR to a `9×9` triangular factor.  Forming the
  `9×9` Gram matrix instead would square the condition number and destroy `α`
  (`σ_min/σ_max ≈ 3e−5` here, so `σ_min²/σ_max²` is within `10⁻⁹` of round-off).

* **Auxiliary-PML stencil cache.**  Stencils in the moving PML depend on the
  point only through the local frequency `ω√(1−m)` and the two *increments*
  `σ_d(x_d ± h) − σ_d(x_d)` of the complex stretching, so they are cached on
  `(frequency sample, side, PML width, position in the ramp, x₂ position)`.  A
  few thousand `9×8` SVDs suffice for `N = 10⁶`.  `nsamp = 200` frequency
  samples is already converged — at ω/(2π) = 128, raising it to 1023 or 4000
  does not change the iteration count.

* **Ordering.**  Unknowns are numbered `x₁`-fastest, so each auxiliary system is
  a narrow band matrix along `x₂` and UMFPACK factorises it in `O(b²n)`.

* **Not implemented.**  The parallelisation of §5 (the auxiliary factorisations
  and the two fronts are independent and could run concurrently; everything here
  is single-threaded), and the general-domain variant of Ying (2015) §4.

---

## Usage

```julia
using SparsifyAndSweep

f  = 32
n  = 8f - 1                       # h = λ/8
ω  = 2π * f
c  = velocity(:converging, n)     # or :diverging, :multi, :random, :constant

P  = LSProblem(n, ω, c; b = 8)    # dense LS system, K applied by FFT
M  = SweepPreconditioner(P)       # Algorithm 1; C = 12, fronts = 2
b  = rhs(P, plane_wave(n, ω))     # g = -ω² K M u_I

r  = solve(P, b; M = M, tol = 1e-6, restart = 20)
r.iters      # 5
r.relres     # 6.6e-8
u  = reshape(r.u, n, n)           # scattered field on Ω
```

The same code in 3D — only `Val(3)` and the smaller `b` change:

```julia
n, ω = 31, 2π * 4
c = velocity(Val(3), :converging, n)
P = LSProblem(n, ω, c; b = 4)              # Grid{3} is inferred from c
M = SweepPreconditioner(P)
r = solve(P, rhs(P, plane_wave(Val(3), n, ω)); M = M, tol = 1e-6)
r.iters      # 4
u = reshape(r.u, n, n, n)
```

Useful parameters on `SweepPreconditioner`:

| keyword | default | meaning |
|:--|:--|:--|
| `C` | `12.0` | PML amplitude, `σ_max = C/ω` |
| `fronts` | `2` | `2` = sweep from both ends, `1` = plain left-to-right |
| `nsamp` | `200` | local-frequency samples for the moving-PML stencils |
| `mode` | `:sweep` | `:direct` inverts `H` exactly with a sparse LU (§4) |
| `levels` | `1` | `2` sweeps each slice subproblem again along `x₂` (§3.1, 3D only) |
| `boundary` | `:pml` | `:ying` uses the one-sided ∂Ω stencils of Ying (2015) instead; requires `mode = :direct` |
| `buffer` | `2` | for `boundary = :ying`: layers next to ∂Ω where `m` must vanish |

### The two radiation conditions: `boundary = :pml` vs `boundary = :ying`

`boundary` selects how the radiation condition is discretised.  This is the one
place where the 2018 paper departs from

> L. Ying, *Sparsifying preconditioner for the Lippmann–Schwinger equation*,
> Multiscale Model. Simul. **13** (2015), 644–660 — ref. [31] of the 2018 paper,

and both are implemented here so the difference can be measured; everything
else is shared. This means the same dense Lippmann–Schwinger system, the same interior
`α`/`β` stencils (Ying's eq. (12) is eq. (11) of the 2018 paper up to one ring of
the index set, and his `C(i,μ(i)) := A(i,μ(i))K(μ(i),μ(i))` is `β* = α*K_{μ,μ}`),
and the same exact sparse LU of the surrogate.

| | `:ying` (2015) | `:pml` (2018, default) |
|:--|:--|:--|
| grid | `n²`, no extension | `(n+2+2b)²`, `b` PML layers |
| boundary rows | one-sided stencils `B` on ∂Ω with `μ(i)` clipped to Ω (6 points on an edge, 4 at a corner), annihilating the half-plane `Eₙ` and quarter-plane `Cₙ` Green's-function sets | `γ` stencils fitted to complex-stretched plane waves, zero Dirichlet on the outer ring |
| free parameter | `buffer` — layers next to ∂Ω where `m` must vanish | `C` — PML amplitude |
| usable with the sweep | no | yes |

`examples/boundary_comparison.jl`, with `H` inverted exactly in both cases.
`err` is the surrogate used as a *direct* solver against the true dense
solution; `iters` is GMRES(20) to `10⁻⁶` with `H⁻¹` as the preconditioner:

```
field (i) converging                          field (iii) 32 narrow lenses
w/2pi  N        PML err   it  Ying err  it    PML err   it  Ying err  it
8      63^2     7.41e-3   3   2.39e-2   3     4.28e-3   3   1.69e-2   4
16     127^2    1.28e-2   3   4.43e-2   4     8.90e-3   3   3.17e-2   4
32     255^2    2.79e-2   3   9.23e-2   4     1.48e-2   3   5.76e-2   4
64     511^2    5.00e-2   3   1.70e-1   4     3.87e-2   3   1.25e-1   4
```

We observe that the PML surrogate is **3.2–3.9× more accurate** as a direct
solver, uniformly in ω and in both fields, and it leads to a preconditioned
iteration count of 3 at every frequency where the Ying preconditioner needs 4,
a minor margin.  On the other hand, Ying's boundary is perfectly serviceable,
and it uses fewer unknowns (`261121` vs `279841` at n = 511, though `3969` vs
`6561` at n = 63, since the PML border is a fixed `b` layers).

Raising Ying's own `buffer` helps monotonically but never really completely
closes the gap:

```
w/2pi  N        b=2       b=4       b=8       (:pml)
8      63^2     2.39e-2   1.69e-2   1.22e-2   7.41e-3
16     127^2    4.43e-2   3.61e-2   2.94e-2   1.28e-2
32     255^2    9.23e-2   8.24e-2   7.31e-2   2.79e-2
64     511^2    1.70e-1   1.59e-1   1.49e-1   5.00e-2
```

#### Cost: when does the sweep actually win?

`examples/cost_crossover.jl`, all three inverting a surrogate of the *same*
dense system:

```
w/2pi  N        variant     Nunk      T_setup  T_apply  iters
16     127^2    PML+sweep   21025     0.34     0.008    4
16     127^2    PML+LU      21025     0.21     0.006    3
16     127^2    Ying+LU     16129     0.20     0.007    4
32     255^2    PML+sweep   74529     1.24     0.072    5
32     255^2    PML+LU      74529     1.36     0.037    3
32     255^2    Ying+LU     65025     1.27     0.043    4
64     511^2    PML+sweep   279841    3.47     0.249    6
64     511^2    PML+LU      279841    3.93     0.155    3
64     511^2    Ying+LU     261121    4.91     0.188    4
128    1023^2   PML+sweep   1083681   11.86    1.047    7
128    1023^2   PML+LU      1083681   21.49    0.657    3
```

**The accuracy margin is not the real reason for the new stencil.**  Ying's `B`
rows are built from the free-space kernel `K` and are valid only where `m ≡ 0`,
which is why they need the buffer assumption and why they can only sit on the
outer boundary.  The modified-plane-wave stencils are built from the *local*
frequency `ω√(1−m(x))`, so they can be placed anywhere in the medium — and that
is exactly what the moving PML of §2.3 does, planting a fresh absorber in front
of every slice.  Without a boundary stencil that works in a heterogeneous
medium there is no `O(N)` sweep, only the `O(N^{3/2})` nested-dissection
factorisation.  The option `boundary = :ying` is therefore restricted to `mode
= :direct` here.

### Running things

```bash
julia --project=. test/runtests.jl                        # 2D + 3D, ~2.5 min
julia --project=. examples/tables.jl 128                  # Tables 1–4
julia --project=. examples/preconditioner_benefit.jl      # with/without
julia --project=. examples/pml_amplitude_study.jl         # the C study
julia --project=. examples/field_width_study.jl           # lens-width study
julia --project=. examples/parameter_study.jl 64          # b / C / nsamp / fronts
julia --project=. examples/boundary_comparison.jl 64      # :pml vs :ying
julia --project=. examples/cost_crossover.jl 128          # sweep vs exact LU
julia --project=. examples/tables3d.jl 8                 # 3D, Tables 5–8
julia --project=. examples/recursive_sweep.jl 8           # levels = 1 vs 2

julia --project=. examples/highfreq.jl diverging 256       # one big run
```

The ω/(2π) = 256 runs need about 14 GB of RAM.

## Dependencies

`FFTW`, `IterativeSolvers`, `SpecialFunctions`, and the standard libraries
`LinearAlgebra`, `SparseArrays`, `Random`, `Printf`.

`LSProblem` implements `size`/`eltype`/`mul!` and `SweepPreconditioner`
implements `ldiv!`.
