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

### Iteration counts, cf. Liu-Ying Tables 1–4

`N_iter` here / `N_iter` in the paper, GMRES(20) to `10⁻⁶`:

| ω/(2π) | N | (i) converging | (ii) diverging — **Table 2** | (iii) 32 lenses — **Table 3** | (iv) random |
|---:|---:|:---:|:---:|:---:|:---:|
| 16  | 127²  | **4** / 5 | **3** / 4 | **8** / 9  | **5** / 7 |
| 32  | 255²  | **5** / 5 | **4** / 4 | **7** / 8  | **5** / 7 |
| 64  | 511²  | **6** / 5 | **4** / 5 | **9** / 9  | **6** / 8 |
| 128 | 1023² | **7** / 6 | **5** / 5 | **10** / 10 | **9** / 9 |
| 256 | 2047² | **9** / 7 | **8** / 6 | **13** / 11 | **11** / 9 |

Through ω/(2π) = 128 every entry is within one iteration of the paper, and
Table 3 matches it exactly at the two largest sizes.  At 256 the performance
gap widens to a uniform +2; this is a `C` effect and not, rather, a field
effect — see below for details.

### Times (field (i), one core)

| ω/(2π) | N | T_setup | T_apply | T_solve | T_setup/N | T_apply/N |
|---:|---:|---:|---:|---:|---:|---:|
| 16  | 127²  | 0.82 s  | 0.021 s | 0.10 s | 5.1e−5 | 1.3e−6 |
| 32  | 255²  | 1.44 s  | 0.150 s | 0.60 s | 2.2e−5 | 2.3e−6 |
| 64  | 511²  | 3.75 s  | 0.420 s | 1.93 s | 1.4e−5 | 1.6e−6 |
| 128 | 1023² | 12.05 s | 0.887 s | 5.81 s | 1.2e−5 | 0.9e−6 |
| 256 | 2047² | 49.54 s | 2.226 s | 27.63 s | 1.2e−5 | 0.5e−6 |

This test confirms setup and application are `O(N)`.  The ω/(2π) = 256 runs (`N
= 4190209`) need about 13 GB of memory.


### 3D (§3.2, Tables 5–8, nonrecursive)

`h = λ/8` (`n = 8·ω/(2π) − 1`), `b = 4`, `C = 12`, two fronts, GMRES(20) to `10⁻⁶`.
Same code path as 2D — only `Val(3)` and `b` differ (`examples/tables3d.jl`).

| ω/(2π) | N | Nunk | T_setup | T_apply | (i) | (ii) | (iii) | (iv) |
|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|
| 4 | 31³ = 29 791  | 68 921  | ~15 s  | 0.59 s | **4** | **4** | **6** | **6** |
| 8 | 63³ = 250 047 | 389 017 | ~117 s | 2.9 s  | **4** | **4** | **8** | **6** |

Liu-Ying Tables 5 and 6 give 5, 5, 5, 6 and 4, 5, 5, 5 for fields (i) and
(ii) at ω/(2π) = 4, 8, 16, 32; the counts here roughly confirm that, and are
flat in frequency, which is the more significant claim.  Setup grows 7× while
`N` grows 8.4×, consistent with the `O(b²N^{4/3})` estimate at these sizes.
The `(iii)` field uses 256 lenses in 3D (as in the paper) versus 32 in 2D.

### Recursive sweeps

Each slice subproblem of the 3D sweep is itself a quasi-2D sparse system.  The
nonrecursive method factorises it directly, `O(b³n³)` per slice and
`O(b²N^{4/3})` overall; the recursive version sweeps *that* slab
along `x₂` into quasi-1D pieces instead, giving `O(b⁴N)` setup and `O(b²N)`
application costs.  It is the same algorithm one level down, so it is implemented as a
recursion over sweep axes and selected with `levels` (`examples/recursive_sweep.jl`):

| ω/(2π) | N | levels | T_setup | T_apply | iters | leaves | max leaf |
|---:|---:|:--:|---:|---:|:--:|---:|---:|
| 4 | 31³ | 1 | 20.0 s  | 0.58 s | 4 | 8   | 21 853 |
| 4 | 31³ | 2 | 25.8 s  | 1.16 s | 4 | 64  | 6 929  |
| 8 | 63³ | 1 | 167.1 s | 2.92 s | 4 | 16  | 69 277 |
| 8 | 63³ | 2 | **61.4 s** | 3.40 s | 5 | 256 | 12 337 |

At `N = 250 047` the recursion cuts setup by **2.7×** (167 s → 61 s), at the
cost of one extra iteration and a slightly slower apply.  That matches the
paper, which reports the recursive approach costing "zero or one more
iteration". At `N = 29 791` it gives a net loss in solve time (20 s → 26 s):
the extra `γ` stencils and the deeper bookkeeping outweigh the cheaper leaves
while the quasi-2D factorisations are still small.

The application cost going *up* while the asymptotics improve is expected:
`O(bN log N)` → `O(b²N)` trades a log for a factor `b`, and the recursion
replaces a few large triangular solves with many small ones.

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

`σ_max = C/ω`, so `C` is the total attenuation in units of nepers (meaning an
attenuation factor of e^{-C}) across the `b` layers of a PML; the paper does
not specify this parameter.  It matters much more than one would guess
(`examples/pml_amplitude_study.jl`, iteration counts):

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

`C = 12` is optimal or tied-optimal in all twelve cells, so it is set as the
default.  The trend continues past 128: at ω/(2π) = 256 (256 slices) raising
`C` to 16 buys another iteration on the three fields that are not already
saturated,

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

## L-sweeps: a parallel inner solver

The layered sweep applies its factorization one slice after another.  With two
fronts (as outlined above) the critical path is one slab solve per slice, so a
*single* right-hand side never occupies more than two cores, however many are
available, and the distributed version would spend its ranks on the slabs
rather than on the sweep.  L-sweeps provides the preconditioner of

> M. Taus, L. Zepeda-Núñez, R. J. Hewett, L. Demanet, **L-Sweeps: a scalable,
> parallel preconditioner for the high-frequency Helmholtz equation**, J. Comput.
> Phys. **420** (2020) 109706,

as a second inner solver for the sparsified system `H`, selected with
`mode = :lsweep`.  Sparsification followed by polarized traces on the sparse
system was first done, in layered form, by Zepeda-Núñez & Zhao (SISC **38**
(2016) B866–B888); L-sweeps is the Cartesian-decomposition version, which
propagates information in 90° cones and reaches `O(N/p)` parallel complexity
for one right-hand side on `p = O(n)` ranks.

### L-sweep construction

* **Cartesian decomposition.**  `Ω^{h+η}` is cut into `q x r` boxes.  Each box
  is extended by two unstretched layers and then by `pad` layers of the fitted
  modified-plane-wave PML rows of §2.2.2, at the local frequency `ω√(1−m)`, on
  all four sides — the rows the moving PML plants inside the medium, now also at
  corners.  Boxes on the outer boundary keep the global PML.  A local matrix is
  assembled from `α`, `β`, `γ` alone; `H` is never needed, so every rank builds
  and factorises only the boxes it owns.
* **Polarization.**  Given the values of a field on the two grid layers
  `Γ₁ | Γ₂` either side of an interface, the local solve with right-hand side
  `(0, A_{Γ₁Γ₂} v_{Γ₂}, −A_{Γ₂Γ₁} v_{Γ₁}, 0)` reproduces the field beyond the
  interface and vanishes behind it (TZNHD eq. 8–9).  This is an algebraic identity
  for any block-tridiagonal splitting, so it holds for `H` verbatim: the 9-point
  stencil couples only adjacent layers.  On the assembled `H` it holds to `1e-13`
  for straight and for L-shaped interfaces.
* **Sweep.**  Stage 1 solves every box with its own part of the source; stage 2
  sweeps right, left, up and down along rows and columns; stage 3 sweeps along
  the four diagonals through L-shaped traces (their §2.1.3, "scenario 3", which
  covers arbitrary sources because the interfaces run between grid lines).  A
  box keeps its field as nine components — own, four cardinal, four diagonal —
  and the trace handed to a neighbour sums exactly the components carrying the
  sources of the relevant half-plane or quadrant, preventing double counting.
* **Dataflow.**  A node is (box, direction).  It activates when its trace data
  have all arrived, locally or by message, and its activation releases the
  traces that depend on it.  Within a rank a pool of tasks executes ready nodes,
  so rows, columns and anti-diagonals overlap as far as the dependencies allow;
  across ranks each trace is one small message.  The same code runs on threads,
  on MPI ranks, or on both, and every node's arithmetic depends only on its
  inputs in a fixed order, so the three agree **bitwise** (tested: 1 vs 8
  workers, and 3 ranks x 2 threads vs one process).

(Technical aside: two conventions turned out to matter, both by a factor of a
thousand in the error of one transmission.  The stretching of a fitted PML pad
must vanish on the *first* pad row and start inside the pad, as the moving PML
does; putting the zero on the last interior row gives 15 % instead of `1e-4`.
And the corner degree of freedom of an L-shaped trace belongs to the diagonal
neighbour and must take its value from there; taking it from a cardinal
neighbour's polarized field, which vanishes at that point, leaves a
pad-independent 1 % error.)

### Usage

```julia
M = SweepPreconditioner(P; mode = :lsweep, nbox = 64, pad = 8)   # run julia with -t 16
r = solve(P, b; M = M)

M2 = SweepPreconditioner(P; mode = :lsweep, inner = 6)            # two-level (see below)
M3 = SweepPreconditioner(P; mode = :lsweep, shifts = ((0, 0), (0.5, 0.5)))   # + a second L-sweep on a half-box-shifted board

F = lsweep_setup(P.g, P.ω, P.m, M.astar, M.bstar; boxes = (8, 8), pad = 8)
ũ = lsweep_solve(F, f)                                            # H ũ ≈ f directly
```

`nbox` is the target box side in grid points (`boxes = (q, r)` fixes the count),
`pad` the PML thickness; `nworkers` (default `Threads.nthreads()`) the number of
tasks executing the sweep.  `factors = :compact` (the default) re-stores each
box's LU as single-precision CSC with 32-bit indices, half UMFPACK's bytes, which
is what the application streams; `:umfpack` keeps UMFPACK's own factors.
Distributed:

```julia
using MPI, SparsifyAndSweep
comm = MPIComm()                                                  # MPI.COMM_WORLD
M = SweepPreconditioner(P; mode = :lsweep, comm = comm)           # rows of boxes over the ranks
r = solve(P, b; M = M)                                            # identical on every rank
```

Three dimensions use beams — a `q x q` decomposition of the `(x₁, x₂)` plane with
every box spanning the whole of `x₃`, global PML included — through the same
keywords; `nfact` caps the factorisations in flight during setup, which matters
there because a beam in flight holds about four times its final size in UMFPACK
transients:

```julia
M = SweepPreconditioner(P3; mode = :lsweep, boxes = (4, 4), pad = 4, nfact = 8)  # P3 a 3D problem
```

Boxes are assigned row by row (`row_partition`, the paper's Fig. 18; pass
`owner = ...` for another assignment), so the horizontal sweeps never
communicate and columns and diagonals pipeline across ranks.  In this first
version every rank holds the full vectors: the preconditioner is distributed,
the outer GMRES and the FFT of the dense operator run replicated.

`inner = k` runs up to `k` GMRES iterations on `H` inside every application,
preconditioned by the sweep (the two-level scheme of Zepeda-Núñez & Zhao).  That
makes the preconditioner slightly nonlinear, so `solve` then switches to a
right-preconditioned *flexible* GMRES (`fgmres`), which stores the preconditioned
basis and monitors the true residual; with plain GMRES the iteration reports
convergence while the true residual stalls near `1e-6`.

### Results

Converging lens, `h = λ/8`, global PML width `b = 8`, `C = 12`, GMRES(20) to
`1e-6`, one process with 16 threads on a 16-core desktop
(`examples/lsweep_compare.jl`, output in `results/lsweep_compare.txt`).  Boxes of
about 64 points, so the CDD has `ntot/64` boxes per side, each box padded by 8
points — one wavelength at this `h` — unless the row says otherwise.  Each row was
measured in a process of its own, so the peak resident set belongs to that row
alone:

| ω/2π | N | inner solver | T_setup | T_apply | iters | T_solve | peak |
|---:|---:|:--|---:|---:|:--:|---:|---:|
| 32  | 255²  | layered sweep      | 1.08 s | 0.036 s | 5 | 0.25 s | 0.7 GB |
|     |       | L-sweeps           | 0.50 s | 0.024 s | 4 | 0.18 s | 0.9 GB |
|     |       | L-sweeps, inner 6  | 0.46 s | 0.075 s | 3 | 0.27 s | 0.9 GB |
| 64  | 511²  | layered sweep      | 1.99 s | 0.131 s | 6 | 1.03 s | 1.4 GB |
|     |       | L-sweeps           | 1.24 s | 0.115 s | 5 | 0.68 s | 1.8 GB |
|     |       | L-sweeps, inner 6  | 1.25 s | 0.398 s | 3 | 1.09 s | 1.9 GB |
| 128 | 1023² | layered sweep      | 6.22 s | 0.551 s | 7 | 4.55 s | 4.1 GB |
|     |       | L-sweeps           | 3.33 s | 0.393 s | 7 | 3.61 s | 5.6 GB |
|     |       | L-sweeps, pad 16   | 5.03 s | 0.595 s | 6 | 4.26 s | 7.2 GB |
|     |       | L-sweeps, inner 6  | 3.30 s | 1.531 s | 3 | 4.95 s | 5.7 GB |
| 256 | 2047² | layered sweep      | 29.36 s | 2.059 s | 9 | 23.99 s | 15.1 GB |
|     |       | L-sweeps           | 11.10 s | 1.554 s | 14 | 28.34 s | 20.5 GB |
|     |       | L-sweeps, pad 16   | 15.36 s | 2.239 s | 10 | 27.95 s | 27.0 GB |
|     |       | L-sweeps, pad 24   | 22.53 s | 3.042 s | 8 | 29.70 s | 34.9 GB |
|     |       | L-sweeps, inner 6  | 11.14 s | 9.245 s | 3 | 27.98 s | 20.5 GB |

L-sweeps sets up 2.0–2.6x faster at every size although it does nine local solves
per box against two slab solves per slice: at ω/2π = 256 it factorises 1024 local
problems of 7 140 unknowns where the layered sweep factorises 256 slabs of up to
51 625.  On time to solution with setup included L-sweeps wins at every size
measured, but the margin narrows steadily: 1.95x at ω/2π = 32, then 1.57x, 1.55x
and 1.35x at 256.

**What methods are threaded.**  `src/sweep.jl` contains no threading
constructs at all.  The layered sweep is serial in both stages: its application is
a sequential recurrence, which is inherent, but its factorisation loop over slices
writes only into per-slice arrays, so it is embarrassingly parallel and simply is
not threaded.  Speedups from 1 to 16 threads, BLAS pinned to the Julia thread
count so a `-t 1` run cannot quietly use all 16 cores inside UMFPACK
(`examples/thread_scaling.jl`, output in `results/thread_scaling.txt`):

| ω/2π | layered: fact | layered: apply | L-sweeps: fact | L-sweeps: apply |
|---:|:--:|:--:|:--:|:--:|
| 32  | 1.03x | 0.97x | 1.10x | 4.17x |
| 64  | 1.03x | 1.05x | 2.00x | 6.40x |
| 128 | 0.96x | 1.01x | 2.30x | 6.38x |
| 256 | 0.93x | 0.98x | 3.06x | 6.39x |

The layered columns are flat, slightly below one at 256.  The dense operator is
serial too, for a duller reason: the package never calls `FFTW.set_num_threads`,
so the FFT convolution applied once per GMRES iteration costs the same at every
thread count.

The layered sweep is fast regardless because it does far less work.  On one core
at ω/2π = 256 one layered application costs 1.97 s against 9.01 s for one L-sweeps
application, a factor of 4.6; at their respective iteration counts the layered
sweep reaches the answer in 22.6 + 9 x 1.97 = 40.3 s against
17.9 + 14 x 9.01 = 144.1 s, or 3.6x faster.  Serial factorisation costs are close,
22.6 s against 17.9 s.  So every L-sweeps advantage in the table above is bought
with the other fifteen cores: this compares a parallel implementation against a
serial one, and what L-sweeps really trades is ~4.6x more application work for
parallelism.

**The absorber may need to grow with the frequency (with complexity
implications).**  Consider a pad held fixed in wavelengths:

| ω/2π | layered sweep | pad 1λ | pad 2λ | pad 3λ |
|---:|:--:|:--:|:--:|:--:|
| 32  | 5 | 4  | 4  | 4 |
| 64  | 6 | 5  | 5  | 5 |
| 128 | 7 | 7  | 6  | 6 |
| 256 | 9 | 14 | 10 | 8 |

Some growth is expected: the paper reports `N_it = O(log ω)`, which over these
four frequencies would read roughly 4, 5, 6, 7.  The 3λ column, 4, 5, 6, 8, about
recovers that; the 1λ column outruns it.  The natural explanation — that the
reflection each absorber leaves accumulates over the `q ∝ ω` interfaces a wave
crosses — is wrong, and was tested (`results/lsweep_boxsize.txt`): at ω/2π = 256
and a 1λ pad, boxes of 64, 96 and 128 points (`q` = 32, 22, 16) need 14, 14 and
15 iterations, and at 2λ the 64- and 128-point boxes both need 10.  Halving the
interfaces changes nothing; and 128-point boxes at 256 against 64-point boxes at
128 — the same `q` = 16, the same pad — need 15 against 7.  What sets the count is
the pad in wavelengths and the frequency itself; why propagation distance should
matter at fixed per-interface absorption is open.  Widening the pad to 3λ at
ω/2π = 256 restores the count to 8, below the layered sweep, at the price of a
local problem of 13 572 unknowns instead of 7 140 — setup and application both
roughly double.  For one right-hand side the narrow pad is still the better trade;
for many right-hand sides, where setup amortises and the solve phase alone
decides, the layered sweep overtakes every L-sweeps variant *with UMFPACK
factors* at 256 — the compact factors below reverse that.
The two-level variant reaches the exact-`H` floor of 3 outer iterations; it pays
when the dense operator is expensive relative to the sweep, which is not yet the
case on one node (for the structured operators).

**Half the bytes.**  If the application is bound by the bytes it streams, a
lever for improvement is the factors.  `factors = :compact` (now the default)
re-stores each box's UMFPACK factorisation once at setup as `L` and `U` in
`ComplexF32` with `Int32` indices — 12 bytes per nonzero against UMFPACK's 24 —
with the row scaling and permutations that reproduce its solve, `(Rs .* A)[p,
q] == L * U`.  The triangular solves run in Julia, accumulate in double
precision, allocate nothing, and skip the zeros a trace right-hand side leaves
in the running vector.  Per-box solves agree with UMFPACK's to 1e-6, the
preconditioner is unchanged to the digits shown, and results still agree
bitwise across worker counts (`results/compact_factors.txt`).  At ω/2π = 256:

| 64-point boxes unless stated | T_setup | T_apply | iters | T_solve | peak |
|:--|---:|---:|:--:|---:|---:|
| pad 8, UMFPACK factors            | 10.27 s | 1.563 s | 14 | 28.61 s | 20.4 GB |
| pad 8, compact                    | 12.13 s | 1.089 s | 14 | 19.44 s | 14.5 GB |
| pad 16, compact                   | 16.82 s | 1.472 s | 10 | 18.36 s | 18.2 GB |
| pad 16, 128-point boxes, compact  | 17.93 s | 1.069 s | 10 | 14.85 s | 15.8 GB |
| layered sweep, for reference      | 29.36 s | 2.059 s |  9 | 23.99 s | 15.1 GB |

The application is 1.44x faster and the peak 29% lower, for two seconds of
extraction at setup; the gain is short of the 2x in bytes because the solves are
latency-bound per core as well as bandwidth-bound in aggregate.  The solve phase
is again ahead of the layered sweep's, 19.4 s against 24.0 s, and time to solution
is 31.6 s against 53.4 s.  Larger boxes bought no iterations but do buy bytes —
the padded box's redundancy falls from 1.72x at 64 points to 1.34x at 128 — which
is why 128-point boxes with a 2λ pad give the fastest solve phase measured,
14.9 s. Additionally, BLAS is pinned
to one thread while the boxes factorise (its threads only spun — 92 CPU-seconds
of 150 — and the pin must be unconditional because the BLAS thread count changes
the factors in their last bits, which would break bitwise agreement between
one-worker and sixteen-worker runs), and the local matrix is assembled straight
into CSC instead of through a coordinate list and `sparse()`.

**Memory.**  A box padded by one wavelength carries 7 140 unknowns against the
4 160 it owns, a 1.7x redundancy.  With UMFPACK's factors the peak is about 1.35x
the layered sweep's at every size; with the compact ones it is 14.5 GB against
15.1 GB at ω/2π = 256, below it.  At ω/2π = 512 (N = 4095² = 1.7e7) neither method
fits the 61 GB of this machine: both were stopped by a watchdog at 40 GB while
still in setup, and scaling the 256 peaks by the fourfold growth in N puts them
near 65 GB (layered) and 58 GB (L-sweeps, compact) — closer, but not this
machine.  Distributing the boxes is what lifts that ceiling.

**Threads, ranks, and both.**  Setup and one application of `lsweep_solve` at
ω/2π = 256 (1024 boxes), best of three, ranks bound to disjoint cores, timed as
the slowest rank (`examples/lsweep_mpi.jl` via `examples/run_mpi_bound.jl`,
output in `results/lsweep_mpi_scaling.txt`):

| configuration | T_setup | T_apply | peak per rank |
|:--|---:|---:|---:|
| 1 process x 16 threads  | 6.80 s | 1.43 s | 15.1 GB |
| 4 ranks x 4 threads     | 4.54 s | 1.62 s |  4.7 GB |
| 8 ranks x 2 threads     | 4.77 s | 1.98 s |  2.9 GB |
| 16 ranks x 1 thread     | 4.48 s | 1.69 s |  2.1 GB |

The application costs the same in every mixture; the messages are small (two grid
layers per interface) and hide behind the local solves.  Setup is flat across the
distributed configurations and half again slower with threads alone, the sparse
LU's poor threading showing through again.  Memory per rank falls sevenfold from
one process to sixteen, since only the grid-sized vectors and the kernel are
replicated — that is what puts problems larger than one node can hold within
reach of a cluster.

Two configuration traps, each costing a factor of three.  Without core binding
the 4 x 4 run applies in 1.18 s, because the launcher pins each rank to one core
and its four threads share it — bind the ranks when mixing.  And with OpenBLAS at
its default thread count the 4 x 4 setup takes 13.5 s instead of 4.5 s, every
rank starting a full pool inside its four cores; `lsweep_setup` now pins BLAS to
one thread itself for the factorisation, and `OPENBLAS_NUM_THREADS=1` is still
the right setting for the rest of a run when more than one rank shares a node.

**A constant-contrast disc, validated against the Mie series.**  Radius 1/4 at
(1/2, 1/2) — the unit disc after rescaling lengths by 4, size parameter `ωR = ω/4`
— velocity 0.8 inside (index 1.25), 1 outside, the default plane wave; otherwise
the setup of the lens table.  Below ω/2π = 16 the 64-point box is the whole grid,
so the L-sweeps rows there are the exact solve of `H` in single precision
(`examples/disc_compare.jl`, `results/disc_compare.txt`):

| ω/2π | inner solver | T_setup | T_apply | outer | sweeps | fwd | Hmv | t_sweep | t_fwd | t_Hmv | T_solve | peak |
|---:|:--|---:|---:|:--:|:--:|:--:|:--:|---:|---:|---:|---:|---:|
| 2 | layered sweep | 0.33 s | 0.000 s | 4 | 5 | 4 | 0 | 0.0000 s | 0.0000 s | 0.0000 s | 0.02 s | 0.5 GB |
|  | one box (= exact H) | 0.09 s | 0.000 s | 2 | 3 | 2 | 0 | 0.0000 s | 0.0000 s | 0.0000 s | 0.02 s | 0.6 GB |
|  | one box, inner 6 | 0.10 s | 0.001 s | 2 | 4 | 3 | 2 | 0.0005 s | 0.0000 s | 0.0000 s | 0.00 s | 0.6 GB |
| 4 | layered sweep | 0.49 s | 0.001 s | 6 | 7 | 6 | 0 | 0.0010 s | 0.0000 s | 0.0000 s | 0.02 s | 0.5 GB |
|  | one box (= exact H) | 0.11 s | 0.000 s | 3 | 4 | 3 | 0 | 0.0000 s | 0.0000 s | 0.0000 s | 0.02 s | 0.6 GB |
|  | one box, inner 6 | 0.10 s | 0.001 s | 3 | 6 | 4 | 3 | 0.0005 s | 0.0000 s | 0.0000 s | 0.00 s | 0.6 GB |
| 8 | layered sweep | 0.63 s | 0.003 s | 9 | 10 | 9 | 0 | 0.0030 s | 0.0001 s | 0.0000 s | 0.04 s | 0.6 GB |
|  | one box (= exact H) | 0.13 s | 0.001 s | 3 | 4 | 3 | 0 | 0.0010 s | 0.0001 s | 0.0000 s | 0.02 s | 0.6 GB |
|  | one box, inner 6 | 0.14 s | 0.003 s | 3 | 6 | 4 | 3 | 0.0015 s | 0.0001 s | 0.0000 s | 0.02 s | 0.6 GB |
| 16 | layered sweep | 0.75 s | 0.009 s | 12 | 13 | 12 | 0 | 0.0090 s | 0.0005 s | 0.0001 s | 0.14 s | 0.6 GB |
|  | L-sweeps pad 8 | 0.22 s | 0.003 s | 12 | 13 | 12 | 0 | 0.0030 s | 0.0005 s | 0.0001 s | 0.09 s | 0.7 GB |
|  | L-sweeps pad 8, inner 6 | 0.22 s | 0.058 s | 3 | 24 | 4 | 21 | 0.0073 s | 0.0005 s | 0.0001 s | 0.08 s | 0.7 GB |
| 32 | exact solve of H | 0.47 s | 0.018 s | 3 | 4 | 3 | 0 | 0.018 s | 0.0021 s | 0.0005 s | 0.11 s | 0.7 GB |
|  | layered sweep | 1.05 s | 0.039 s | 16 | 17 | 16 | 0 | 0.039 s | 0.0021 s | 0.0005 s | 0.62 s | 0.8 GB |
|  | L-sweeps pad 8 | 0.47 s | 0.011 s | 19 | 20 | 19 | 0 | 0.011 s | 0.0021 s | 0.0005 s | 0.40 s | 1.2 GB |
|  | L-sweeps pad 8, inner 6 | 0.48 s | 0.127 s | 5 | 40 | 6 | 35 | 0.016 s | 0.0021 s | 0.0005 s | 0.54 s | 1.1 GB |
| 64 | exact solve of H | 1.60 s | 0.069 s | 4 | 5 | 4 | 0 | 0.069 s | 0.0092 s | 0.0029 s | 0.47 s | 1.4 GB |
|  | layered sweep | 1.98 s | 0.136 s | 34 | 36 | 35 | 0 | 0.136 s | 0.0092 s | 0.0029 s | 5.38 s | 1.5 GB |
|  | L-sweeps pad 8 | 1.23 s | 0.066 s | 32 | 34 | 33 | 0 | 0.066 s | 0.0092 s | 0.0029 s | 2.72 s | 1.7 GB |
|  | L-sweeps pad 8, inner 6 | 1.20 s | 0.481 s | 6 | 48 | 7 | 42 | 0.060 s | 0.0092 s | 0.0029 s | 3.13 s | 1.9 GB |
| 128 | exact solve of H | 7.06 s | 0.304 s | 5 | 6 | 5 | 0 | 0.304 s | 0.055 s | 0.012 s | 2.19 s | 4.4 GB |
|  | layered sweep | 6.30 s | 0.523 s | 51 | 54 | 53 | 0 | 0.523 s | 0.055 s | 0.012 s | 31.50 s | 4.5 GB |
|  | L-sweeps pad 8 | 3.62 s | 0.312 s | 53 | 56 | 55 | 0 | 0.312 s | 0.055 s | 0.012 s | 17.97 s | 4.3 GB |
|  | L-sweeps pad 16 | 5.22 s | 0.366 s | 53 | 56 | 55 | 0 | 0.366 s | 0.055 s | 0.012 s | 22.77 s | 5.4 GB |
|  | L-sweeps pad 8, inner 6 | 3.65 s | 1.975 s | 8 | 64 | 9 | 56 | 0.247 s | 0.055 s | 0.012 s | 16.21 s | 4.6 GB |
| 256 | exact solve of H | 31.91 s | 1.264 s | 10 | 11 | 10 | 0 | 1.264 s | 0.308 s | 0.046 s | 18.25 s | 16.6 GB |
|  | layered sweep | 31.40 s | 2.082 s | 156 | 164 | 163 | 0 | 2.082 s | 0.308 s | 0.046 s | 403.49 s | 16.4 GB |
|  | L-sweeps pad 8 | 12.46 s | 1.100 s | 328 | 345 | 344 | 0 | 1.100 s | 0.308 s | 0.046 s | 457.84 s | 14.6 GB |
|  | L-sweeps pad 16 | 16.97 s | 1.552 s | 317 | 333 | 332 | 0 | 1.552 s | 0.308 s | 0.046 s | 566.54 s | 18.4 GB |
|  | L-sweeps pad 24 | 23.72 s | 1.997 s | 297 | 312 | 311 | 0 | 1.997 s | 0.308 s | 0.046 s | 660.11 s | 23.3 GB |
|  | L-sweeps pad 8, inner 6 | 12.40 s | 7.673 s | 22 | 176 | 24 | 154 | 0.959 s | 0.308 s | 0.046 s | 177.83 s | 18.3 GB |

**Columns.**  `ω/2π` — frequency; the grid is `n = 8(ω/2π) − 1` at eight points
per wavelength and the disc is `ω/4π` wavelengths across.  `inner solver` — the
approximate inverse of `H` the preconditioner uses; "inner 6" runs up to six
GMRES iterations on `H` per application, to 1e-3, in place of a single sweep.
`T_setup` — building it: sparsification, `H`, and the factorisation.  `T_apply` —
one preconditioner application, timed once, comprising `S·r`, the inner solve and
the scatter.  `outer` — iterations of the outer Krylov solver on the dense system.
`sweeps` — preconditioner applications performed inside that solve (LU solves on
the exact-`H` row).  `fwd` — forward maps, applications of `A = I + ω²K[m ·]`,
each an FFT convolution.  `Hmv` — applications, *not* solves, of the sparsified
`H`, one per inner GMRES iteration; only the two-level rows perform any.
`t_sweep` — the cost of one sweep, i.e. `T_apply` divided by the sweeps that
application performs (one for the single-level rows, eight for the two-level rows
from ω/2π = 16 up); it carries `S·r` and the scatter with it, 0.050 s of 1.100 at
n = 2047.  `t_fwd`, `t_Hmv` — the cost of one forward map and of one `H` matvec;
both depend on the grid alone, not on the medium or the preconditioner, so one
measurement per size serves every row (`--costs-only`), and `t_Hmv` is already
inside the two-level rows' `t_sweep`.  `T_solve` — the outer Krylov solve, setup
excluded.  `peak` — peak resident set.  Every column is measured.

The counts follow the solvers' schedule.  A left-preconditioned GMRES(20) applies
operator and preconditioner once per iteration and once more at every restart,
plus one preconditioner application on the initial residual, so the single-level
rows have `fwd = outer + r` and `sweeps = outer + 1 + r` with
`r = ⌊(outer−1)/20⌋` restarts, and the two-level rows have `Hmv = sweeps − outer`.
Only the two-level sweep counts cannot be derived: each application runs an inner
GMRES that stops at 1e-3 or its cap of six, which is 2 sweeps per application at
ω/2π ≤ 8 and 8 from 16 upward.  As a check, `sweeps × t_sweep + fwd × t_fwd`
accounts for 392 s of the layered sweep's 403 s at ω/2π = 256, 176 s of the
two-level scheme's 178 s and 17.0 s of the exact solve's 18.3 s; it overshoots
the single-level L-sweeps rows by about 8%, so their single timed application is
slightly pessimistic against the same sweep run back to back inside a solve.  At
n = 2047 a forward map costs 6.6× an `H` matvec — 0.308 s against 0.046 s — and
that ratio is the whole reason the two-level scheme pays.

The iterations tell a different story from the lens.  The exact solve of `H`
needs 3, 4, 5, 10 at ω/2π = 32 … 256 — the sparsification floor is modest — and
everything above it is the sweeps' inexactness on a medium that reflects: the
lens reflects weakly and one pass captures nearly the whole field (9 and 14
iterations at 256), while the disc's interface sends waves back and each pass
captures one more reflection, so the layered sweep needs 156 and the L-sweep 328
(and with GMRES(20), 8 and 16 restarts).  It is not the absorber: a 3λ pad takes
328 to 297 while doubling the application cost, where on the lens it took 14 to 8.
Restarts are a real cost at these counts: 328 iterations of GMRES(20) carry 16
restarts, each an extra forward map and sweep.

The work columns show why the two-level scheme, useless on the lens, is decisive
here.  It does not sweep less — 176 against the layered sweep's 164 — but its
sweeps are L-sweeps at ~0.96 s rather than layered sweeps at 2.08 s, and it cuts
the forward maps from 163 to 24.  Against single-level L-sweeps the gain is
plainer: 176 sweeps against 345, 24 forward maps against 344.  The inner
iterations move the work onto the sparse system `H`, where an iteration costs a
sweep and a sparse matvec, and off the dense system, where one costs a forward
map plus orthogonalisation against a basis of 4.2-million-entry vectors: 190 s to
solution against 435 s and 470 s.  The exact LU of `H` is the limit of that trade
and beats all of them — ten forward maps, eleven LU solves, 31.9 s of setup (the
same as the layered sweep's 31.4 s), 50 s in total at the same 16.6 GB.  In 2D at
these sizes the factorisation the sweeps exist to avoid is not expensive, and on
a medium that reflects, avoiding it costs an order of magnitude.

**Three dimensions.**  L-sweeps as beams — a `q x q` decomposition of the
`(x₁, x₂)` plane with each box spanning `x₃`, the construction of the paper's
own 3D experiments, though at a different scale: they run q = 12–32 beams per
side on p = q nodes with 2λ boxes and a 1λ PML in constant media; here 3×3 and
4×4 beams on one node, 2.25λ boxes, a 0.5λ pad, in the lens.  Converging lens,
`h = λ/8`, `b = 4`, `C = 12`, as in the 3D tables; L-sweeps on 16 threads, the
layered sweep untouched — its
application on one core, its 3D factorisation using OpenBLAS's default 16 threads
inside UMFPACK's dense fronts, which are large enough here to matter (7.5 s with
one BLAS thread against 5.0 s with sixteen at n = 31).  `inner × outer` is the nominal
count `outer × max(inner, 1)`; `sweeps` is the number of sweep applications
actually performed inside the outer solve, and differs from it in both
directions: a left-preconditioned GMRES applies its preconditioner to the
initial residual too, so the `inner = 0` rows sweep `outer + 1` times, while with
inner iterations each of the `outer` applications is an inner GMRES that stops at
1e-3 after two iterations — three sweeps — so those rows sweep about `3 × outer`
times, half of `inner × outer` (`examples/lsweep_compare3d.jl`,
`results/lsweep_compare3d.txt`):

| ω/2π | inner solver | T_setup | T_apply | outer | inner × outer | sweeps | T_solve | peak |
|---:|:--|---:|---:|:--:|:--:|:--:|---:|---:|
| 4 | layered sweep                 | 4.93 s   | 0.223 s | 4 | 4  | 5 | 1.20 s  | 2.5 GB  |
|   | layered sweep, inner 6        | 4.97 s   | 0.716 s | 3 | 18 | 9 | 2.21 s  | 2.4 GB  |
|   | L-sweeps 3x3, pad 4           | 10.54 s  | 0.161 s | 3 | 3  | 4 | 0.74 s  | 8.2 GB  |
|   | L-sweeps 3x3, pad 4, inner 6  | 12.11 s  | 0.345 s | 3 | 18 | 8 | 1.33 s  | 8.1 GB  |
| 8 | layered sweep                 | 28.26 s  | 1.748 s | 4 | 4  | 5 | 8.89 s  | 13.3 GB |
|   | layered sweep, inner 6        | 28.87 s  | 5.218 s | 3 | 18 | 9 | 15.97 s | 13.4 GB |
|   | L-sweeps 4x4, pad 4, nfact 8  | 169.90 s | 1.628 s | 4 | 4  | 5 | 8.36 s  | 39.9 GB |
|   | L-sweeps 4x4, pad 4, inner 6  | 162.59 s | 4.926 s | 3 | 18 | 9 | 15.01 s | 40.6 GB |

Inner iterations do not pay here: the outer count sits at or next to the exact-`H`
floor of 3 without them, so they only multiply the application cost.  At n = 31
L-sweeps is the better inner solver — 3 iterations against 4, the application
1.4x and the solve phase 1.6x faster — at twice the setup and three times the
memory.  At n = 63 the layered sweep wins: the same 4 iterations, an application
only 7% faster, the solve phase within 6%, and a setup six times slower at three
times the memory.  The reasons are geometric.  A padded beam carries
`(s + 2(2 + pad))² × ntot` unknowns for `s² × ntot` owned — 3.1x redundancy at
`s = 16`, `pad = 4`, against 1.7x for the 2D default — 3D LU cost grows like N²
rather than N^1.5, and pad 4 is already half a wavelength, so there is no thinner
absorber to reach for.  With sixteen beams factorising at once the process passed
45 GB and was stopped; `nfact = 8` holds it to 39.9 GB in two rounds.

**Accuracy of one application** against `H⁻¹f` for a random `f` supported on
all of `Ω` (n = 127, ω/2π = 16): in a constant medium `2e-4`–`6e-4` for 2 x 2 to
4 x 4 boxes, halving from `pad = 8` to `16`, which is the accuracy of the fitted
stencils themselves; in the lens `2e-3`–`3e-2` depending on where the interfaces
cut it, and almost independent of the pad.  As in §4 of the paper, the local
problems do not see reflections and refractions across the interfaces; the
outer iteration supplies them, in the 3–7 iterations above.

**Accuracy of one transmission** (`examples/lsweep_transmission.jl`, output in
`results/lsweep_transmission.txt`): a 2 x 2 CDD of 64-point boxes with the corner
at the centre of the lens and the source in `Ω11`, each box compared with
`lu(H) \ f`.  `Ω21` and `Ω12` agree to all printed digits — the configuration is
symmetric about the diagonal:

| ω/2π | pad | source box | straight steps | diagonal step |
|---:|:--|---:|---:|---:|
| 32  | 1λ | 4.0e-4  | 3.8e-3  | 4.5e-4  |
| 32  | 2λ | 2.6e-4  | 1.5e-3  | 1.7e-4  |
| 64  | 1λ | 1.1e-3  | 1.8e-2  | 2.0e-3  |
| 64  | 2λ | 4.7e-4  | 8.5e-3  | 4.6e-4  |
| 64  | 3λ | 4.2e-4  | 4.1e-3  | 5.2e-4  |
| 128 | 1λ | 1.03e-3 | 1.10e-2 | 1.46e-3 |
| 128 | 2λ | 7.75e-4 | 5.17e-3 | 1.06e-3 |
| 128 | 3λ | 7.82e-4 | 3.51e-3 | 1.10e-3 |

The straight steps halve with each doubling of the absorber and then flatten near
the accuracy of the fitted stencils; the diagonal step through the L-shaped trace
is as accurate as the source box.

### Scope and next steps

The 3D variant is in — beams along the third axis, which is the paper's own 3D
construction (§1 and its Table 9: `q x q` beams, one row per node, q = 12–32 on
constant media at 6 points per wavelength, 2λ boxes, 1λ PML) — and at n = 63 it
loses to the layered sweep on setup and memory, for the geometric reasons
above.  The paper's cost estimate agrees: 3D factorisation scales like `n³
n_pml⁶ / p`, "therefore it is crucial to keep the thickness of the PML region
as thin as possible", and their own 3D applications take 52 s at N = 156³ on 12
nodes. The row partition caps the useful rank count at the number of box rows;
the paper's own runs use 64–128 ranks at 6 000–13 000 points per side. The
absorber wants a rule: held fixed in wavelengths it lets the iteration count
grow about linearly with the frequency at one wavelength and roughly
logarithmically at three.  The box-size study rules out the interface count as
the driver, which leaves the size of the domain in wavelengths, and no
mechanism for that yet.  The distributed test runs 3 ranks x 2 threads under
`mpiexec` and checks bitwise agreement with the single-process result.

---

## Bi-directional sweeps (Zepeda-Núñez & Zhao)

The layered sweep is one factorisation along one axis.  Zepeda-Núñez & Zhao,

> L. Zepeda-Núñez, H. Zhao, **Fast alternating bi-directional preconditioner for
> the 2D high-frequency Lippmann–Schwinger equation**, SIAM J. Sci. Comput.
> **38** (2016) B866–B888,

precondition the same sparsified system with two layered decompositions, one in
horizontal and one in vertical layers, applied in turn (their Alg. 2):

```
u₁ = Sweep_horizontal(f)
e  = f − H u₁                      # one step of iterative refinement
u  = u₁ + Sweep_vertical(e)
```

and wrap that in a GMRES on `H` to `1e-3` inside every application of the
Lippmann–Schwinger preconditioner (their Alg. 3), with the outer GMRES flexible.
Their sweeps are polarized-trace sweeps; here the same construction is built
from the moving-PML sweep of Liu & Ying.

### What is implemented

* `sweep_setup(...; axis = 2)` cuts the slices along `x₂` instead of `x₁`; the
  code never singled out an axis, only the default did.  `axis = -1` runs the
  one-front sweep towards `-x₁` (two fronts are symmetric and ignore the sign).
  On the transposed medium the `x₁` sweep reproduces the `x₂` sweep to `1e-8`,
  once the right-hand side vanishes on the PML rows: the fitted PML stencils are
  null vectors, defined up to a unimodular factor, so the PML rows of `H(m)` and
  `H(mᵀ)` carry different phases, which is harmless because those rows have no
  data (`test/test_sweep.jl`, "the sweep axis is a free choice").
* `AlternatingSweep` holds a list of factorisations and applies each to the
  residual the previous ones left; `alternating_setup(g, ω, m, H; axes = (1, 2))`
  builds it.  With two orthogonal axes it is Alg. 2 above, a fixed linear
  operator `M₁ + M₂ − M₂ H M₁`, so plain GMRES can use it; each application
  costs one sweep per factorisation and one sparse matvec per refinement, and
  the setup builds and stores one factorisation per axis.
* `SweepPreconditioner(P; axes = (1, 2))` selects it; `axes = (2,)` is the
  single `x₂` sweep, `axes = (1, -1)` with `fronts = 1` alternates the direction
  on one axis, and `axes = (1, 2)` with `inner = k, inner_tol = 1e-3` is Alg. 3.
  The sweep counter `nsweeps` counts sweeps, so an alternating application adds
  one per axis.  Inti's extension passes the keyword straight through.
* `SweepPreconditioner(P; mode = :lsweep, shifts = ((0, 0), (0.5, 0.5)))` is the
  same alternation for L-sweeps: a second L-sweep on a checkerboard shifted by
  half a box on each axis, applied to the residual of the first
  (`lsweep_setup(...; shift = (0.5, 0.5))` builds the shifted board, which
  carries one box more per axis).  `AlternatingSweep` chains any factorisations
  that have an `approx_solve` method, layered or L-shaped. (Untested: 3D
  alternating L-sweeps.)

### Four discs in a square lattice

`examples/lattice_compare.jl`; raw output in `results/lattice_compare.txt`.
Four discs of radius `0.15` centred at `(½ ± 0.2, ½ ± 0.2)`, `m = −1` inside
(`n² = 2`, `c_in = 1/√2`) with a sharp edge, plane wave `e^{iωx₂}`, `h = λ/8` in
the background (`λ/5.7` inside the discs), `b = 8`, `C = 12`, GMRES(60) to
`1e-6`.  Entries are outer iterations, with the sweeps performed inside the
solve in parentheses; every row solved the same system, and the solutions agree
with the exact-`H` row to the tolerance.

| ω/2π | slices | exact H | x₁ sweep | x₂ sweep | x₁ then −x₁, one front | x₁ then x₂ (Alg. 2) | x₁, x₂, x₁ | x₁, inner 6 | x₁ then x₂, inner 6 | x₁ then x₂, inner 12 |
|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 8   | 8   | 3  | 13 (14)   | 11 (12)   | 8 (18)   | 8 (18)   | 6 (21)   | 3 (24)   | 3 (34)   | 3 (34)   |
| 16  | 16  | 4  | 22 (23)   | 16 (17)   | 15 (32)  | 13 (28)  | 10 (33)  | 4 (32)   | 3 (44)   | 3 (44)   |
| 32  | 32  | 4  | 27 (28)   | 21 (22)   | 19 (40)  | 19 (40)  | 14 (45)  | 5 (40)   | 3 (48)   | 3 (60)   |
| 64  | 64  | 5  | 38 (39)   | 31 (32)   | 22 (46)  | 21 (44)  | 20 (63)  | 7 (56)   | 4 (64)   | 3 (68)   |
| 128 | 128 | 14 | 160 (163) | 112 (114) | 99 (202) | 94 (192) | 86 (264) | 24 (192) | 14 (224) | 19 (532) |

Solve times, setup excluded, one process with 8 threads (the sweep itself is
serial):

| ω/2π | exact H | x₁ sweep | x₂ sweep | x₁ then x₂ | x₁, inner 6 | x₁ then x₂, inner 6 | x₁ then x₂, inner 12 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 2.0 s  | 19.7 s  | 16.2 s  | 17.9 s  | 20.8 s  | 26.0 s  | 25.1 s  |
| 128 | 15.2 s | 305.2 s | 203.5 s | 276.2 s | 243.7 s | 264.6 s | 339.0 s |

**Payoff of `alternating' sweeps.**  Outer iterations fall by a third to a half
at every size — 38 → 21 at ω/2π = 64, 160 → 94 at 128. Each application is two
sweeps and a matvec, though, so the sweep count goes the other way, 39 → 44 and
163 → 192, and the time to solution is within 10 % either way (for this forward
map).  The single-sweep error on a disc is about `0.28` per application and the
alternated one about `0.12` (`test_sweep.jl`), which is worth about 1.7 single
sweeps; at two sweeps a pass it is break-even in work, and the gain in
iterations is the gain in Krylov overhead.  Three passes (`x₁, x₂, x₁`) save a
few more iterations and cost more sweeps than they save.

**What it does not change.**  The growth.  The single sweep goes 13, 22, 27, 38,
160 over five doublings and the alternated one 8, 13, 19, 21, 94: both jump by
4× at the last step, as the single disc did in the table further up, and the
exact solve of `H` itself goes from 5 to 14 there.  Zepeda-Núñez & Zhao's
second pass cures a defect of their first — spurious grazing waves from the
smooth cut-off between layers, and a residual concentrated on the interfaces
that a single polarized-trace sweep cannot restrict — and on smooth media the
moving-PML sweep does not have that defect: it is at 4–9 iterations already.
What limits it on discs is multiple reflection between the interfaces, and a
wave that turns back along `x₁` *and* along `x₂` is missed by both passes.

**Incidence.**  The single sweeps depend on it and the alternation does not.
With the wave along `x₂` the `x₂` sweep, whose slices the wave crosses, needs
31 iterations at ω/2π = 64 against 38 for the `x₁` sweep, whose slices it
grazes; with the wave along `x₁` the numbers swap (31 and 38); at 45° both need
38.  The alternated sweep needs 20–21 in all three cases.

**As the inner solver (Alg. 3).**  With GMRES on `H` to `1e-3` inside every
application, capped at six iterations, the alternated sweep reaches the
exact-`H` count at every size (3, 3, 3, 4, 14 outer iterations) where the
single sweep does not (3, 4, 5, 7, 24), at 15–40 % more sweeps and 10–35 % more
time.  Raising the cap to twelve brings the single sweep to 3 outer iterations
at ω/2π ≤ 32 and 4 at 64, so the cap rather than the sweep was binding there.
At 128 the cap of twelve gives 23 outer iterations and 322 sweeps for the
single sweep (335 s) against 19 and 532 for the alternated one (339 s).

**Contrast and resolution.**  At `m = −3` (`n² = 4`, twice the index) and
`h = λ/16` in the background — `λ/8` inside the discs, the same interior
resolution as the table above — the picture is the same with larger numbers.
ω/2π = 32 (64 slices, discs 9.6λ across): exact `H` 6; `x₁` sweep 91; `x₂`
sweep 54; `x₁` then `−x₁` 57 (116 sweeps); `x₁` then `x₂` 50 (102 sweeps); with
six inner iterations 25 outer and 200 sweeps for the single sweep against 10
outer and 160 sweeps for the alternated one, 69 s against 48 s — the one setting
in these runs where the alternation wins on work as well as on iterations, because
the single sweep no longer reaches `1e-3` within the cap and the alternated one
still does.  At ω/2π = 16 the corresponding counts are 4, 39, 30, 30 (62), 25 (52),
10 (80), 5 (80).  The `m = −1` control at `h = λ/16`, ω/2π = 32, gives 4, 25, 20,
14 (30), 5 (40), 3 (46), within a few iterations of the `λ/8` row.  Under-resolved
discs are another matter: `m = −3` at `h = λ/8` has four points per interior
wavelength, and there the single sweep needs 297 iterations at ω/2π = 64, the
alternated sweep does not converge in 300, and its two-level version stalls — the
sparsified surrogate has stopped being one, and composing two bad sweeps
compounds the damage.  Those rows are kept in `results/lattice_compare.txt` as a
warning; the interior resolution, `ppw/√(1−m)`, is the number to watch.

### What the residual says

The residual `f − H M f` of one layered sweep lies entirely on one grid line per
slice, the line facing the slice's eliminated neighbour, next to its moving PML
(100.0000 % of its norm on 11 % of the unknowns; `results/lattice_compare.txt`).
That decides what a second pass can do with it.  The same sweep applied again
puts its absorbers right on top of the residual and, from ω/2π = 32 up, makes it
larger (reduction 1.06 at 32, 1.49 at 64, against 0.62–0.77 for the flipped or
orthogonal sweeps).  Hence the controls, outer iterations with sweeps in
parentheses, one front:

| ω/2π | x₁ | x₁ then x₁ again | x₁ then −x₁ | x₁, −x₁, x₂, −x₂ | x₁, x₂, −x₁, −x₂ |
|---:|:--:|:--:|:--:|:--:|:--:|
| 32  | 27 (28)   | 21 (44)        | 19 (40)  | 12 (52)  | 15 (64)   |
| 64  | 38 (39)   | 39 (80)        | 22 (46)  | 18 (76)  | 24 (100)  |
| 128 | 160 (163) | > 300 (612)    | 99 (202) | 59 (240) | 120 (488) |

Repeating a sweep buys nothing; flipping it halves the count; four passes with
each axis flipped before the next give the lowest outer count of any layered
combination and the fewest forward maps, at 47 % more sweeps than one sweep.
Interleaving the axes instead doubles the cost.  Along any chain, only the first
two passes reduce the residual much; the third and fourth take 0.75–1.07 off it
each, and what survives sits increasingly inside the discs (61 % of the error
after four passes, on 28 % of the points).

### Shifted checkerboard: the alternation for L-sweeps

An L-sweep's residual lies on the grid layer either side of every box interface
(again 100.0000 % of its norm, on 5 % of the unknowns), and a second L-sweep on
the same board reduces it by only 0.70–0.85.  A board shifted by half a box on
each axis holds those layers in the interiors of its boxes and reduces the same
residual by 0.58 at ω/2π = 32 and 0.75 at 64 — better than the same board, not as
good as a layered sweep (0.30, 0.48), which keeps every reflection along its
slab axis where the L-sweep keeps none.  On the lattice, `nbox = 64`, `pad = 8`,
GMRES(60) to `1e-6`, outer iterations with L-sweeps in parentheses:

| ω/2π | boards | exact H | L-sweeps | L then shifted L | L-sweeps, inner 6 | L then shifted L, inner 6 |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 16  | 2² / 3²   | 4 | 13 (14) | 6 (14)  | 3 (20)  | 3 (24)  |
| 32  | 4² / 5²   | 4 | 33 (34) | 15 (32) | 6 (48)  | 3 (46)  |
| 64  | 8² / 9²   | 5 | 57 (58) | 31 (64) | 10 (80) | 5 (80)  |
| 128 | 16² / 17² | 14 | > 300 (306) | 147 (300) | 40 (320) | 16 (256) |

The alternation halves the outer count at every size, at equal or fewer
L-sweeps up to ω/2π = 32 and 10 % more at 64; at 128 the single L-sweep does
not converge in 300 iterations and the alternation does, in 147, at 300
L-sweeps.  It is the one alternation in this study that pays in work as well as
in iterations, and inside the two-level scheme it does so at every size: the
exact-`H` count at ω/2π ≤ 64 where the unshifted version stays at twice it, and
16 outer iterations at 256 L-sweeps against 40 at 320 at ω/2π = 128.  The two boards are as accurate as each other
on the lattice (0.084 and 0.080 at ω/2π = 64); on a lens whose centre lies on
the first board's skeleton the shifted board is fifteen times more accurate,
because its cuts avoid the strongest part of the medium (`test/test_lsweep.jl`).
The shifted board carries one box more per axis and the pads of its interior
boxes must stay out of the global PML, so a half-box must exceed `pad + 2`
points: with 64-point boxes that allows `pad ≤ 29`.

---

## Unstructured discretisations (in `Inti.jl`)

This package is deliberately limited to structured Cartesian grids.  Coupling
to *unstructured* Lippmann–Schwinger discretisations is done with the
preconditioner,

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

julia --project=. -t 16 examples/lsweep_compare.jl 32 64 128  # layered sweep vs L-sweeps
julia --project=. -t 8 examples/lattice_compare.jl 8 16 32 64 128  # four discs: single vs bi-directional sweeps
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=test test/runtests.jl                      # ... including the MPI test
```

The ω/(2π) = 256 runs need about 14 GB of RAM.

## Dependencies / Extensions of Base

`FFTW`, `IterativeSolvers`, `SpecialFunctions`, and the standard libraries
`LinearAlgebra`, `SparseArrays`, `Random`, `Printf`.  `MPI` is optional: loading
it activates the extension `SparsifyAndSweepMPIExt` with the distributed
backend `MPIComm` of the L-sweeps.

`LSProblem` implements `size`/`eltype`/`mul!` and `SweepPreconditioner`
implements `ldiv!`.
