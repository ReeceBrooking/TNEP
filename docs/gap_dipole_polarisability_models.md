# GAP models for the dipole and polarisability of liquid water: what worked and why

*Branch `ExperimentalPolarisability`, 18–20 October 2026. All numbers are convex
sparse-kernel fits (gap_dipole.py and the experimental modules gap_bond.py,
gap_ewald.py), evaluated on the 500-frame test set of the 32-molecule bulk-water
data (96 atoms, cubic cell $L \approx 9.94$ Å, 500 training frames).*

## 1. Scope, metric and protocol

Every result below is a sparse Gaussian-process (GAP) fit in the gap_fit
estimator: for design matrix $\Phi$, targets $y$, per-frame observation error
$\sigma_\mathrm{obs} = \sigma\sqrt{N}$ and sparse-sparse prior $K_{MM}$,

$$
\hat\alpha = \arg\min_\alpha \; \sum_f \frac{\lVert \Phi_f\alpha - y_f\rVert^2}{\sigma_\mathrm{obs}^2} + \alpha^\top K_{MM}\,\alpha ,
$$

solved in whitened coordinates $\beta = (L^\top\!\otimes I)\alpha$ with $K_{MM}=LL^\top$,
in the dual ($[DF\times DF]$ Gram, $D=3$ or $6$ rows per frame). Fits take one to
two minutes, which is what made the whole study possible: every readout question
was settled on the *representability floor* of the basis before any
SNES/Adam training was considered.

Accuracy is quoted as the relative RMSE of the TNEP paper (Xu et al., JCTC 2024),

$$
\mathrm{RRMSE} = \frac{\mathrm{RMSE}}{\mathrm{std}(\text{reference})},\qquad R^2 = 1-\mathrm{RRMSE}^2 ,
$$

computed over components (the three Cartesian dipole components; the three
diagonal and the three off-diagonal Voigt components of $\alpha$ separately).
Note that the code's own `rrmse` history divides by the raw target magnitude and
is dominated by the 44 Å$^3$ isotropic trace of $\alpha$; it is not comparable
to published values.

Protocol: hyperparameters ($\sigma$, kernel-group amplitudes) were selected on a
400/100 split of the training set and the winner refitted on all 500 frames
before scoring the test set. Where a table says "fit 400", the model was fitted
on 400 frames and scored on test directly. Published reference values: dipole
0.85 % (same label spread as ours, $0.20\ e\cdot a_0$ per molecule, so the
comparison is fair); polarisability 16.3 % (diagonal) / 20.4 % (off-diagonal),
on the susceptibility $\chi/\rho$ rather than the cell polarisability, so only
the relative numbers are comparable.

## 2. The readout family

Both properties are read out from per-atom *site functions* multiplying fixed
*geometric kernels* built from the bond vectors $\mathbf d_{ij} = \mathbf r_i-\mathbf r_j$
of all image pairs within $r_c$:

$$
\mu = \sum_i \sum_k s^{(k)}_i\, \mathbf v^{(k)}_i, \qquad
\alpha = \sum_i \sum_k s^{(k)}_i\, \mathbf W^{(k)}_i ,
$$

where $k$ runs over radial channels $n$, neighbour species $t'$ and kernel
groups. In the GAP the site functions are kernel expansions on the SOAP
descriptor $D_i$ of the environment,

$$
s^{(k)}_i = \sum_m \alpha^{(k)}_m\,\big(\hat D_i\cdot\hat D_m\big)^{\xi}, \qquad \xi = 2,
$$

over $M$ farthest-point-sampled sparse environments per species. Because the
model is linear in $\alpha$, every geometric kernel is a *column block* and every
proposal in this report was testable as "add these columns and refit".

The radial channels are the warped Chebyshev family of the production readout,
$R_n(r) = T_n\!\big(2(r/r_c-1)^2-1\big)\,f_c(r)$ with a cosine or squared-cosine
cutoff, $C = 10$ channels, with independent coefficients per (channel, neighbour
species) pair.

## 3. Dipole models

Three models are documented: the original neighbour-basis GAP (the baseline
every change is measured against), the best *local* model, which keeps the
3 Å readout and adds nothing long-ranged, and the best model overall, which
adds the Ewald field of fixed reference charges as one extra readout column.
All three share the same readout, so switching between them is a change of
configuration, not of code.

### 3.1 The original model (baseline)

$$
\mu=\sum_i\sum_{n,t'} s^{(n,t')}_i\sum_{j\in t'}R_n(r_{ij})\,\mathbf d_{ij},
\qquad
s^{(n,t')}_i=\sum_m\alpha^{(n,t')}_m\big(\hat D_i\cdot\hat D_m\big)^2 .
$$

A bond basis of $C=10$ warped Chebyshev channels per neighbour species on
$r_c=3.0$ Å with a squared-cosine cutoff, each channel with its own site
function: a sparse kernel on the compressed $l_{\max}=\alpha_{\max}=4$ SOAP
(hard/soft cutoff 3.5/3.0 Å), $M=250$ farthest-point environments per species,
$\xi=2$, $\sigma=10^{-3}$. 10 000 columns. **Test RRMSE 1.76 %** on 500
training frames (1.86 % on 400); 0.28 ms/atom.

### 3.2 Best local model (no field)

Two changes to §3.1, both inside the site function; the readout is untouched.

1. **A richer descriptor.** $l_{\max}=\alpha_{\max}=7$ with no compression (the
   full 840-dimensional power spectrum instead of the 4-4 compressed one).
   1.76 → **1.56 %** (fit 500; 1.67 on 400). The selected $\sigma$ stays pinned
   at the smallest value scanned, the signature of a capacity-limited site
   function, which is why more descriptor helped and why $M=500$ did not
   (saturated at 250).
2. **A summed two-body site model** (§3.7). The site function becomes
   $s_i=\sum_m\alpha_m(\hat D_i\!\cdot\!\hat D_m)^2+\delta_{2b}\sum_{m'}\beta_{m'}(\hat{\mathbf q}^{(2)}_i\!\cdot\!\hat{\mathbf q}^{(2)}_{m'})^2$
   with $q^{(2)}_{i,nt'}=\sum_{j\in t'}R_n(r_{ij})$ the 20-number radial
   distribution of the environment on the same 3 Å, its own FPS set
   ($M=250$) and its own prior, $\delta_{2b}=0.1$. 1.56 → **1.49 %** (fit 500;
   1.56 on 400) at $\sigma=3\times10^{-3}$. Columns double to 20 000; the
   forward pass is unchanged within noise.

The result is a fully local model — 3 Å neighbourhoods, no charges, no
periodic sums — at **1.49 %**, 0.45 ms/atom. Everything else that was tried
on the local model (longer readout, longer 2b range, more channels, a second
SOAP scale, bilinear and bond kernels, the TurboGAP 2b/3b blocks, φ kernels,
QEq columns) is in §3.7, §5 and §7 and did not beat it.

### 3.3 Best model: the Ewald field column

$$
\mu=\sum_i\sum_{n,t'} s^{(n,t')}_i\sum_{j\in t'}R_n(r_{ij})\,\mathbf d_{ij}
\;+\;\sum_i s^{(E)}_i\,\lambda_E\,\mathbf E_i ,
$$

with the site functions of §3.2 (SOAP-7 kernel plus the $\delta_{2b}$ 2b
kernel, now with $\delta_{2b}=0.05$–$0.1$ and $\sigma=1.5$–$2\times10^{-3}$)
and one new column: $\mathbf E_i$, the Ewald field at atom $i$ of fixed SPC/E
reference charges ($q_\mathrm O=-0.8476$, $q_\mathrm H=+0.4238$) from all
other atoms and all periodic images, scaled once by $\lambda_E$ so that the
column has the RMS of the bond columns. The field column gets its own site
function $s^{(E)}_i$ like every bond channel, so it is one more column per
sparse environment: 11 000 columns without the 2b model, 22 000 with it.
**Test RRMSE 0.96–0.97 %** (fit 500; 1.05 on 400); 1.08 % without the 2b
model. 1.47 ms/atom, of which the Python Ewald sum is ≈ 1.0.

### 3.4 Results (test set, 500 training frames unless stated)

| Model | fit 400 | fit 500 | columns | ms/atom |
|--------------------------------------------|-----------|-----------|---------------|---------------|
| §3.1 original: $l=\alpha=4$ compressed SOAP, bond basis | 1.86 % | 1.76 % | 10 000 | 0.28 |
| $l=\alpha=7$ uncompressed SOAP | 1.67 % | 1.56 % | 10 000 | 0.43 |
| **§3.2 best local: + 2b site model, $\delta_{2b}=0.1$** | 1.56 % | **1.49 %** | 20 000 | 0.45 |
| $l=\alpha=7$ + Ewald field column | 1.17 % | 1.08 % | 11 000 | 1.45 |
| **§3.3 best: + field + 2b site model, $\delta_{2b}=0.05$–$0.1$** | 1.05 % | **0.96–0.97 %** | 22 000 | 1.47 |
| Published TNEP (NEP descriptors, 6 Å radial / 4 Å angular, 200k SNES generations) | | 0.85 % | | |

Per-component RMSE of the best model: $0.060\ e\cdot a_0$ per cell,
$1.9\times10^{-3}\ e\cdot a_0$ per molecule (TNEP: $1.7\times10^{-3}$);
$R^2=0.9999$.

### 3.5 Field versus no field

The two best models differ by exactly one column and by 0.52 points of
RRMSE, a third of the local model's error. What that column is, and is not:

- **It is the only long-range ingredient that transferred.** Every other way
  of giving the model reach failed on the test set: a second SOAP scale at
  4.5 or 6 Å (1.21–1.55 % on 400, against 1.17 with the field), the 2b
  descriptor on 4–6 Å (monotonically worse), and above all a *longer
  readout*. Widening the bond basis itself to 4.5 or 6 Å — the TNEP's
  mechanism, a learned 6 Å two-body readout — drives the training error from
  0.60 to 0.03 % and the test error from 0.97 to 1.11 and 1.41 %. Without the
  field a 6 Å readout reaches 1.42 % against the 3 Å model's 1.49, and with it
  the field adds nothing (1.41 vs 1.42): the long basis can fit on the
  training set what the field describes, and cannot carry it to new frames.
  With one independent site function per channel and species the basis has
  too much freedom for 500 frames. The field column has one site function.
- **It is the total field, not the near or the far part.** Decomposed, the
  bare Coulomb field of neighbours within 3, 4.5 or 6 Å gives 1.62–1.66 %,
  the Ewald field minus its 3 Å part gives 1.68, the full field 1.17 (all fit
  400, base 1.67). Near and far cancel in a dielectric; the induced dipole
  responds to the small total. A local basis reconstructs only the near part.
- **It is not sensitive to how the charges are placed.** Gaussian-smeared
  sources with $\sigma=0.5$ Å give the same 1.17 %, $\sigma=1.0$ Å 1.20; a
  Thole-screened field is worse (1.64); the SPC/E values are not tuned. A
  learned QEq charge column instead of fixed charges was neutral (§5).
- **What it costs.** The model needs the charges, the periodic cell and an
  Ewald sum at inference; the local model needs neighbour lists within 3 Å.
  Here that is 1.47 against 0.45 ms/atom, almost all of it the pure-Python
  Ewald sum. In a compiled code the field is a standard $O(N\log N)$ term and
  the ratio would be far smaller; the structural cost is that the model is no
  longer local.
- **The 2b site model helps both tracks by the same amount** (0.07–0.11
  points) and for the same reason (§3.7): a weak, smooth, separately
  regularised two-body prior that lets the composite tolerate a larger
  $\sigma$. It does not interact with the field.

The gap to the published TNEP (0.85 %) is 0.11 points with the field and 0.64
without it. The non-field model is therefore not competitive with TNEP on
accuracy; it is the right model when locality is a requirement, and it shows
what the 3 Å environment alone determines: about 98.5 % of the dipole
variance.

### 3.6 Why the field column works

The single most informative experiment of the study was the decomposition of
the field column:

| Field column | Test RRMSE |
|------------------------------------------------------------------------|----------------------------|
| none | 1.67 % (fit 400) |
| bare Coulomb field of neighbours within 3.0 / 4.5 / 6.0 Å only | 1.62 / 1.66 / 1.62 % |
| Ewald field minus the 3 Å near part (beyond-cutoff only) | 1.68 % |
| **full Ewald field** | **1.17 %** |

Neither the near field at any cutoff nor the far field alone carries the
signal; only their sum does. This is dielectric screening: in liquid water the
near and far contributions to the field at a molecule largely cancel, and the
induced part of the molecular dipole responds to the small *total*. A local
basis, however rich, reconstructs only the near part and so misses the
quantity the induced dipole depends on. The column therefore adds information
that no enrichment of the local geometry can supply, which is why every local
proposal in §5 failed and this one succeeded by 30 %.

The literature says the same thing from three directions: the long-distance
equivariant (LODE) features of Grisafi and Ceriotti are Coulomb-potential
expansions of the species density, of which $\mathbf E_i$ is the $\ell=1$
point-charge member; the self-consistent-field neural network of Gao and
Remsing was built because short-range networks cannot describe dielectric
screening in water; and the 2026 water model of Cheng's group adds non-local
polarisation to local latent multipoles. The TNEP model's 6 Å radial cutoff is
the same physics approached by brute force.

Two secondary ingredients: the richer SOAP ($l=\alpha=7$, no compression) was
worth 11 % on its own because $\sigma$ pinned at the smallest value scanned,
the signature of a capacity-limited site function; and the warped Chebyshev
radial family (or Gaussians with $C=20$, $\sigma$-scale 0.5) outperformed plain
Chebyshev, wide Gaussians and spherical Bessel functions by 7–60 %.

### 3.7 The summed 2b site model and its probes

The site function of §3.1 is a single sparse kernel on the SOAP vector. The
NEP decomposition of the site energy by body order suggests the sum

$$
s_i = \sum_m \alpha_m\,(\hat{\mathbf q}_i\cdot\hat{\mathbf q}_m)^{2}
\;+\;\delta_{2b}\sum_{m'} \beta_{m'}\,(\hat{\mathbf q}^{(2)}_i\cdot\hat{\mathbf q}^{(2)}_{m'})^{2},
\qquad
q^{(2)}_{i,\,n t'} = \sum_{j\in t'} R_n(r_{ij}),
$$

a SOAP kernel plus a kernel on the radial distribution of the environment
($C=10$ warped Chebyshev channels per neighbour species, 20 numbers), each
with its own FPS sparse set ($M=250$ per species) and its own $K_{MM}$ prior,
never mixed; $\delta_{2b}$ is the relative prior amplitude. Both read out
through the same bond basis and field column. A 3b model (angular power
spectrum $q_{nn'l}$, $l\le2$) reads out through an angular kernel
$\sum_{j,k\ne j}R_aR_b\,P_l(\cos\theta_{jik})\,\mathbf d_{ij}$ instead, so that the
bond vector stays uncontracted.

Val-selected on 400/100 and refit on 500 frames (test RRMSE): SOAP 1.08 %;
**SOAP + 2b, $\delta_{2b}=0.3$, $\sigma=3\times10^{-3}$: 1.00 %**; SOAP + 3b
($\delta_{3b}=0.3$) 1.02 %; SOAP + 2b + 3b ($\delta=0.3$) 1.05 %. Alone, the
2b model reaches 1.76 % and the 3b model 2.65 %, i.e. neither replaces SOAP,
and at $\delta=1$ every sum is worse than SOAP alone (1.19–1.37 % on 400
frames against 1.17). The gain is modest (0.08 points, ≈ 3× the σ-to-σ
noise) and comes with a shift of the selected $\sigma$ from $10^{-3}$ to
$3\times10^{-3}$: a small-amplitude, smooth two-body component lets the
composite tolerate stronger regularisation, which is where the improvement
appears. It costs 2× the columns (22 000) and nothing in fit time. This is the
mechanism the TurboGAP 2b *column block* of §7 could not deliver: there the
distance features were linear columns inside the one SOAP site function; here
the 2b term is a separately regularised site function of its own.

**Probe of the 2b knobs and of multiscale** (`gap_2b_probe.py`, same
protocol, forward-pass cost from `predict_timed`). The amplitude keeps
improving down to $\delta_{2b}=0.05$–$0.1$ with $\sigma=1.5$–$2\times10^{-3}$
(test 0.96–0.97 %, val 1.07 against 1.08–1.13 for $\delta_{2b}=0.1$–$0.5$),
then plateaus. Everything that makes the 2b kernel richer makes it worse:
evaluating $q^{(2)}$ on a longer cutoff than the readout ($r_c^{(2)}=4,5,6$ Å:
val 1.23, 1.38, 1.49 % against 1.10 at 3 Å, monotone), 20 channels (1.29) or
6 (1.21), $\xi\in\{2,4\}$ (1.18), $M=500$ (1.10, no change), and the raw
polynomial kernel on the un-normalised distribution, which keeps the
neighbour count (val 1.17, test 1.00–1.01). A second SOAP scale
(`gap_extra_rcut`) does not help the dipole: alone, 4.5 Å gives 1.18–1.24 on
val but 1.21–1.33 on test (against 1.17) and 6.0 Å is clearly worse
(1.43–1.55); stacked on SOAP + 2b it lowers val to 1.04 but the test stays at
1.00 %. So the 2b term is what this section said it is — a weak, smooth, short-range
prior — and the long range of the dipole is already carried by the field
column.

*Cost.* The forward pass (neighbour lists, SOAP, 2b descriptor, geometry and
kernel columns; two processes sharing 16 cores) is 1.45 ms/atom for SOAP
alone and 1.47–1.54 ms/atom with the 2b model (+2–7 %); a 6 Å 2b cutoff costs
1.6×, a second SOAP scale at 4.5 Å 1.5× and at 6 Å 1.85×. The Ewald
field column is the largest single item: 1.47 ms/atom with it against
0.45 ms/atom without, in a pure-Python Ewald sum.

**A long readout basis does not transfer.** The published TNEP reads the
dipole out through $\partial U_i/\partial\mathbf r_{ij}$ of a learned 6 Å
two-body basis, i.e. a bond basis with learned radial functions reaching 6 Å
and site functions tied to one network. The kernel analogue — the same SOAP +
2b site functions reading out through a bond basis with $r_c=4.5$ or 6 Å,
2b descriptor kept at 3 Å — is monotonically worse: 0.97 → 1.11 → 1.41 % at
10 channels, 1.46 % at 20, while the training RRMSE falls from 0.60 to 0.03 %.
Without the field column the 6 Å readout gives 1.42 % against 1.49 % at 3 Å,
so the field's whole gain disappears once the basis can reach 6 Å: the long
bond basis fits what the field explains on the training set and does not
transfer it. With one independent site function per (channel, species) the
long basis simply has too much freedom for 500 frames; the TNEP can afford
the range because every channel is the derivative of a single 1097-parameter
scalar field. In a kernel model the long range has to enter as a fixed
physics column with one site function, which is what the field column is.

## 4. Polarisability models

### 4.1 Form

The **original model** is the first two terms below — the isotropic and
traceless bond groups, $C=10$ channels each on $r_c=4$ Å with a cosine cutoff,
site functions on the $l=\alpha=7$ compressed SOAP, $M=250$, $\sigma=0.1$,
20 000 columns: **16.6 / 19.9 %** diagonal / off-diagonal on 500 training
frames (16.9 / 20.2 on 400), 0.63 ms/atom. Every later number is measured
against it. The full model has four even kernel groups, each with its own
site function:

$$
\alpha = \sum_i\sum_{n,t'}\Big[
 s^{(n,t')}_i \sum_{j\in t'} R_n(r_{ij})\,|\mathbf d_{ij}|^2\,\mathbf I
\;+\;
 u^{(n,t')}_i \sum_{j\in t'} R_n(r_{ij})\Big(\mathbf d_{ij}\mathbf d_{ij}^\top - \tfrac{|\mathbf d_{ij}|^2}{3}\mathbf I\Big)\Big]
\;+\;
\sum_i\sum_{a\le b} w^{(ab)}_i\,\mathrm{sym}\!\big(\mathbf v^{(a)}_i\otimes\mathbf v^{(b)}_i\big)_{\text{traceless}}
\;+\;
\sum_i z_i\,\Big[\sum_j \big((\mathbf I - \mathbf A\mathbf T)^{-1}\mathbf A\big)_{ij}\Big]_{\text{traceless}} .
$$

The first two are the isotropic and traceless *bond* groups ($C=10$ channels
each, $r_c = 4$ Å, cosine cutoff); the third is the *cross* group built from
four radial channels of the dipole kernel vector $\mathbf v^{(a)}_i=\sum_j R_a(r_{ij})\mathbf d_{ij}$;
the fourth is the traceless part of the symmetrised Applequist induced-coupling
site tensor $\mathbf W_i=\tfrac12(\mathbf R_i+\mathbf R_i^\top)$,
$\mathbf R_i=\sum_j\big[(\mathbf I-\mathbf A\mathbf T)^{-1}\mathbf A\big]_{ij}$, with fixed
$\alpha_\mathrm O=1.45$ Å$^3$, $\alpha_\mathrm H=0$ and the Ewald dipole–dipole
tensor $\mathbf T$ (intramolecular pairs excluded); it is one extra column per
site function, scaled once to the RMS of the bond columns (production:
`pol_field_groups=["induced"]`, `pol_induced_alpha`, pinned `pol_field_scale`).
Site functions on the $l=\alpha=7$ compressed SOAP, $M=250$, $\sigma=0.1$,
38 500 columns. This is the recommended model: it adds a single physics column
to the base and nothing else, costs no forward time, and takes
1.9 points off the diagonal and 1.4 off the off-diagonal (§4.2).

### 4.2 Results (test set, diagonal / off-diagonal RRMSE; all fits on 500 frames unless stated)

| Model | Diagonal | Off-diagonal | ms/atom |
|----------------------------------------------------|--------------|------------------|--------------|
| Predict the component means | 100 % | 100 % | |
| Single $\mathbf d\otimes\mathbf d$ kernel (SNES, 5900 generations) | at the mean-predictor floor | | |
| **Two bond groups (iso + traceless): the original model, §4.1** | 16.6 % | 19.9 % | 0.63 |
| + cross group, 4 channels (the base) | 14.9 % | 16.0 % | 0.86 |
| **+ traceless induced-coupling group (§4.1, recommended)** | **14.0 %** | **14.6 %** | 0.80 |
| + induced coupling + field quadrupole (§4.5) | 13.9 % | 14.5 % | 0.82 |
| + all five long-range groups (§4.5) | 13.9 % | 14.3 % | 0.84 |
| + second SOAP scale at 6.5 Å, $\sigma=0.3$ (§4.4) | 13.9 % | 16.2 % | 2.5 |
| + induced coupling + quadrupole + 6.5 Å scale, $\sigma=0.3$ / $0.1$ | 13.0 % / 13.9 % | 15.2 % / 14.6 % | 2.6 |
| Published TNEP | 16.3 % | 20.4 % | |

Forward-pass cost from `predict_timed` on the 96-atom test frames, one process
on 12 threads; per-frame Python overhead dominates at this size, so only the
ratios are meaningful.

### 4.3 Why it works

**Scale separation.** A single $w\,\mathbf d\otimes\mathbf d$ bond term is purely
longitudinal: its trace $w|\mathbf d|^2$ and its anisotropy $w(\mathbf d\mathbf d^\top-|\mathbf d|^2\mathbf I/3)$
are locked to one coefficient, while a real bond has independent $\alpha_\parallel$
and $\alpha_\perp$ and the isotropic ($44\pm0.7$ Å$^3$) and anisotropic
($\pm0.5$ Å$^3$) parts of the cell tensor differ by two orders of magnitude.
Splitting the groups is the classical bond-polarisability decomposition and
mirrors GPUMD's scalar-plus-virial pair. In the ANN the split was decisive
(mean-predictor floor to 0.25 Å$^3$); in the GAP the per-channel prior already
absorbs scale, so group amplitudes and the shear weight moved results by half a
point.

**Two-neighbour terms.** Expanding $\mathbf v\otimes\mathbf v$ gives
$\sum_{j,k}R_aR_b\,\mathbf d_{ij}\otimes\mathbf d_{ik}$, the bond-angle anisotropy that
GPUMD obtains from the derivative of its angular descriptors and that bond-axis
tensors cannot represent. It saturated at four radial channels (6, 8 and 10
channels were flat or worse), so the angular information is low-dimensional.

**Long-range induced coupling.** The learning curve (24.4 / 20.9 % at 100
frames, 18.6 / 17.8 at 200, 15.3 / 16.3 at 400, 14.9 / 16.0 at 500; diagonal
$\propto N^{-0.33}$) showed the off-diagonal flattening first, which pointed at
a form limit for the anisotropy rather than a data limit. The induced
dipole–dipole coupling $\alpha_i\mathbf T_{ij}\alpha_j$ is traceless, decays as
$r^{-3}$ and reaches well beyond 4 Å. Solving it as a linear response with
fixed $\alpha_\mathrm O$ (one $3N\times3N$ solve per frame) and keeping only the
traceless part of the per-site response improved both the diagonal and the
off-diagonal; the full tensor's isotropic part competed with the isotropic bond
group and made the diagonal worse (16.7 %). The result was identical for
$\alpha_\mathrm O = 1.0$ and $1.45$ Å$^3$, i.e. the geometry of the coupling
matters and its amplitude is absorbed by the site function.

### 4.4 A second SOAP scale

The one addition that moved the polarisability in the 2026-10-20 probes is
multiscale: a second SOAP site function on the same $l=\alpha=7$ compressed
descriptor built with a longer hard cutoff (`gap_extra_rcut`), its own FPS set
and prior, reading out through the same three groups. Val-selected and refit
on 500 frames: 5.5 Å 14.2 / 16.1 %, **6.5 Å 13.9 / 16.2 %**, 7.5 Å
14.0 / 16.2 % (SOAP alone 14.9 / 16.0). The gain is on the diagonal only and
saturates at 6.5 Å; the selected $\sigma$ moves from 0.1 to 0.3; $\sigma=0.5$
is worse (14.1 / 17.2). This is consistent with §4.3: the isotropic
polarisability of a molecule depends on its wider environment (the
learning curve said the diagonal was data-limited, i.e. there was information
the 4 Å SOAP could not reach), whereas the anisotropy is a form limit that
neither more data nor a longer descriptor touches — the induced-coupling group
of §4.1 remains the only thing that has moved the off-diagonal. The 2b site
model, by contrast, hurts $\alpha$ at every amplitude and range tried
($\delta_{2b}=0.1$: 16.2 / 17.8 val; $\delta_{2b}=0.3$ on a 6 Å cutoff:
16.0 / 18.0; stacked on the 5.5 Å scale: 16.1 / 17.6).

*Cost.* 0.76 ms/atom for the base, 1.9 ms/atom with the 5.5 Å scale, 2.4 with
6.5 Å and 3.0 with 7.5 Å (the SOAP build dominates; columns double to 76 000).
Combined with the induced-coupling group (§4.5) the two gains stack on the
diagonal: 13.3 / 15.5 at $\sigma=0.3$, 13.0 / 15.2 with the quadrupole group as
well, against 14.0 / 14.6 for the same groups without the second scale. The
price is the one the whole section describes — three times the forward cost,
twice the columns — and the off-diagonal is better *without* the scale at
$\sigma=0.1$ (14.6 vs 15.2), because the two components want different
$\sigma$ once the model is this large.

### 4.5 Long-range column groups

The induced-coupling group of §4.1 is one member of a family: fixed,
parameter-free rank-2 tensors of the environment that enter as single columns
with their own site function, in the way the Ewald field enters the dipole
(§3). Five were built as production columns (`pol_field_groups`), all from the
SPC/E reference charges or the fixed Applequist response, each scaled once to
the RMS of the bond columns:

| group | tensor | columns | fit 400 → test ($\sigma=0.1$) | fit 500 → test |
|------------|------------------------------------------|----------|------------------|------------------|
| none (base) | | 38 000 | 15.3 / 16.3 | 14.9 / 16.0 |
| `induced` | $[\tfrac12(\mathbf R_i+\mathbf R_i^\top)]_{\text{traceless}}$ | +500 | **14.5 / 14.9** | **14.0 / 14.6** |
| `grad` | $[\nabla\mathbf E_i]_{\text{traceless}}$, Ewald field gradient of the charges | +500 | 15.3 / 16.2 | |
| `cross` | $(\mathbf E_i\!\cdot\!\mathbf v^{(a)}_i)\,\mathbf I$ and $\mathrm{sym}(\mathbf E_i\otimes\mathbf v^{(a)}_i)_{\text{traceless}}$, $a<4$ | +4000 | 15.4 / 16.4 | |
| `quad` | $|\mathbf E_i|^2\,\mathbf I$ and $[\mathbf E_i\otimes\mathbf E_i]_{\text{traceless}}$ | +1000 | 15.0 / 16.2 | |
| `qeq` | $-\tfrac12\sum_j L^{+}_{ij}\,\mathbf d_{ij}\otimes\mathbf d_{ij}$, charge-flow polarisability of a 6 Å QEq hardness matrix | +1000 | 15.4 / 16.4 | |
| `induced` + `quad` | | +1500 | 14.2 / 14.8 | 13.9 / 14.5 |
| all five | | +7000 | 14.4 / 14.8 | 13.9 / 14.3 |

Only the induced coupling moves the off-diagonal, and it carries almost the
whole gain; the field quadrupole adds a tenth or two, and the field gradient,
the field–bond cross and the charge-flow tensor are null within the 0.3-point
noise. Every group preferred $\sigma=0.1$ for the off-diagonal and $0.3$ for
the diagonal. The reading is the same as in §4.3: the anisotropy of liquid
water's polarisability is set by dipole–dipole coupling between molecules,
which the local basis cannot represent and the charges' own field tensors do
not carry. Ranked by gain per unit of added complexity, the induced group alone
is the model to take forward; the quadrupole is a cheap second column if the
last 0.1 matters; the three null groups should not be used.

## 5. What did not work: dipole

| Proposal | Result | Diagnosis |
|------------------------------------|--------------------------|--------------------------------------|
| Bond-angle charge flux $(\sum_j R_a\mathbf d\mathbf d^\top)\cdot\mathbf v^{(b)}$ (cross group for $\mu$) | 1.76 → 1.71 % | The one-hop basis already carries what the dipole needs from angles; $\sigma$ pinned at the floor, so the site function, not the geometry, was starved. |
| Antisymmetric bond kernel on sparse bonds, 17k–25k columns | ≤ 2 % gain | Same: bond-level site functions are not the limiter. |
| Rank-1 bilinear $a_ib_j-a_jb_i$ by alternating fits | 1.97 → 2.16 % (val), worse every step | Adds flexibility without regularising it; the convex superset above already showed no information to gain. |
| Applequist-screened field $(\mathbf I-\mathbf A\mathbf T)^{-1}\mathbf A\,\mathbf E_0$ as the column | 1.64 % (vs 1.17 for $\mathbf E_0$) | The induced response to the *permanent* field is what the site function should learn; pre-screening with a fixed $\alpha$ removes the signal. |
| Self-consistent induced field by iteration on fitted coefficients | 1.14 → 1.17 → 1.13 %; induced moments grow 1.6× per pass | Not a contraction; the literature solves induction as linear response, never by iterating a fit. |
| Gaussian-smeared (LODE-like) sources, $\sigma_s = 0.5 / 1.0$ Å | 1.17 / 1.20 % | Point charges are already right; smoothing the source removes near-field structure. |
| Ewald potential $\varphi_i$ as a product kernel $k\cdot e^{-(\varphi_i-\varphi_m)^2/2\ell^2}$ | 1.9–3.4 %, harmful | Narrows the kernel and reduces similarity between environments; the potential is a scalar and carries no direction. |
| QEq electronegativity channel $\tilde{\mathbf V}=-P L^{+}\mathbf v_{f_c}$ (verified to $10^{-14}$) | 1.69–1.70 % at $H=5/10/20$; nothing on top of the field | Charge equilibration driven by site electronegativities is what the difference form $(s_i-s_j)$ already represents. |
| QEq "polarise" (bonds built on $\mathbf d-(\mathbf y_i-\mathbf y_j)$) | 3.6–4.9 % | Over-relaxes: bond shifts of 0.6 Å RMS at $H=10$ eV/e$^2$. |
| TurboGAP 2b block (Gaussians on $r_{ij}$) | neutral at small amplitude (1.18 %), worse at $\delta_{2b}\ge1$ | See §7. |
| TurboGAP 3b block (triplet kernel) | 1.75–3.6 % at every $\delta_{3b}$, $\sigma$ and kernel tried | Train RMSE drops to $10^{-4}$: the block interpolates the training set. See §7. |
| 2b or 3b *site model* alone (own kernel, §3.7) | 1.76 % / 2.65 % | A radial distribution or an $l\le2$ angular spectrum is not a sufficient environment descriptor for $\mu$; SOAP is. |
| Summed site models at $\delta=1$ | 1.19–1.37 % (fit 400, vs 1.17) | Equal prior weight on the weaker kernel inflates the variance; only $\delta\le0.3$ helps (§3.7). |
| Spherical Bessel radial family | 1.85 % (vs 1.17 warped Chebyshev) | Oscillatory over the whole $[0, r_c]$ with large amplitude at short range. |
| More channels ($C=20$) or more sparse points ($M=500$) | 1.75 % / 1.55 % | Kernel resolution saturates; extra columns need $\sigma$ re-selection and gain nothing. |
| Bond basis widened to 4.5 / 6 Å (10 or 20 channels), SOAP + 2b site functions, ± field | 1.11 / 1.41–1.46 % with the field, 1.42–1.50 without (vs 0.97 / 1.49 at 3 Å) | Training error collapses to 0.03–0.26 %: independent site functions per channel let a 6 Å basis interpolate 500 frames. See §3.5. |

## 6. What did not work: polarisability

| Proposal | Result | Diagnosis |
|------------------------------------|--------------------------|--------------------------------------|
| Single $\mathbf d\otimes\mathbf d$ kernel | mean-predictor floor | Isotropic and anisotropic parts locked to one coefficient. |
| Group amplitudes $\delta$ and shear weight $\lambda_\mathrm{shear}$ in the GAP | ± 0.5 points | The per-channel prior already handles scale. |
| 6 Å kernel with a 6 Å kernel-only SOAP scale | 16.5 / 18.3 % (vs 16.6 / 19.9) | Modest, 4× slower, and worse when combined with the cross group without re-selecting $\sigma$ (17.4 / 20.5). |
| Cross group wider than 4 channels | flat diagonal, off-diagonal 16.0 → 17.1 % at 10 | Angular information is low-dimensional; extra columns fit noise. |
| Ewald field-gradient, field × bond and QEq charge-flow column groups (§4.5) | 15.3 / 16.2 · 15.4 / 16.4 · 15.4 / 16.4 (fit 400, base 15.3 / 16.3) | The charges' own field tensors and a hardness-matrix charge flow carry no anisotropy the base lacks; only the induced dipole–dipole coupling does. |
| Full (non-traceless) induced-coupling tensor | 16.7 / 15.2 % | Its isotropic part competes with the isotropic bond group. |
| TurboGAP 2b / 3b blocks | 23–31 % diagonal at every $\sigma$ | See §7. |
| 2b site model alone / summed with SOAP ($\delta_{2b}=0.3$; also $\delta_{2b}=0.1$, a 6 Å 2b cutoff, and stacked on a 5.5 Å second scale) | 24.2 / 19.2 % · 16.1 / 17.0 % (vs 14.9 / 16.0); val 16.0–16.2 / 17.6–18.0 for the variants | The radial distribution carries no anisotropy; added to SOAP it only dilutes the prior. |
| Second SOAP scale at 7.5 Å, or 6.5 Å with $\sigma=0.5$ | 14.0 / 16.2 % · 14.1 / 17.2 % | No better than 6.5 Å with $\sigma=0.3$ (§4.4) at 1.25× the cost. |
| 3b site model through its angular kernel, alone / + SOAP ($l_x\le1$) / + SOAP + 2b | off-diagonal 18.6 % (no isotropic group) · 14.9 / 17.0 % · 15.5 / 15.9 % | The $l=0$ member *is* the cross group already in the base; the $l=1$ member and the 2b/3b descriptors add columns (74k–112k) that the fit cannot pin down — $\sigma$ moves to 0.3. See §7. |

## 7. The 2b/3b blocks: correct, and still harmful

The blocks are faithful to gap.f90: the 2b site function is
$q^{(2)}_{i,t',s}=\sum_{j\in t'} f_c(r_{ij})\,e^{-(r_{ij}-Q_s)^2/2\sigma_2^2}$,
the 3b one the analogous sum over triplets in the TurboGAP coordinates
$(r_{ij}+r_{ik},\,(r_{ij}-r_{ik})^2,\,r_{jk})$ with the exp or compact "pol" kernel,
each with its own $K_{MM}$ built from the same kernel on its own centres; the
transcription tests against the Fortran loops pass. No block ever mixes SOAP
with distance descriptors.

They nevertheless degrade both properties, and the run records show how: the
training RMSE of a +3b fit is $10^{-4}$ against 0.061 for the base, i.e. the
block interpolates the training set. Two mechanisms were tested and ruled out.
Numerical conditioning: a 20-centre Gaussian grid at 0.16 Å spacing with
$\sigma_2=0.3$ Å has $K_{MM}$ eigenvalues at the $10^{-10}$ jitter floor, so
whitening amplifies those directions $\sim10^{5}$×; but raising the jitter to
$10^{-6}$ or $10^{-4}$, or using a well-conditioned set (8 centres, $\sigma_2=0.5$
Å; 20 triplet centres, $\sigma_3=1.0$ Å) changed nothing (μ +2b 1.22–1.35 %,
+3b 2.29–2.59 %; α +2b 21.4 / 18.6 %). Hyperparameter balance: reducing the
block amplitude ($\delta_{3b}: 1\to0.1\to0.03\to0.01$ gives 3.5 → 2.6 → 2.2 →
1.75 %) and raising $\sigma$ both move the result monotonically back towards
the base and never past it. The block therefore carries no transferable
information for these targets: it adds variance that the 1200 training rows
cannot pin down, and its genuine content, pair distances and bond angles within
3.5 Å, is exactly what a 7-7 SOAP site function already resolves. The 2b/3b
blocks earn their keep in energy GAPs because the energy has a large, smooth
two-body part that SOAP represents inefficiently; the dipole and
polarisability of water have no such part.

The follow-up scan (2026-10-20, `gap_site_models_scan.py`) tested the other
way of separating body orders: not extra columns inside one site function but
*separate site functions*, each a sparse kernel on its own descriptor (SOAP;
the radial distribution $q^{(2)}_{nt'}$; the angular power spectrum
$q^{(3)}_{nn'l}$) with its own FPS set and prior, summed with relative
amplitudes $\delta$ — the NEP ansatz transplanted to a kernel. The hypothesis
under test was that the SOAP site function already encodes the 2b and 3b
information, so the decomposition should not help. For the polarisability it
holds: val-selected and refit on 500 frames, SOAP alone 14.9 / 16.0 %,
SOAP + 2b 16.1 / 17.0, SOAP + 3b 14.9 / 17.0, SOAP + 2b + 3b 15.5 / 15.9, 2b
alone 24.2 / 19.2, 3b alone (angular kernel only, so no isotropic group)
18.6 % off-diagonal; every sum pushed the selected $\sigma$ from 0.1 to 0.3
and none beat the base beyond the ≈ 0.3-point noise. For the dipole it holds
for the 3b part (SOAP + 3b 1.02 %, within noise of SOAP + 2b + 3b at 1.05;
the μ 3b rows also carry the $T\cdot v$ cross group on the SOAP/2b readout,
$C_x=4$ forced by the 3b kernel where the base has none, which §5 found
neutral at 1.76 → 1.71 % without the field)
but a small-amplitude 2b kernel is a real, if modest, gain: 1.08 → 1.00 %
(§3.7), where the same 2b content as a column block had been neutral at best.
The difference is regularisation, not information: a separately regularised
two-body site function with a weak prior ($\delta_{2b}=0.3$) can be smooth
where the SOAP kernel is not, whereas 20 Gaussian columns inside the SOAP
site function can only interpolate. Separate body orders therefore do not add
information the 7-7 SOAP lacks; they add a second, smoother prior, which is
worth 7 % on the dipole and nothing on the polarisability.

## 8. Methodological lessons

1. **Floors before optimisers.** The convex fit on the exact readout basis
   separated "basis limited" from "optimiser limited" in a minute; the one SNES
   run of the study sat at twice the floor after 20 000 generations.
2. **Read the $\sigma$ trend.** A monotone preference for weak regularisation
   means the site function is capacity- or data-limited (dipole); a mid-range
   optimum means the basis is the limit (polarisability).
3. **Re-select $\sigma$ whenever the column set changes.** Every combination
   that skipped this was worse than its parts.
4. **Scale separation needs separate coefficients in the ANN, not in the GAP.**
5. **Validate physics code against an independent route.** The Ewald dipole
   tensor needed two sign corrections, caught only by comparing against the
   exact charge-pair field; the QEq identity was checked to $10^{-14}$ on a
   connected cluster.
6. **Concurrent scans must not share a training file or run-record directory**;
   one baseline row and the column counts of one scan were corrupted by a race.

## 9. Reproduction

Commits a32cdec … 02d0899 on `ExperimentalPolarisability`. Drivers in
`diagnostics/gap_experiments/` (README lists each script and its outcome).
Production: `gap_dipole.py` with `dipole_field`, `gap_cross_channels`,
`gap_group_delta`, `gap_lambda_shear`, `gap_site_models` / `gap_model_delta` /
`gap_site_rcut` (summed site models, §3.7), `gap_extra_rcut` (second scale,
§4.4), `pol_field_groups` / `pol_induced_alpha` / `pol_field_scale` (§4.5;
drivers `gap_site_models_scan.py`, `gap_2b_probe.py`, `gap_pol_groups_scan.py`);
experimental: `gap_bond.py` (bond kernel,
bilinear ALS), `gap_ewald.py` (Ewald field/potential, Applequist response,
induced site tensors, QEq columns). The ANN kernel inherits the field column
through `cfg.dipole_field` (staged channel on V, own head slot).
