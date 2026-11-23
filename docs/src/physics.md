# [Physics & SDE Formulation](@id Physics-SDE)

This page introduces Parker's transport equation (PTE), explains the physical meaning and
regime of validity of each term, and derives the equivalent forward-in-time Itô stochastic
differential equation (SDE) solved by `ParticleTransport.jl`.

## Physical Regime & Assumptions

Charged energetic particles (such as solar energetic particles, anomalous cosmic rays, and
galactic cosmic rays) propagate through magnetized plasmas whose bulk flow ``\mathbf{u}``
and magnetic field ``\mathbf{B}`` vary on macroscopic length scales ``L``. When magnetic
turbulence scatters particles in pitch angle on a mean free path ``\lambda`` satisfying

```math
r_g \ll \lambda \ll L, \qquad v \gg |\mathbf{u}|,
```

where ``r_g = p / (|q| B)`` is the particle gyroradius and ``v`` is the particle speed, the
pitch-angle distribution remains nearly isotropic in the local plasma rest frame. Averaging
the Vlasov–Fokker–Planck equation over gyrophase and pitch angle yields Parker's transport
equation (Parker 1965) for the omni-directional phase-space distribution function
``f(\mathbf{x}, p, t)``.

In this regime, resolving individual gyro-orbits with a Lorentz solver like
`TestParticle.jl` becomes prohibitively expensive over macroscopic transport times
``T \gg \Omega_c^{-1}``. Instead, `ParticleTransport.jl` advances the macroscopic
distribution ``f(\mathbf{x}, p, t)`` directly via equivalent stochastic trajectories.

## Parker's Transport Equation

For an omni-directional distribution function ``f(\mathbf{x}, p, t)`` (normalized such that
the number density in ``(\mathbf{x}, p)`` is ``4\pi p^2 f(\mathbf{x}, p, t)\,dp``),
Parker's transport equation reads:

```math
\frac{\partial f}{\partial t}
+ (\mathbf{u} + \mathbf{v}_d) \cdot \nabla f
- \nabla \cdot (\boldsymbol{\kappa} \cdot \nabla f)
- \frac{1}{3} (\nabla \cdot \mathbf{u}) \, p \frac{\partial f}{\partial p}
- \frac{1}{p^2} \frac{\partial}{\partial p} \left( p^2 D_{pp} \frac{\partial f}{\partial p} \right)
= Q(\mathbf{x}, p, t).
```

Each term represents a distinct transport or acceleration process:

1. **Bulk plasma convection**, ``\mathbf{u} \cdot \nabla f``:
   Particles are frozen-in on average to the scattering centers moving with the background
   plasma flow ``\mathbf{u}(\mathbf{x}, t)``.
2. **Guiding-center magnetic drift**, ``\mathbf{v}_d \cdot \nabla f``:
   Averaging the single-particle grad-``B`` and curvature drifts,
   ```math
   \mathbf{v}_{\nabla B} + \mathbf{v}_c
   = \frac{p v}{q B^3} \left[
     \frac{1 - \mu^2}{2}\, \mathbf{B} \times \nabla B
     + \mu^2 B\, \hat{\mathbf{b}} \times (\hat{\mathbf{b}} \cdot \nabla)\hat{\mathbf{b}}
   \right],
   ```
   over an isotropic pitch-angle distribution (``\langle \mu^2 \rangle = \tfrac{1}{3}``,
   ``\langle 1 - \mu^2 \rangle = \tfrac{2}{3}``) collapses both drifts into the single
   divergence-free velocity
   ```math
   \mathbf{v}_d = \frac{p v}{3 q} \nabla \times \left( \frac{\mathbf{B}}{B^2} \right),
   \qquad \nabla \cdot \mathbf{v}_d = 0.
   ```
   Here ``v = p / \sqrt{m^2 + (p/c)^2}`` is the relativistic particle speed (reducing to
   ``v = p/m`` when ``c = \infty``). Equivalently, ``\mathbf{v}_d`` is the divergence of
   the antisymmetric (Hall) part of the full diffusion tensor.
3. **Anisotropic spatial diffusion**, ``-\nabla \cdot (\boldsymbol{\kappa} \cdot \nabla f)``:
   Pitch-angle scattering along magnetic field lines and field-line random walk across them
   produce the symmetric spatial diffusion tensor
   ```math
   \boldsymbol{\kappa}(\mathbf{x}, p, t)
   = \kappa_\perp \mathbf{I} + (\kappa_\parallel - \kappa_\perp)\,\hat{\mathbf{b}}\hat{\mathbf{b}}^T,
   \qquad \hat{\mathbf{b}} = \frac{\mathbf{B}}{|\mathbf{B}|},
   ```
   with parallel diffusion coefficient ``\kappa_\parallel \ge 0`` and perpendicular diffusion
   coefficient ``\kappa_\perp \ge 0``.
4. **Adiabatic / compressional momentum change**, ``-\frac{1}{3}(\nabla \cdot \mathbf{u})\,p\frac{\partial f}{\partial p}``:
   - In a **compressing** flow (``\nabla \cdot \mathbf{u} < 0``, such as a shock ramp or
     converging magnetic islands), particles gain momentum (**first-order Fermi / diffusive
     shock acceleration**).
   - In an **expanding** flow (``\nabla \cdot \mathbf{u} > 0``, such as the radial solar
     wind), particles lose momentum (**adiabatic cooling**).
5. **Stochastic acceleration (momentum diffusion)**,
   ``-\frac{1}{p^2}\frac{\partial}{\partial p}\left(p^2 D_{pp}\frac{\partial f}{\partial p}\right)``:
   Scattering off counter-propagating MHD waves (e.g., Alfvénic or magnetosonic turbulence)
   drives diffusion in momentum magnitude (**second-order Fermi acceleration**).
6. **Particle source**, ``Q(\mathbf{x}, p, t)``:
   Represents impulsive or continuous injection of particles into the domain.

## Derivation of the Forward-in-Time Itô SDE

To solve Parker's transport equation with Monte Carlo pseudo-particles, we must first cast
it into a **conservative Fokker–Planck equation** of the form

```math
\frac{\partial F}{\partial t}
= -\sum_\alpha \frac{\partial}{\partial z_\alpha}\bigl(A_\alpha F\bigr)
  + \frac{1}{2} \sum_{\alpha,\beta} \frac{\partial^2}{\partial z_\alpha \partial z_\beta}\bigl(B_{\alpha\beta} F\bigr)
  + \tilde{Q},
```

whose Green's function is the exact transition probability density of the Itô SDE

```math
d\mathbf{z} = \mathbf{A}(\mathbf{z}, t)\,dt + \mathbf{G}(\mathbf{z}, t)\cdot d\mathbf{W},
\qquad \mathbf{G}\mathbf{G}^T = \mathbf{B}.
```

### Step 1: Change of Variables to ``y = \ln p``

Introduce the log-momentum coordinate ``y = \ln p`` (so ``p\,\partial_p = \partial_y`` and
``dp = p\,dy``) and define the pseudo-particle number density in ``(\mathbf{x}, y)``:

```math
F(\mathbf{x}, y, t) = 4\pi p^3 f(\mathbf{x}, p, t),
\qquad dN = 4\pi p^2 f\,dp\,d^3x = F\,dy\,d^3x.
```

Using ``y = \ln p`` has two major practical advantages:
- It guarantees ``p = e^y > 0`` at all times regardless of the stochastic step size.
- It spans many orders of magnitude in momentum uniformly without numerical underflow.

### Step 2: Rewriting Each Term in Conservative Form

Multiply Parker's transport equation by ``4\pi p^3`` and rewrite each term in terms of
``F`` and ``y``:

1. **Convection, magnetic drift, and adiabatic compression**:
   Using ``\nabla \cdot \mathbf{v}_d = 0`` and
   ``4\pi p^4 \partial_p f = p\,\partial_p(4\pi p^3 f) - 3(4\pi p^3 f) = \partial_y F - 3F``:
   ```math
   4\pi p^3 (\mathbf{u} + \mathbf{v}_d)\cdot\nabla f
   = \nabla \cdot \bigl[(\mathbf{u} + \mathbf{v}_d) F\bigr] - (\nabla \cdot \mathbf{u}) F,
   ```
   ```math
   -\frac{1}{3}(\nabla\cdot\mathbf{u})\,4\pi p^4 \frac{\partial f}{\partial p}
   = -\frac{1}{3}(\nabla\cdot\mathbf{u})\frac{\partial F}{\partial y} + (\nabla\cdot\mathbf{u}) F.
   ```
   The ``\pm (\nabla\cdot\mathbf{u}) F`` terms cancel **identically**! No particle weight
   factor or killing term is required:
   ```math
   \text{Convection + Drift + Adiabatic}
   = \nabla\cdot\bigl[(\mathbf{u} + \mathbf{v}_d) F\bigr]
     + \frac{\partial}{\partial y}\left[\left(-\frac{1}{3}\nabla\cdot\mathbf{u}\right) F\right].
   ```

2. **Spatial diffusion**:
   Integrating the inner derivative by parts,
   ```math
   \sum_{i,j} \frac{\partial}{\partial x_i}\left(\kappa_{ij} \frac{\partial F}{\partial x_j}\right)
   = \sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j}(\kappa_{ij} F)
     - \sum_i \frac{\partial}{\partial x_i}\left[\left(\sum_j \frac{\partial \kappa_{ij}}{\partial x_j}\right) F\right].
   ```
   Thus spatial diffusion contributes both a second-order diffusion matrix
   ``\mathbf{B}_{xx} = 2\boldsymbol{\kappa}`` and an Itô drift velocity
   ``\nabla \cdot \boldsymbol{\kappa}`` with components
   ``(\nabla \cdot \boldsymbol{\kappa})_i = \sum_j \partial \kappa_{ij} / \partial x_j``.
   This term is nonzero whenever ``\kappa_\parallel``, ``\kappa_\perp``, or the magnetic
   field direction ``\hat{\mathbf{b}}`` varies in space; omitting it violates detailed
   balance and causes particles to drift artificially off curved field lines.

3. **Momentum diffusion**:
   Since ``4\pi p^3 \cdot p^{-2}\partial_p(p^2 D_{pp}\partial_p f) = \partial_y\bigl(p D_{pp}\,\partial_y(p^{-3}F)\bigr)``
   and ``\partial_y(p^{-3}F) = p^{-3}(\partial_y F - 3F)``:
   ```math
   \frac{\partial}{\partial y}\left[\frac{D_{pp}}{p^2}\left(\frac{\partial F}{\partial y} - 3F\right)\right]
   = \frac{\partial^2}{\partial y^2}\left(\frac{D_{pp}}{p^2} F\right)
     - \frac{\partial}{\partial y}\left[\left(\frac{\partial}{\partial y}\left(\frac{D_{pp}}{p^2}\right) + \frac{3 D_{pp}}{p^2}\right) F\right].
   ```
   Using ``\partial_y(D_{pp}/p^2) + 3 D_{pp}/p^2 = p\,\partial_p(D_{pp}/p^2) + 3 D_{pp}/p^2 = \frac{1}{p^2}\frac{\partial(p D_{pp})}{\partial p}``,
   this becomes
   ```math
   -\frac{\partial}{\partial y}\left[\left(\frac{1}{p^2}\frac{\partial(p D_{pp})}{\partial p}\right) F\right]
   + \frac{1}{2}\frac{\partial^2}{\partial y^2}\left(\frac{2 D_{pp}}{p^2} F\right).
   ```

### Step 3: Resulting Itô SDE

Collecting all terms gives the conservative Fokker–Planck equation corresponding to the
4-component Itô SDE in ``\mathbf{z} = (x, y, z, \ln p)``:

```math
d\mathbf{x} = \bigl(\mathbf{u} + \mathbf{v}_d + \nabla \cdot \boldsymbol{\kappa}\bigr)\,dt
              + \boldsymbol{\sigma} \cdot d\mathbf{W}_x,
```

```math
d(\ln p) = \left(-\frac{1}{3} \nabla \cdot \mathbf{u} + \frac{1}{p^2} \frac{\partial (p D_{pp})}{\partial p}\right) dt
           + \frac{\sqrt{2 D_{pp}}}{p}\, dW_p,
```

where ``d\mathbf{W}_x = (dW_1, dW_2, dW_3)^T`` and ``dW_p`` are independent standard Wiener
increments (``\langle dW_\alpha dW_\beta \rangle = \delta_{\alpha\beta}\,dt``).

### Exact Symmetric Square Root of ``2\boldsymbol{\kappa}``

Because ``\hat{\mathbf{b}}\hat{\mathbf{b}}^T`` is an orthogonal projection matrix
(``(\hat{\mathbf{b}}\hat{\mathbf{b}}^T)^2 = \hat{\mathbf{b}}\hat{\mathbf{b}}^T``), the
symmetric positive-semidefinite matrix

```math
\boldsymbol{\sigma}
= \sqrt{2\kappa_\perp}\,\mathbf{I}
  + \bigl(\sqrt{2\kappa_\parallel} - \sqrt{2\kappa_\perp}\bigr)\,\hat{\mathbf{b}}\hat{\mathbf{b}}^T
```

satisfies

```math
\boldsymbol{\sigma}\boldsymbol{\sigma}^T
= 2\kappa_\perp \mathbf{I}
  + \Bigl[2\sqrt{2\kappa_\perp}(\sqrt{2\kappa_\parallel} - \sqrt{2\kappa_\perp})
    + (\sqrt{2\kappa_\parallel} - \sqrt{2\kappa_\perp})^2\Bigr]\hat{\mathbf{b}}\hat{\mathbf{b}}^T
= 2\kappa_\perp \mathbf{I} + 2(\kappa_\parallel - \kappa_\perp)\hat{\mathbf{b}}\hat{\mathbf{b}}^T
= 2\boldsymbol{\kappa}
```

**identically**. Unlike Cholesky or spherical-angle factorizations (which divide by
``\sqrt{B_x^2 + B_y^2}``), this closed-form square root is smooth everywhere and completely
free of coordinate-pole singularities when ``\hat{\mathbf{b}} \parallel \pm\hat{\mathbf{z}}``.

### Recovering ``f(p, t)`` from Pseudo-Particles

Because pseudo-particles sample ``F(\mathbf{x}, \ln p, t) = 4\pi p^3 f(\mathbf{x}, p, t)``,
binning weighted pseudo-particles into momentum bins ``[p_k, p_{k+1})`` of width
``\Delta p_k = p_{k+1} - p_k``, ``\Delta(\ln p)_k = \ln(p_{k+1}/p_k)``, and geometric
center ``\bar{p}_k = \sqrt{p_k p_{k+1}}`` directly yields:

```math
\frac{dN}{d\ln p}\Big|_{\bar{p}_k} \approx \frac{\Delta N_k}{\Delta(\ln p)_k},
\qquad
f(\bar{p}_k) \approx \frac{\Delta N_k}{4\pi \bar{p}_k^2 \Delta p_k}.
```

This is exactly what [`momentum_spectrum`](@ref) and [`spectrum_from_counts`](@ref) compute.

## Canonical Analytical Solutions

The documentation includes executable benchmarks against the following exact analytical
solutions of Parker's transport equation:

| Benchmark | Active Physics | Exact Analytical Solution | Demo |
| :--- | :--- | :--- | :--- |
| **1D Convection–Diffusion** | ``u_0``, ``\kappa`` | Gaussian Green's function ``(4\pi\kappa t)^{-1/2} e^{-(x - u_0 t)^2/(4\kappa t)}`` | Convection–Diffusion |
| **Absorbing Boundary** | ``\kappa``, wall at ``x=0`` | Method of images: difference of two Gaussians | Convection–Diffusion |
| **Adiabatic Heating / Cooling** | ``u(x) = a x`` | Exact exponential momentum ``p(t) = p_0 e^{-a t / 3}`` | Convection–Diffusion |
| **1D Diffusive Shock Acceleration** | Shock ``u_1 \to u_2``, ``\kappa`` | ``f(p) \propto p^{-3r/(r-1)}`` and upstream precursor ``e^{u_1 x/\kappa}`` | 1D Diffusive Shock Acceleration |
| **Stochastic Acceleration** | ``D_{pp} = D_0 p^2`` | Log-normal ``F(y,t) = \mathcal{N}(y_0 + 3D_0 t,\, 2D_0 t)`` | Stochastic Acceleration |
| **Leaky-Box Steady State** | ``D_{pp} = D_0 p^2``, slab escape ``\tau_{\mathrm{esc}}`` | Power-law wings ``f(p) \propto p^{-3/2 \pm \sqrt{9/4 + 1/(D_0 \tau_{\mathrm{esc}})}}`` | Stochastic Acceleration |
| **3D Anisotropic & Curved-Field Diffusion** | ``\kappa_\parallel \gg \kappa_\perp``, curved ``\mathbf{B}`` | Anisotropic 3D Gaussian & exact circular invariance ``r(t) = r_0`` via ``\nabla\cdot\boldsymbol{\kappa}`` | 3D Anisotropic Diffusion |

## Focused Transport & Spherical Coordinates

In addition to Cartesian Parker transport, `ParticleTransport.jl` supports:

- **Focused Transport Equation (FTE)** (`model = FocusedTransport()`): Evolves the 5D state
  ``(x, y, z, \ln p, \mu)`` with parallel streaming ``v\mu\hat{\mathbf{b}}``, magnetic
  mirroring ``-\frac{1-\mu^2}{2} v (\hat{\mathbf{b}}\cdot\nabla\ln B)``, anisotropic flow
  compression/shear, pitch-angle diffusion ``D_{\mu\mu}``, cross-field spatial diffusion
  ``\boldsymbol{\sigma}_\perp = \sqrt{2\kappa_\perp}(\mathbf{I} - \hat{\mathbf{b}}\hat{\mathbf{b}}^T)``,
  and pitch-angle-dependent guiding-center drifts ``\mathbf{v}_d(\mu)``.
- **Spherical Coordinates** (`coords = SphericalCoordinates()`): Solves both Parker and
  Focused transport in 1D ``(r)``, 2D ``(r, \theta)``, or 3D ``(r, \theta, \phi)`` using the
  Jacobian-weighted density ``F_1 = F r^2 \sin\theta`` (Pei et al. 2010) and the
  singularity-free metric noise factorization
  ``\mathbf{P} = \operatorname{diag}(1, 1/r, 1/(r\sin\theta))\,\boldsymbol{\sigma}``.
- **Physics-Based Diffusion Models**: Quasi-linear theory (`QLTDiffusion`, Giacalone &
  Jokipii 1999), Nonlinear Guiding Center perpendicular diffusion (`NLGCDiffusion`,
  Matthaeus et al. 2003), Alfvén-wave stochastic acceleration (`AlfvenWaveMomentumDiffusion`,
  Skilling 1975), and viscous flow-shear acceleration (`FlowShearMomentumDiffusion`,
  Earl, Jokipii & Morfill 1988).

## References

- Parker, E. N. (1965). *The passage of energetic charged particles through interplanetary
  space*. Planetary and Space Science, 13(1), 9–49.
- Zhang, M. (1999). *A Markov stochastic process theory of cosmic-ray modulation*.
  The Astrophysical Journal, 513(1), 409–420.
- Pei, C., Bieber, J. W., Burger, R. A., & Clem, J. (2010). *A general time-dependent
  stochastic method for solving Parker's transport equation in spherical coordinates*.
  Journal of Geophysical Research, 115, A12107.
- Li, X., Guo, F., Li, H., & Birn, J. (2018). *The roles of fluid compression and shear in
  electron energization during magnetic reconnection*. The Astrophysical Journal, 855(2), 80.
