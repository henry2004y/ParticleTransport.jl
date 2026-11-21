# Physics & SDE Formulation

## Parker's Transport Equation

For an omni-directional phase-space distribution function ``f(\mathbf{x}, p, t)``,
Parker's transport equation reads

```math
\frac{\partial f}{\partial t}
+ (\mathbf{u} + \mathbf{v}_d) \cdot \nabla f
- \nabla \cdot (\boldsymbol{\kappa} \cdot \nabla f)
- \frac{1}{3} (\nabla \cdot \mathbf{u}) \, p \frac{\partial f}{\partial p}
- \frac{1}{p^2} \frac{\partial}{\partial p} \left( p^2 D_{pp} \frac{\partial f}{\partial p} \right)
= Q(\mathbf{x}, p, t),
```

where ``\mathbf{u}`` is the plasma bulk flow velocity, ``\mathbf{v}_d = \frac{pv}{3q}\nabla \times (\mathbf{B}/B^2)``
is the guiding-center magnetic drift, ``\boldsymbol{\kappa} = \kappa_\perp \mathbf{I} + (\kappa_\parallel - \kappa_\perp)\hat{\mathbf{b}}\hat{\mathbf{b}}^T``
is the anisotropic spatial diffusion tensor, and ``D_{pp}`` is the momentum diffusion coefficient.

## Forward-in-Time Itô SDE in ``(\mathbf{x}, \ln p)``

Transforming to ``y = \ln p`` and pseudo-particle density ``F(\mathbf{x}, y, t) = p^3 f(\mathbf{x}, p, t)``
yields the conservative Fokker-Planck equation corresponding to the Itô SDE

```math
d\mathbf{x} = \left(\mathbf{u} + \mathbf{v}_d + \nabla \cdot \boldsymbol{\kappa}\right) dt + \boldsymbol{\sigma} \cdot d\mathbf{W}_x,
```

```math
d(\ln p) = \left(-\frac{1}{3} \nabla \cdot \mathbf{u} + \frac{1}{p^2} \frac{\partial (p D_{pp})}{\partial p}\right) dt + \frac{\sqrt{2 D_{pp}}}{p} dW_p,
```

where the exact symmetric positive-semidefinite square root of ``2\boldsymbol{\kappa}`` is

```math
\boldsymbol{\sigma} = \sqrt{2\kappa_\perp} \, \mathbf{I} + \left(\sqrt{2\kappa_\parallel} - \sqrt{2\kappa_\perp}\right) \hat{\mathbf{b}}\hat{\mathbf{b}}^T.
```

This representation avoids coordinate-pole singularities when ``\hat{\mathbf{b}} \parallel \pm\hat{\mathbf{z}}``,
guarantees ``p > 0`` at all times, and allows 1D/2D/3D Cartesian problems to share a uniform 4-component state vector.
