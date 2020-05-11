######
Theory
######

This section gives a brief theoretical overview of the functions that are
implemented in bri17. For a full derivation, refer to the paper [Bri17]_.

.. note:: The preprint of this paper is freely available on the `HAL archive
	  <https://hal-enpc.archives-ouvertes.fr/hal-01304603>`_; it is not the
	  final version, but the theoretical sections were unchanged through the
	  review process.

.. note:: For the sake of consistency with the code, we use zero-based indices
	  here: index `0` is the `x`-coordinate, index `1` is the `y`-coordinate
	  and index `2` is the `z` coordinate.  We also use the bracket notation
	  (``σ[i, j]``) for indices.


Assumptions
===========

bri17 is a matrix-free implementation of the finite element method. It exposes
methods to compute the solution to linear elasticity problems, under the
following assumptions

- the number of spatial dimensions is ``DIM`` (``DIM ∈ {2, 3}``),
- the material is **homogeneous** and obeys the standard Hooke law for
  isotropic, linear elasticity: ``σ[i, j] = λ⋅ε[k, k]⋅δ[i, j] + 2⋅μ⋅ε[i, j]``
- periodic boundary conditions apply to the unit-cell ``Ω = (0, L[0]) × …
  × (0, L[DIM-1])``,
- the mesh is a uniform cartesian grid of size ``N[0] × … × N[DIM-1]``,
- each cell of the grid is a displacement-based finite element with linear shape
  functions (Q4/Q8 element).


Discrete Fourier Transform: conventions adopted in bri17
========================================================

The essential assumption is homogeneity. The loading might be very complex,
provided that each cell is made of the same material. The global stiffness
matrix of the system is then block-circulant, which allows for an efficient
formulation in the Fourier space, by means of the `discrete Fourier transform
<https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_. Conventions
adopted here regarding this transform are recalled in this section.

Let ``X[n[0], …, n[DM-1]]`` be a set of data points defined at the vertices of
the grid. Owing to periodic boundary conditions, the node indices ``n[d]`` are
such that: ``0 ≤ n[d] < N[d]`` for all ``d = 0, …, DIM-1``. Wxse will adopt the
short-hand notation ``X[n]`` where ``n`` is a multi-index. It is understood that
all multi-indices span ``{0, …, N[0]-1} × … × {0, …, N[DIM-1]-1}``.

The discrete Fourier transform ``̂DFT(X)[k]`` of ``X[n]`` is also a
``DIM``-dimensional array, where ``k`` now denotes the frequency
multi-index. Like ``n``, ``k`` spans ``{0, …, N[0]-1} × …
× {0, …, N[DIM-1]-1}``. The Fourier components are defined through the formula::

  (1)    DFT(X)[k] = ∑ X[n]⋅exp[-i⋅(φ[0] + … + φ[DIM-1])],
	             n

where the above sum extends to all multi-indices ``n`` and::

                   k[d]⋅n[d]
  (2)    φ[d] = 2π⋅─────────    (d = 0, … DIM-1),
                     L[d]

The above formula is inverted as follows::

                 1
  (3)    X[n] = ─── ∑ DFT(X)[k]⋅exp[i⋅(φ[0] + … + φ[DIM-1])],
                |N| k

where the sum now extends to all multi-indices ``k``.


The modal strain-displacement vector
====================================


The modal stiffness matrix
==========================


References
==========

.. [Bri17] Brisard, S. (2017). Reconstructing displacements from the solution to
           the periodic Lippmann–Schwinger equation discretized on a uniform
           grid. *International Journal for Numerical Methods in Engineering*,
           109(4), 459–486. https://doi.org/10.1002/nme.5263
