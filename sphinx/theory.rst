.. _20200514061549:

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

- the number of spatial dimensions is ``DIM ∈ {2, 3}``,
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
× {0, …, N[DIM-1]-1}``. The Fourier components ``X^[k]`` are defined through the
formula::

  (1)    X^[k] = DFT(X)[k] = ∑ X[n]⋅exp(-i⋅φ[n, k]),
	                   n

where the above sum extends to all multi-indices ``n`` and::

                      ┌ k[0]⋅n[0]       k[DIM-1]⋅n[DIM-1] ┐
  (2)    φ[n, k] = 2π │ ───────── + … + ───────────────── │
                      └   L[0]   	     L[DIM-1]     ┘

The above formula is inverted as follows::

                 1
  (3)    X[n] = ─── ∑ X^[k]⋅exp(i⋅φ[n, k]),
                |N| k

where the sum now extends to all multi-indices ``k``. ``|N|`` denotes the total
number of cells (product of the components of ``N``)::

  (4)    |N| = N[0] … N[DIM-1].


The modal strain-displacement vector
====================================

The nodal displacements are ``u[n, i]``, where ``n`` is the multi-index of the
node and ``i`` is the index of the component. The cell-averages of the strains
are denoted ``ε[n, i, j]``::

                       1  ⌠        1 ┌ ∂u[i]   ∂u[j] ┐
  (5)    ε[n, i, j] = ─── │        ─ │ ───── + ───── │ dx[0] … dx[DIM-1],
                      |h| ⌡cell[n] 2 └ ∂x[j]   ∂x[i] ┘

where ``|h|`` is the cell volume::

                                                L[d]
  (6)    |h| = h[0] … h[DIM-1],    where h[d] = ────.
                                                N[d]

.. _20200514061118:

In [Bri17]_, the DFT ``ε^`` of ``ε`` is expressed as follows::

                    1 ┌                                       ┐
  (7) ε^[k, i, j] = ─ │ u^[k, i]⋅B^[k, j] + u^[k, j]⋅B^[k, i] │,
                    2 └                                       ┘

where ``B^`` is the so-called modal strain-displacement vector, which is
computed (for a specified value of ``k``) by the method
:cpp:member:`Hooke\<DIM>::modal_strain_displacement`.


The modal stiffness matrix
==========================

It is recalled that the strain energy ``U`` is defined as the following integral
over the whole unit-cell ``Ω``::

             1
  (8)    U = ─ ∫ [λ⋅tr(ε)² + 2μ⋅ε:ε] dx[0] … dx[DIM-1].
             2 Ω

.. _20200514055905:

For the FE descretization considered here, the strain energy appears as a
quadratic form of the nodal displacements. This quadratic form is best expressed
in Fourier space [Bri17]_::

             1 |h|   ________
  (9)    U = ─ ─── ∑ u^[k, i]⋅K^[k, i, j]⋅u^[k, j],
             2 |N| k

where overlined quantities denote complex conjugates. ``K^`` is the *modal
stiffness matrix*. For each frequency ``k``, ``K^[k, i, j]`` is a ``DIM × DIM``
matrix. Its value is delivered by the method
:cpp:member:`Hooke\<DIM>::modal_stiffness`.

.. _20200514060358:

The strain energy is in general expressed in the real space by means of the
*nodal stiffness matrix* ``K[m, n, i, j]`` as follows::

              1
  (10)    U = ─ ∑ ∑ ∑ ∑ u[m, i]⋅K[m, n, i, j]⋅u[n, j],
              2 m n i j

where ``m`` and ``n`` span all node indices, while ``i`` and ``j`` span the
whole range of component indices. There is of course a connection between the
*modal* stiffness matrix ``K^`` and the *nodal* stiffness matrix ``K``, that is
expressed below. To do so, we introduce the following vector field, first in
Fourier space (*modal* forces)::

  (11)    F^[k, i] = |h| ∑ K^[k, i, j]⋅u^[k, j]
	                 k

then in the real space (*nodal* forces), ``F = DFT⁻¹(F^)``::

                     1
  (12)    F[n, j] = ─── ∑ F^[k, j]⋅exp(i⋅φ[n, k]),
		    |N| k

.. _20200514060430:

and Eq. :ref:`(9) <20200514055905>` reads (using Plancherel theorem)::

              1  1      ________            1
  (13)    U = ─ ─── ∑ ∑ u^[k, i]⋅F^[k, i] = ─ ∑ ∑ u[n, i]⋅F[n, i].
	      2 |N| k i                     2 n i

Comparing Eqs. :ref:`(10) <20200514060358>` and :ref:`(13) <20200514060430>`, we
find::

  (14)    F[m, i] = ∑ ∑ K[m, n, i, j]⋅u[n, j],
		    n j

which provides the link between ``K^`` and ``K``.


The case of eigenstressed materials
===================================

When the loading reduces to eigenstresses only, the boudary-value
problem to be solved reads::

  (15a)    div σ = 0
  (15b)    σ = λ⋅tr(ε)⋅I + 2μ⋅ε + ϖ
  (15c)    ε = sym grad u

where ``u`` is periodic over the unit-cell. The eigenstress ``ϖ`` is
assumed *constant in each grid-cell*; let ``ϖ[n, i, j]`` denote the
``(i, j)`` component of the eigenstress in cell ``n`` and ``ϖ^[k, i,
j]`` its discrete Fourier transform. Then, the total potential energy
of the unit-cell reads ``Π = U + U*``, where ``U`` is given by
Eq. :ref:`(9) <20200514055905>` and::

               |h|   ________             ________
  (16)    U* = ─── ∑ u^[k, i]⋅ϖ^[k, i, j]⋅B^[k, j].
               |N| k

.. _20200515061319:

Optimization of ``Π`` w.r.t. the nodal displacements delivers the
following equations (in matrix form)::

  (17)    K^[k]⋅u^[k] = -ϖ^[k]⋅B^[k].

The solution to these equations delivers the modal displacements. The
nodal displacements are then retrieved by inverse DFT.

.. note:: Eq. :ref:`(17) <20200515061319>` is singular for
          ``k = 0``. Indeed, in a periodic setting, the displacement
          is defined up to a constant translation. It is convenient to
          select the solution with zero average, that is
          ``u^[0] = 0``.

References
==========

.. [Bri17]  Brisard, S. (2017). Reconstructing displacements from the solution
           to the periodic Lippmann–Schwinger equation discretized on a uniform
           grid. *International Journal for Numerical Methods in Engineering*,
           109(4), 459–486. https://doi.org/10.1002/nme.5263
